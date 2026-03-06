"""
Cry Detection Dataset and DataLoader
"""

import hashlib
import json
import logging
import os
import random
from typing import Optional

import numpy as np
import soundfile as sf
import tqdm
from torch.utils.data import Dataset

from .audio_reader import AudioReader
from .augmentation import AudioAugmenter
from config import DatasetConfig

MIN_DURATION = 1.0  # Minimum duration of audio files to consider (in seconds)
LOGGER = logging.getLogger(__name__)


class CryDataset(Dataset):
    """
    Dataset for baby cry detection
    """
    def __init__(
        self,
        data_dict: dict,
        config: DatasetConfig,
        use_augmentation: bool = False
    ):
        self.audio_reader = AudioReader(
            target_sr=config.sample_rate,
            cache_dir=config.cache_dir,
            force_mono=config.force_mono
        )
        self.config = config
        self.data_dict = data_dict
        self.use_augmentation = use_augmentation

        # 构建文件调度字典
        self._get_schedule_dict()

        # Initialize augmenter if enabled
        self.augmenter: Optional[AudioAugmenter] = None
        if config.aug_config is not None:
            self.augmenter = AudioAugmenter(
                config=config.aug_config,
                sample_rate=config.sample_rate,
                audio_reader=self.audio_reader,
            )
            # Pass file schedule dict to augmenter for mixup
            self.augmenter.file_schedule_dict = self.file_schedule_dict

    def __getitem__(self, index: tuple[str, int]):
        label, file_idx = index
        file_schedule = self.file_schedule_dict[label][file_idx]
        file_path, start_time, actual_len, need_pad = file_schedule

        # 加载音频片段
        waveform, _ = self.audio_reader.load_by_time(file_path, start_time, start_time + actual_len)

        # 如果需要补全
        if need_pad:
            target_samples = int(self.config.slice_len * self.config.sample_rate)
            waveform = self._pad_waveform(waveform, target_samples)

        # Apply augmentation if enabled (all logic encapsulated in augmenter)
        if self.augmenter is not None:
            waveform = self.augmenter.augment(waveform, label)

        return waveform, label

    def _pad_waveform(self, waveform: np.ndarray, target_length: int) -> np.ndarray:
        """
        补全波形到目标长度 (随机补零或噪声)

        Args:
            waveform: 原始波形
            target_length: 目标长度(采样点数)

        Returns:
            补全后的波形
        """
        current_length = len(waveform)
        pad_length = target_length - current_length

        if pad_length <= 0:
            return waveform

        # 随机选择补零或补噪声
        if random.random() < 0.5:
            padding = np.zeros(pad_length, dtype=np.float32)
        else:
            noise_level = 0.01 * np.std(waveform) if np.std(waveform) > 0 else 0.001
            padding = np.random.randn(pad_length).astype(np.float32) * noise_level

        # 随机选择补在前面还是后面
        if random.random() < 0.5:
            waveform = np.concatenate([padding, waveform])
        else:
            waveform = np.concatenate([waveform, padding])

        return waveform

    def __len__(self):
        others = sum(len(schedules) for label, schedules in self.file_schedule_dict.items() if label != 'cry')
        cry = len(self.file_schedule_dict.get('cry', []))
        return int(max(others / (1 - self.config.cry_rate), cry / self.config.cry_rate))

    def generate_schedule(self):
        """Regenerate file schedules for the next epoch"""
        self.file_schedule_dict = self._get_schedule_dict(self.data_dict)
        self._other_labels = [label for label in self.file_schedule_dict if label != 'cry' \
                              for _ in range(self.data_dict[label][0])]
        self._num_samples = {label: len(schedules) for label, schedules in self.file_schedule_dict.items()}
        # Sync with augmenter
        if self.augmenter is not None:
            self.augmenter.file_schedule_dict = self.file_schedule_dict

    def shuffle(self):
        """Shuffle file schedules for each label"""
        for label in self.file_schedule_dict:
            random.shuffle(self.file_schedule_dict[label])

    @property
    def other_labels(self):
        return self._other_labels

    @property
    def num_samples(self):
        return self._num_samples

    def _get_file_infos(self, data_dir: str) -> list:
        """
        Get file information from directory with caching support

        Args:
            data_dir: Directory path to search for audio files

        Returns:
            List of (file_path, duration) tuples
        """
        # Generate cache file path based on directory and config
        cache_key = f"{data_dir}_{self.config.audio_suffixes}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_file = os.path.join(self.config.cache_dir, f"{data_dir}_{cache_hash}.json")

        # Try to load from cache
        if self.config.use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                return cached_data
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to load cache: {e}, rescanning...")

        # Scan directory for audio files
        file_infos = []
        for root, _, files in tqdm.tqdm(os.walk(data_dir), desc=f"Scanning {data_dir}"):
            for file in files:
                if not file.lower().endswith(self.config.audio_suffixes):
                    continue
                file_abs_path = os.path.join(root, file)
                try:
                    duration = sf.info(file_abs_path).duration
                except RuntimeError:
                    LOGGER.warning(f"Could not read audio info for {file_abs_path}, skipping.")
                    continue
                file_infos.append((file_abs_path, duration))

        # Save to cache
        if self.config.use_cache and self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(file_infos, f, ensure_ascii=False, indent=2)
                LOGGER.info(f"Saved {len(file_infos)} file infos to cache: {cache_file}")
            except IOError as e:
                raise ValueError(f"Failed to save cache: {e}")

        return file_infos

    def _get_file_schedule(self, file_infos: list) -> list:
        """
        生成文件调度列表

        规则:
        - duration < MIN_DURATION: 跳过
        - duration < slice_len: 取整个文件, 需要 pad
        - duration >= slice_len: 随机选择起始位置 [0, 1), 按 stride 切分

        Returns:
            [(file_path, start_time, actual_len, need_pad), ...]
        """
        file_schedules = []
        slice_len = self.config.slice_len
        stride = self.config.stride

        for file_path, duration in file_infos:
            # 跳过过短的音频
            if duration < MIN_DURATION:
                LOGGER.warning(f"Skipping {file_path} due to short duration ({duration:.3f}s)")
                continue

            if duration < slice_len:
                # 短于 slice_len: 取整个文件, 需要 pad
                file_schedules.append((file_path, 0.0, duration, True))

            else:
                # 长于等于 slice_len: 基于随机起始位置，按 stride 切分
                s = random.random()
                while s + slice_len <= duration:
                    file_schedules.append((file_path, s, slice_len, False))
                    s += stride
                if s < duration and duration - s >= MIN_DURATION:
                    # 最后一个片段不足 slice_len, 取剩余部分, 需要 pad
                    file_schedules.append((file_path, s, duration - s, True))

        return file_schedules

    def _get_schedule_dict(self, dataset_dict: dict):
        file_schedule_dict = {}
        for label, dir_list in dataset_dict.items():
            # non-cry duplicate directories for better sampling balance
            dir_list = dir_list[1:] if label != 'cry' else dir_list
            file_infos = []
            for dir_ in dir_list:
                file_infos.extend(self._get_file_infos(dir_))
            file_schedule_dict[label] = self._get_file_schedule(file_infos)
            if not file_schedule_dict[label]:
                raise ValueError(f"No valid audio files found for label '{label}' in directories: {dir_list}")
        return file_schedule_dict
