"""
Cry Detection Dataset and DataLoader
"""

import hashlib
import logging
import os
import pickle
import random
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
import tqdm
from torch.utils.data import Dataset

from .audio_reader import AudioReader
from .augmentation import AudioAugmenter
from .feature import FeatureExtractor
from .utils import pad_pcm
from config import DatasetConfig, AugmentationConfig, FeatureConfig

MIN_DURATION = 1.0  # Minimum duration of audio files to consider (in seconds)
# Default minimum energy threshold for cry samples (in dB, relative to max)
DEFAULT_CRY_MIN_ENERGY_DB = -40.0
LOGGER = logging.getLogger(__name__)


def compute_energy_db(waveform: np.ndarray) -> float:
    """
    Compute energy of waveform in dB (relative to full scale)

    Args:
        waveform: Audio waveform array

    Returns:
        Energy in dB (0 dB = max, negative values = quieter)
    """
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-10:
        return -100.0  # Essentially silent
    db = 20 * np.log10(rms)
    return float(db)


class CryDataset(Dataset):
    """
    Dataset for baby cry detection with low-energy cry filtering and feature extraction

    Returns pre-extracted features in [T, F] format for Transformer input:
    - T: time frames (e.g., 157 for 5s audio)
    - F: feature dimension (64/128/192 depending on delta configuration)
    """
    def __init__(
        self,
        data_dict: dict,
        config: DatasetConfig,
        aug_config: Optional[AugmentationConfig] = None,
        feat_config: Optional[FeatureConfig] = None,
        cry_min_energy_db: float = -40.0,
    ):
        self.audio_reader = AudioReader(
            target_sr=config.sample_rate,
            cache_dir=config.cache_dir,
            force_mono=config.force_mono
        )
        self.config = config
        self.data_dict = data_dict
        self.cry_min_energy_db = cry_min_energy_db

        # Initialize feature extractor
        self.feat_config = feat_config or FeatureConfig()
        self.feature_extractor = FeatureExtractor(self.feat_config)

        # Initialize augmenter if config provided
        self.augmenter: Optional[AudioAugmenter] = None
        if aug_config is not None:
            self.augmenter = AudioAugmenter(
                config=aug_config,
                sample_rate=config.sample_rate,
                audio_reader=self.audio_reader,
            )
        self.build_schedule()

    def __getitem__(self, index: tuple[str, int]):
        label, file_idx = index
        file_schedule = self.file_schedule_dict[label][file_idx]
        file_path, start_time, actual_len, need_pad = file_schedule

        # 加载音频片段
        waveform, _ = self.audio_reader.load_by_time(file_path, start_time, start_time + actual_len)

        # 如果需要补全
        if need_pad:
            target_samples = int(self.config.slice_len * self.config.sample_rate)
            waveform = pad_pcm(waveform, target_samples, pad_silence_prob=0.5, pad_front_prob=0.5, truncate=False)

        # Apply augmentation if enabled (all logic encapsulated in augmenter)
        if self.augmenter is not None:
            waveform = self.augmenter.augment(waveform, label)

        # Extract features in [T, F] format for Transformer
        features = self.feature_extractor.extract_with_deltas(waveform, self.config.sample_rate)

        return features, label

    def __len__(self) -> int:
        """Return the total number of samples per epoch based on cry rate"""
        cry_count = len(self.file_schedule_dict.get('cry', []))
        other_count = sum(
            len(schedules)
            for label, schedules in self.file_schedule_dict.items()
            if label != 'cry'
        )
        # Calculate epoch size based on the limiting class
        return int(max(
            other_count / (1 - self.config.cry_rate),
            cry_count / self.config.cry_rate
        ))

    def build_schedule(self, shuffle: bool = False, seed: Optional[int] = None):
        """Regenerate file schedules with new random slicing and energy filtering.

        Args:
            shuffle: Whether to shuffle the schedules after generation
            seed: Random seed for reproducibility across distributed processes.
                  If provided, will be used to ensure all ranks generate identical schedules.
        """
        # Set random seed if provided (for distributed training consistency)
        if seed is not None:
            random_state = random.getstate()
            random.seed(seed)

        self.file_schedule_dict = self._get_schedule_dict(self.data_dict)

        # Restore random state if we set a seed
        if seed is not None:
            random.setstate(random_state)

        if 'cry' in self.file_schedule_dict:
            original_count = len(self.file_schedule_dict['cry'])
            self.file_schedule_dict['cry'] = self._filter_low_energy_samples(
                self.file_schedule_dict['cry'],
                min_energy_db=self.cry_min_energy_db
            )
            filtered_count = len(self.file_schedule_dict['cry'])
            if filtered_count < original_count:
                LOGGER.info(f"Filtered {original_count - filtered_count} low-energy cry samples "
                           f"(threshold: {self.cry_min_energy_db} dB, remaining: {filtered_count})")

        if shuffle:
            for schedules in self.file_schedule_dict.values():
                random.shuffle(schedules)

        self._label_schedule_count = {label: len(s) for label, s in self.file_schedule_dict.items()}

        if self.augmenter is not None:
            self.augmenter.file_schedule_dict = self.file_schedule_dict

    def _filter_low_energy_samples(
        self,
        schedules: List[Tuple[str, float, float, bool]],
        min_energy_db: float
    ) -> List[Tuple[str, float, float, bool]]:
        """
        Filter out samples with energy below threshold

        Args:
            schedules: List of (file_path, start_time, actual_len, need_pad) tuples
            min_energy_db: Minimum energy threshold in dB

        Returns:
            Filtered list of schedules
        """
        valid_schedules = []

        for file_path, start_time, actual_len, need_pad in tqdm.tqdm(
            schedules, desc="Filtering low-energy cry samples", leave=False
        ):
            try:
                # Load audio segment
                waveform, _ = self.audio_reader.load_by_time(
                    file_path, start_time, start_time + actual_len
                )

                # Compute energy
                energy_db = compute_energy_db(waveform)

                if energy_db >= min_energy_db:
                    valid_schedules.append((file_path, start_time, actual_len, need_pad))
                else:
                    LOGGER.debug(f"Filtered: {file_path} @ {start_time:.2f}s (energy: {energy_db:.1f} dB)")

            except Exception as e:
                LOGGER.warning(f"Error loading {file_path} @ {start_time:.2f}s: {e}")
                continue

        return valid_schedules

    @property
    def label_schedule_count(self) -> List[str]:
        return self._label_schedule_count

    def _get_file_infos(self, data_dir: str) -> List[Tuple[str, float]]:
        """
        Get file information from directory with caching support

        Args:
            data_dir: Directory path to search for audio files

        Returns:
            List of (file_path, duration) tuples
        """
        # Generate cache file path using hash of directory path
        dir_hash = hashlib.md5(data_dir.encode()).hexdigest()[:12]
        cache_file = os.path.join(self.config.cache_dir, f"fileinfo_{dir_hash}.pkl")

        # Get directory modification time for cache validation
        dir_mtime = os.path.getmtime(data_dir)

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # Validate cache by checking directory mtime
                if cached_data.get('mtime') == dir_mtime:
                    return cached_data.get('file_infos', [])
            except Exception as e:
                LOGGER.warning(f"Cache load failed: {e}, rescanning...")

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

        # Save to cache with pickle (faster than JSON)
        if self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            try:
                cache_data = {
                    'dir': data_dir,
                    'mtime': dir_mtime,
                    'file_infos': file_infos
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                LOGGER.info(f"Saved {len(file_infos)} file infos to cache: {cache_file}")
            except IOError as e:
                LOGGER.warning(f"Failed to save cache: {e}")

        return file_infos

    def _get_file_schedule(self, file_infos: List[Tuple[str, float]]) -> List[Tuple[str, float, float, bool]]:
        """
        Generate file schedule list for sampling

        Rules:
        - duration < MIN_DURATION: skip
        - duration < slice_len: use entire file, needs padding
        - duration >= slice_len: slice with random start position using stride

        Returns:
            List of (file_path, start_time, actual_len, need_pad) tuples
        """
        file_schedules = []
        slice_len = self.config.slice_len
        stride = self.config.stride

        for file_path, duration in file_infos:
            if duration < MIN_DURATION:
                LOGGER.warning(f"Skipping {file_path} due to short duration ({duration:.3f}s)")
                continue

            if duration < slice_len:
                # Short file: use entire file, needs padding
                file_schedules.append((file_path, 0.0, duration, True))
                continue

            # Long file: slice with random start position
            current_pos = random.random() * (duration - slice_len)
            while current_pos + slice_len <= duration:
                file_schedules.append((file_path, current_pos, slice_len, False))
                current_pos += stride

            # Handle remaining segment if it's long enough
            remaining = duration - current_pos
            if remaining >= MIN_DURATION:
                file_schedules.append((file_path, current_pos, remaining, True))

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
