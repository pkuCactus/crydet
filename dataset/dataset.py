"""
Cry Detection Dataset and DataLoader
"""

import logging
import os
import random

import soundfile as sf
import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler

from .audio_reader import AudioReader
from config import DatasetConfig

MIN_DURATION = 0.02  # Minimum duration of audio files to consider (in seconds)
LOGGER = logging.getLogger(__name__)


class CryDataset(Dataset):
    """
    Dataset for baby cry detection
    """
    def __init__(
        self,
        data_dict: dict,
        config: DatasetConfig
    ):
        self.audio_reader = AudioReader(
            target_sr=config.sample_rate,
            cache_dir=config.cache_dir,
            use_cache=config.use_cache,
            force_mono=config.force_mono
        )
        self.config = config
        self.data_dict = data_dict

        # 构建文件调度字典
        self.file_schedule_dict = self._get_schedule_dict(data_dict)
        self._other_labels = [label for label in self.file_schedule_dict if label != 'cry' \
                              for _ in range(self.data_dict[label][0])]
        self._num_samples = {label: len(schedules) for label, schedules in self.file_schedule_dict.items()}

    def __getitem__(self, index: tuple[str, int]):
        label, file_idx = index
        file_schedule = self.file_schedule_dict[label][file_idx]
        file_path, start_time, slice_len = file_schedule
        waveform, _ = self.audio_reader.load_by_time(file_path, start_time, start_time + slice_len)
        return waveform, label

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
        file_infos = []
        for root, _, files in tqdm.tqdm(os.walk(data_dir)):
            for file in files:
                if file.lower().endswith(self.config.audio_suffixes):
                    file_abs_path = os.path.join(root, file)
                    try:
                        duration = sf.info(file_abs_path).duration
                    except RuntimeError:
                        print(f"Warning: Could not read audio info for {file_abs_path}, skipping.")
                        continue
                    file_infos.append((file_abs_path, duration))
        return file_infos

    def _get_file_schedule(self, file_infos: list, slice_len: int = 10, rand: int = 5):
        file_schedules = []
        for f, duration in file_infos:
            if duration < MIN_DURATION:
                LOGGER.warning(f"Skipping {f} due to short duration ({duration:.2f}s)")
                continue
            if duration < self.config.duration + 1:
                file_schedules.append((f, 0., max(duration, self.config.duration)))
            elif duration < slice_len + rand:
                s = random.random() / 2
                file_schedules.append((f, s, duration - s))
            else:
                s = random.random() / 2
                while s < duration:
                    cur_slice_len = random.randint(slice_len - rand, slice_len + rand)
                    cur_slice_len = min(cur_slice_len, duration - s)
                    if s + cur_slice_len + rand >= duration:
                        cur_slice_len = duration - s
                    file_schedules.append((f, s, cur_slice_len))
                    s += cur_slice_len
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
