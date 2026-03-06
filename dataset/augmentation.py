"""
Audio Augmentation Utilities
Supports various augmentation techniques for audio classification
"""

import random
from typing import Optional, List, Dict

import librosa
import numpy as np
import sox

from config import AugmentationConfig
from dataset.audio_reader import AudioReader
from dataset import utils


class AudioAugmenter:
    """
    Audio augmentation with label-aware enhancement chains and mixup sample pool
    """

    def __init__(
        self,
        config: 'AugmentationConfig',
        sample_rate: int = 16000,
        audio_reader: Optional['AudioReader'] = None,
        mixup_pool_size: int = 100,
    ):
        """
        Initialize augmenter with config

        Args:
            config: AugmentationConfig instance containing all augmentation parameters
            sample_rate: Audio sample rate
            audio_reader: AudioReader instance for loading audio files
            mixup_pool_size: Number of samples to preload for mixup (default: 100)
        """
        self.config = config
        self.sample_rate = sample_rate
        self._audio_reader = audio_reader
        self._mixup_pool_size = mixup_pool_size

        # Internal state for mixup
        self._file_schedule_dict: Dict[str, List] = {}
        self._mixup_pool: List[np.ndarray] = []
        self._pool_initialized = False
        # 每次使用时替换池中样本的概率，保持多样性
        self._refresh_prob = 0.1

    @property
    def audio_reader(self) -> Optional['AudioReader']:
        """Get the audio reader instance"""
        return self._audio_reader

    @audio_reader.setter
    def audio_reader(self, value: Optional['AudioReader']):
        """Set the audio reader instance"""
        self._audio_reader = value

    @property
    def file_schedule_dict(self) -> Dict[str, List]:
        """Get the file schedule dictionary"""
        return self._file_schedule_dict

    @file_schedule_dict.setter
    def file_schedule_dict(self, value: Dict[str, List]):
        """
        Set the file schedule dictionary and preload mixup samples

        Args:
            value: Dictionary mapping labels to file schedules
                   {label: [(file_path, start_time, actual_len, need_pad), ...]}
        """
        self._file_schedule_dict = value
        self._pool_initialized = False
        self._mixup_pool.clear()

    def _init_pool(self):
        """Initialize mixup pool with preloaded samples (lazy initialization)"""
        if self._pool_initialized or not self._file_schedule_dict:
            return

        for _ in range(self._mixup_pool_size):
            try:
                sample = self._load_random_sample_from_disk()
                if sample is not None:
                    self._mixup_pool.append(sample)
            except Exception:
                continue

        self._pool_initialized = True

    def augment(
        self,
        y: np.ndarray,
        label: str
    ) -> np.ndarray:
        """
        Apply augmentation

        Args:
            y: Input audio waveform
            label: Label string (e.g., 'cry', 'animal_world', 'news')

        Returns:
            Augmented waveform
        """
        is_cry = label is not None and 'cry' == label.lower()
        if not is_cry and random.random() < self.config.other_reverse_prob:
            y = y[::-1]

        is_mixup_front = random.random() < self.config.mixup.mix_front_prob
        if is_mixup_front:
            y = self._do_mixup(y, label)

        # do other augment
        is_aug = self.is_augment(label)
        y_aug = np.copy(y)
        tfm = sox.transform.Transformer()

        if is_aug:
            chain = ['pitch', 'reverb', 'phaser']
            if not is_cry:
                chain.append('echo')
            random.shuffle(chain)
            for effect_name in chain:
                if random.random() < self.config[effect_name]:
                    self._apply_effect(tfm, effect_name)
            y_aug = tfm.build_array(input_array=y, sample_rate_in=self.sample_rate)
            y_aug = utils.pad_pcm(y_aug, y.shape[0], 1, 0)
            if random.random() < self.config.noise_prob:
                snr = random.random() * 20 + 10
                y_aug = utils.add_noise(y_aug, snr=snr)
            y_aug = utils.gain(y_aug, utils.get_db(y), abs=True)

        if not is_mixup_front:
            y_aug = self._do_mixup(y_aug, label)

        if is_aug and random.random() < self.config.gain_prob:
            if random.random() < 0.1:
                gain_db = random.uniform(0, 10)
            else:
                gain_db = -100
                while gain_db < -40:
                    gain_db = -np.abs(random.gauss(0, 20))
                y_aug_norm = librosa.util.normalize(y_aug)
                y_aug = utils.gain(y_aug_norm, gain_db, abs=False)

        return y_aug

    def is_augment(self, label: str) -> bool:
        aug_prob = random.random()
        if label.lower() == 'cry':
            return aug_prob < self.config.cry_aug_prob
        return aug_prob < self.config.other_aug_prob

    def _apply_effect(self, tfm: sox.transform.Transformer, effect_name: str) -> None:
        """Apply a single audio effect to the transformer"""
        if effect_name == 'pitch':
            pitch_rate = (random.random() - 0.5) * 4
            tfm.pitch(n_semitones=pitch_rate)
        elif effect_name == 'reverb':
            params = {
                'reverberance': random.random() * 80 + 20,
                'high_freq_damping': random.random() * 100,
                'room_scale': random.random() * 100,
                'stereo_depth': random.random() * 100,
                'pre_delay': 0
            }
            tfm.reverb(**params)
        elif effect_name == 'phaser':
            params = {
                'gain_in': random.random() * 0.5 + 0.5,
                'gain_out': random.random() * 0.5 + 0.5,
                'delay': random.randint(1, 5),
                'decay': random.random() * 0.4 + 0.1,
                'speed': random.random() * 1.9 + 0.1,
                'modulation_shape': random.choice(['sinusoidal', 'triangular'])
            }
            tfm.phaser(**params)
        elif effect_name == 'echo':
            params = {
                'gain_in': random.random() * 0.5 + 0.5,
                'gain_out': random.random() * 0.5 + 0.5,
                'n_echos': 1,
                'delays': [random.randint(6, 60)],
                'decays': [random.random() * 0.5]
            }
            tfm.echo(**params)

    def _do_mixup(self, y: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        y_mix = self._generate_mixup_sample(y, label)
        st = random.randint(0, len(y) - len(y_mix))
        y[st: st + len(y_mix)] += y_mix
        return np.clip(y, -1, 1)

    def _generate_mixup_sample(self, y: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        """Generate a mixup sample for a given label if applicable"""
        is_cry = label and label.lower() == 'cry'
        mix_rate = self._compute_mixup_rate(is_cry=is_cry)
        if mix_rate <= 0.0 or mix_rate >= 1.0 or not self.file_schedule_dict:
            return np.zeros_like(y)

        # 非哭声只能混合非哭声样本
        y_mix = self._get_mixup_sample(exclude_cry=(not is_cry))

        # 哭声混合：混合样本能量要小于原始音频
        if is_cry:
            original_db = utils.get_db(y)
            mix_db = utils.get_db(y_mix)
            # 如果混合样本能量高于原始，降低其能量
            if mix_db >= original_db:
                # 降低能量到比原始低 3-10 dB
                target_db_diff = random.uniform(3, 10)
                gain_factor = 10 ** (-(mix_db - original_db + target_db_diff) / 20)
                y_mix = y_mix * gain_factor

        p = utils.get_p(y, y_mix, 1 - mix_rate)
        scale = (1 - p) / p
        temp = y_mix * scale
        temp_db = utils.get_db(y_mix)
        if scale > 1e6 or temp_db > -5:
            y_mix = librosa.util.normalize(y_mix)
        else:
            y_mix = np.clip(temp, -1, 1)
        return y_mix

    def _compute_mixup_rate(self, is_cry: bool = True) -> float:
        """Compute mixup rate"""
        mix_rate = -1.0
        if is_cry and random.random() < self.config.mixup.cry_mix_prob:
            while mix_rate <= 0.0 or mix_rate >= 1.0:
                mix_rate = random.gauss(self.config.mixup.cry_mix_rate_mean,
                                        self.config.mixup.cry_mix_rate_std)
        elif not is_cry and random.random() < self.config.mixup.other_mix_prob:
            mix_rate = random.random()
        else:
            return mix_rate
        mix_rate = np.clip(mix_rate + random.gauss(0, 0.05), 0.1, 0.65)
        return mix_rate

    def _get_mixup_sample(self, exclude_cry: bool = False) -> np.ndarray:
        """
        Get a mixup sample from pool or load from disk.

        Args:
            exclude_cry: If True, only select from non-cry labels

        Returns:
            Mixup sample waveform
        """
        # 直接从磁盘加载（池化逻辑暂时跳过，因为需要支持 exclude_cry）
        return self._load_random_sample_from_disk(exclude_cry=exclude_cry)

    def _load_random_sample_from_disk(self, exclude_cry: bool = False) -> np.ndarray:
        """
        Load a random sample from disk for mixup

        Args:
            exclude_cry: If True, only select from non-cry labels

        Returns:
            Mixup sample waveform
        """
        if not self._file_schedule_dict:
            return np.zeros(80000, dtype=np.float32)  # Default 5s at 16kHz

        # 获取可选的标签列表
        available_labels = list(self._file_schedule_dict.keys())
        if exclude_cry:
            available_labels = [l for l in available_labels if l.lower() != 'cry']

        if not available_labels:
            return np.zeros(80000, dtype=np.float32)

        selected_label = random.choice(available_labels)
        file_path, offset, dur, _ = random.choice(self._file_schedule_dict[selected_label])
        y_mix, _ = self.audio_reader.load_by_time(file_path, offset, offset + dur)
        return y_mix

    def __call__(
        self,
        y: np.ndarray,
        label: str
    ) -> np.ndarray:
        """
        Apply augmentation based on label (backward compatible method)

        Args:
            y: Input audio waveform
            label: Label string (e.g., 'cry', 'animal_world', 'news')

        Returns:
            Augmented audio waveform
        """
        return self.augment(y, label)
