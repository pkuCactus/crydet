"""
Audio Augmentation Utilities
Supports various augmentation techniques for audio classification
"""

import random
from typing import Optional, List, Dict

import librosa
import numpy as np
import sox

from utils.config import AugmentationConfig
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

        # Preload ambient noise files if configured
        self._ambient_noise_files = self._load_ambient_noise_files()

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

    def _load_ambient_noise_files(self) -> List[str]:
        """
        Load list of ambient noise files from config.

        Returns:
            List of ambient noise file paths
        """
        noise_config = self.config.noise

        # If explicit file list is provided, use it
        if noise_config.ambient_noise_files:
            return list(noise_config.ambient_noise_files)

        # If directory is provided, scan for audio files
        if noise_config.ambient_noise_dir:
            from pathlib import Path

            noise_dir = Path(noise_config.ambient_noise_dir)
            if noise_dir.exists() and noise_dir.is_dir():
                audio_exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
                files = [
                    str(f) for f in noise_dir.rglob('*')
                    if f.suffix.lower() in audio_exts
                ]
                return files

        return []

    def _select_noise_type(self) -> str:
        """
        Select noise type based on configured probabilities.

        Returns:
            Selected noise type ('white', 'pink', 'ambient')
        """
        noise_config = self.config.noise

        # Build weighted list of noise types
        weights = []
        types = []

        if noise_config.white_noise_prob > 0:
            weights.append(noise_config.white_noise_prob)
            types.append('white')
        if noise_config.pink_noise_prob > 0:
            weights.append(noise_config.pink_noise_prob)
            types.append('pink')
        if noise_config.ambient_noise_prob > 0 and self._ambient_noise_files:
            weights.append(noise_config.ambient_noise_prob)
            types.append('ambient')

        if not types:
            return 'white'  # Default fallback

        # Normalize weights and select
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(types, weights=weights, k=1)[0]

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

        if is_aug:
            # Select effects first, then apply in pairs for memory efficiency
            available = ['pitch', 'reverb', 'phaser', 'time_stretch']
            if not is_cry:
                available.append('echo')

            selected = [e for e in available if random.random() < self.config[e]]

            # Apply effects: split into two groups only if more than 3 effects
            if len(selected) <= 3:
                y_aug = self._apply_effect_group(y_aug, selected)
            else:
                mid = len(selected) // 2
                y_aug = self._apply_effect_group(y_aug, selected[:mid])
                y_aug = self._apply_effect_group(y_aug, selected[mid:])

            y_aug = utils.pad_pcm(y_aug, y.shape[0], 1, 0)
            if random.random() < self.config.noise_prob:
                self._apply_noise(y_aug)
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

    def _apply_noise(self, y: np.ndarray) -> np.ndarray:
        """
        Apply noise to audio signal based on noise configuration.

        Args:
            y: Input audio waveform (modified in-place)

        Returns:
            Noisy audio waveform
        """
        noise_config = self.config.noise

        # Select noise type
        noise_type = self._select_noise_type()

        # Random SNR within configured range
        snr = random.uniform(noise_config.snr_min, noise_config.snr_max)

        # Apply selected noise type
        if noise_type == 'pink':
            alpha = noise_config.pink_noise_alpha
            return utils.add_pink_noise(y, snr=snr, alpha=alpha)
        elif noise_type == 'ambient' and self._ambient_noise_files:
            return utils.add_ambient_noise(
                y, self._ambient_noise_files, snr=snr, sample_rate=self.sample_rate
            )
        else:
            # Default to white noise
            return utils.add_noise(y, snr=int(snr))

    def _apply_effect(self, tfm: sox.transform.Transformer, effect_name: str) -> None:
        """Apply a single audio effect to the transformer"""
        if effect_name == 'pitch':
            pitch_rate = (random.random() - 0.5) * 8
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
        elif effect_name == 'time_stretch':
            # duration_factor: <1=shorten(faster), >1=lengthen(slower)
            duration_factor = random.uniform(0.8, 1.2)
            # stretch: SOLA good for slight changes; tempo: WSOLA good for extreme changes
            if 0.9 <= duration_factor <= 1.1:
                window = random.uniform(15, 25)
                tfm.stretch(duration_factor, window=window)
            else:
                # tempo uses speed factor (inverse of duration factor)
                speed_factor = 1.0 / duration_factor
                if 0.9 <= speed_factor <= 1.1:
                    speed_factor = 1.11
                tfm.tempo(speed_factor)
            tfm.set_output_format(rate=self.sample_rate, channels=1)

    def _apply_effect_group(self, y: np.ndarray, effects: List[str]) -> np.ndarray:
        """
        Apply a group of effects to audio.

        Args:
            y: Input audio waveform
            effects: List of effect names to apply

        Returns:
            Processed audio waveform (returns original if effects is empty)
        """
        if not effects:
            return y

        tfm = sox.transform.Transformer()
        for effect_name in effects:
            self._apply_effect(tfm, effect_name)
        return tfm.build_array(input_array=y, sample_rate_in=self.sample_rate)

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
