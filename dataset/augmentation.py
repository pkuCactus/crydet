"""
Audio Augmentation Utilities
Supports various augmentation techniques for audio classification
"""

import random
from typing import Optional, Tuple, List, Dict

import librosa
import numpy as np
import sox
import torch
from scipy import signal

from config import AugmentationConfig
from dataset.audio_reader import AudioReader
from dataset import utils


class AudioAugmenter:
    """
    Audio augmentation with label-aware enhancement chains
    """

    def __init__(
        self,
        config: 'AugmentationConfig',
        sample_rate: int = 16000,
        audio_reader: Optional['AudioReader'] = None,
    ):
        """
        Initialize augmenter with config

        Args:
            config: AugmentationConfig instance containing all augmentation parameters
            sample_rate: Audio sample rate
            audio_reader: AudioReader instance for loading audio files
        """
        self.config = config
        self.sample_rate = sample_rate
        self._audio_reader = audio_reader

        # Internal state for mixup
        self._file_schedule_dict: Dict[str, List] = {}

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
        Set the file schedule dictionary and update internal state

        Args:
            value: Dictionary mapping labels to file schedules
                   {label: [(file_path, start_time, actual_len, need_pad), ...]}
        """
        self._file_schedule_dict = value

    def augment(
        self,
        y: np.ndarray,
        sr: int,
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
        is_mixup_front = random.random() < self.config.mixup_config.mix_front_prob
        if is_mixup_front:
            y = self._do_mixup(y)
        # do other augment
        is_aug = self.is_augment(label)
        y_db = utils.get_db(y)
        y_aug = np.copy(y)
        tfm = sox.transform.Transformer()
        def pitch():
            pitch_rate = (random.random() - 0.5) * 4
            tfm.pitch(n_semitones=pitch_rate)
        def reverb():
            params = {
                'reverberance': random.random() * 80 + 20,
                'high_freq_damping': random.random() * 100,
                'room_scale': random.random() * 100,
                'stero_depth': random.random() * 100,
                'pre_delay': 0
            }
            tfm.reverb(**params)
        def phaser():
            params = {
                'gain_in': random.random() * 0.5 + 0.5,
                'gain_out': random.random() * 0.5 + 0.5,
                'delay': random.randint(1, 5),
                'decay': random.random() * 0.4 + 0.1,
                'speed': random.random() * 1.9 + 0.1,
                'modulation_shape': random.choice(['sinusoidal', 'triangular'])
            }
            tfm.phaser(**params)
        def echo():
            params = {
                'gain_in': random.random() * 0.5 + 0.5,
                'gain_out': random.random() * 0.5 + 0.5,
                'n_echos': 1,
                'delays': [random.randint(6, 60)],
                'decays': [random.random() * 0.5]
            }
            tfm.echo(**params)
        if is_aug:
            chain = ['pitch', 'reverb', 'phaser']
            if not is_cry:
                chain.append('echo')
            random.shuffle(chain)
            for k in chain:
                if random.random() < self.config[chain]:
                    eval(chain)()
            y_aug = tfm.build_array(input_array=y, sample_rate_in=sr)
            y_aug = utils.pad_pcm(y_aug, y.shape[[0]], 1, 0)
            if random.random() < self.config.noise_prob:
                snr = random.random() * 20 + 10
                y_aug = utils.add_noise(y_aug, snr=snr)
            y_aug = utils.gain(y_aug, utils.get_db(y), abs=True)
        if not is_mixup_front:
            y_aug = self._do_mixup(y_aug)
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

    def is_augment(self, label: str):
        aug_prob = random.random()
        if label.lower() == 'cry':
            return aug_prob < self.config.cry_aug_prob
        return aug_prob < self.config.other_aug_prob

    def _do_mixup(self, y: np.ndarray):
        y_mix = self._generate_mixup_sample(y)
        st = random.randint(0, len(y) - len(y_mix))
        y[st : st + len(y_mix)] += y_mix
        return np.clip(y, -1, 1)

    def _generate_mixup_sample(self, y: np.ndarray, label: str) -> Optional[np.ndarray]:
        """Generate a mixup sample for a given label if applicable"""
        mix_rate = self._compute_mixup_rate(is_cry=(label.lower() == 'cry'))
        if mix_rate <= 0.0 or mix_rate >= 1.0:
            return np.zeros_like(y)
        y_mix = self._load_random_non_cry(len(y))
        p = utils.get_p(y, y_mix, 1 - mix_rate)
        scale = (1 - p) / p
        temp = y_mix * scale
        temp_db = utils.get_db(y_mix)
        if scale > 1e6 or temp_db > -5:
            # y_mix， y能力差异过大，或者增强的y_mix分贝过大
            y_mix = librosa.util.normalize(y_mix)
        else:
            y_mix = np.clip(temp, -1, 1)
        return y_mix

    def _compute_mixup_rate(self, is_cry: bool = True) -> float:
        """Compute mixup rate"""
        mix_rate = -1.0
        if is_cry and random.random() < self.config.mixup_config.cry_mix_prob:
            while mix_rate <= 0.0 or mix_rate >= 1.0:
                mix_rate = random.gauss(self.config.mixup_config.cry_mix_rate_mean,
                                        self.config.mixup_config.cry_mix_rate_std)
        elif not is_cry and random.random() < self.config.mixup_config.other_mix_prob:
            mix_rate = random.random()
        else:
            return mix_rate
        # 添加随机抖动，使mix_rate在0.1到0.65之间，避免完全没有mixup或完全mixup的情况
        mix_rate = np.clip(mix_rate + np.random.gauss(0, 0.05), 0.1, 0.65)
        return mix_rate

    def _load_random_sampler(self) -> np.ndarray:
        """
        Load a random sample for mixup

        Args:
            target_length: Target length in samples for padding

        Returns:
            waveform
        """
        selected_label = random.choice(list(self.file_schedule_dict.keys()))
        file_path, offset, dur, is_needed_pad = random.choice(self.file_schedule_dict[selected_label])
        y_mix, _ = self.audio_reader.load_by_time(file_path, offset, offset + dur)
        return y_mix

    def _normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform to prevent clipping"""
        max_val = np.max(np.abs(waveform))
        if max_val > 1.0:
            waveform = waveform / max_val
        return waveform.astype(np.float32)

    def __call__(
        self,
        y: np.ndarray,
        label: str
    ) -> np.ndarray:
        """
        Apply augmentation based on label (backward compatible method)

        Args:
            waveform: Input audio waveform
            label: Label string (e.g., 'cry', 'animal_world', 'news')
            non_cry_waveform: Optional non-cry waveform for mixup

        Returns:
            augmented waveform: The augmented audio
        """
        return self.augment(y, label)

    def _sample_mix_rate(self, max_attempts: int = 100) -> float:
        """
        Sample mix_rate from Gaussian distribution, ensuring it's in (0, 1)

        Args:
            max_attempts: Maximum number of resampling attempts

        Returns:
            mix_rate in (0, 1)
        """
        for _ in range(max_attempts):
            mix_rate = np.random.normal(self.cry_mix_mean, self.cry_mix_std)
            if 0.0 < mix_rate < 1.0:
                return mix_rate
        # Fallback to mean if we can't sample a valid value
        return max(0.01, min(0.99, self.cry_mix_mean))

    def _apply_effects(self, waveform: np.ndarray, effects: List[Tuple[str, float]]) -> np.ndarray:
        """
        Apply effects with probabilities in random order

        Args:
            waveform: Input waveform
            effects: List of (effect_name, probability) tuples

        Returns:
            Augmented waveform
        """
        effects = effects.copy()
        random.shuffle(effects)

        augmented = waveform.copy()
        for effect, prob in effects:
            if np.random.random() < prob:
                if effect == 'pitch':
                    augmented = self.pitch_shift_audio(augmented)
                elif effect == 'reverb':
                    augmented = self.add_reverb(augmented)
                elif effect == 'phaser':
                    augmented = self.add_phaser(augmented)
                elif effect == 'echo':
                    augmented = self.add_echo(augmented)

        return augmented

    def apply_mixup(
        self,
        cry_waveform: np.ndarray,
        non_cry_waveform: np.ndarray,
        mix_rate: float
    ) -> np.ndarray:
        """
        Apply mixup between cry and non-cry waveforms

        Args:
            cry_waveform: Cry sample waveform
            non_cry_waveform: Non-cry sample waveform
            mix_rate: Mixing coefficient (in 0-1)

        Returns:
            Mixed waveform
        """
        # Ensure same length
        min_len = min(len(cry_waveform), len(non_cry_waveform))
        mixed = mix_rate * cry_waveform[:min_len] + (1 - mix_rate) * non_cry_waveform[:min_len]

        # Pad if needed
        if len(cry_waveform) > min_len:
            mixed = np.concatenate([mixed, cry_waveform[min_len:]])

        return mixed.astype(cry_waveform.dtype)

    def pitch_shift_audio(
        self,
        waveform: np.ndarray,
        n_steps: Optional[float] = None
    ) -> np.ndarray:
        """
        Shift pitch of audio

        Args:
            waveform: Input audio waveform
            n_steps: Number of semitones to shift

        Returns:
            Pitch-shifted waveform
        """
        if n_steps is None:
            n_steps = np.random.uniform(-self.pitch_shift, self.pitch_shift)

        if abs(n_steps) < 0.1:
            return waveform

        if HAS_LIBROSA:
            try:
                shifted = librosa.effects.pitch_shift(
                    waveform,
                    sr=self.sample_rate,
                    n_steps=n_steps
                )
                return shifted.astype(waveform.dtype)
            except Exception:
                pass

        # Fallback: resampling-based pitch shift
        factor = 2 ** (n_steps / 12.0)
        new_length = int(len(waveform) / factor)
        if new_length < 100:
            return waveform

        resampled = signal.resample(waveform, new_length)

        if len(resampled) < len(waveform):
            pad_len = len(waveform) - len(resampled)
            resampled = np.pad(resampled, (0, pad_len), mode='constant')
        else:
            resampled = resampled[:len(waveform)]

        return resampled.astype(waveform.dtype)

    def add_reverb(
        self,
        waveform: np.ndarray,
        room_scale: Optional[float] = None,
        decay: Optional[float] = None
    ) -> np.ndarray:
        """
        Add reverb effect using delayed echoes

        Args:
            waveform: Input audio waveform
            room_scale: Room size (0-1)
            decay: Decay factor (0-1)

        Returns:
            Waveform with reverb
        """
        if room_scale is None:
            room_scale = np.random.uniform(0.1, 0.4)
        if decay is None:
            decay = np.random.uniform(0.2, 0.4)

        augmented = waveform.copy()

        # Multiple delayed echoes
        delay_times = [0.02, 0.04, 0.06, 0.10]
        decay_factors = [decay, decay * 0.7, decay * 0.5, decay * 0.3]

        for delay_time, decay_factor in zip(delay_times, decay_factors):
            delay_samples = int(delay_time * self.sample_rate)
            if delay_samples < len(waveform):
                delayed = np.zeros_like(waveform)
                delayed[delay_samples:] = waveform[:-delay_samples] * decay_factor
                augmented = augmented + delayed

        return augmented.astype(waveform.dtype)

    def add_phaser(
        self,
        waveform: np.ndarray,
        rate: Optional[float] = None,
        depth: Optional[float] = None
    ) -> np.ndarray:
        """
        Add phaser effect using modulated all-pass filters

        Args:
            waveform: Input audio waveform
            rate: LFO rate in Hz (default: 0.5-2.0)
            depth: Modulation depth (default: 0.3-0.7)

        Returns:
            Waveform with phaser effect
        """
        if rate is None:
            rate = np.random.uniform(0.5, 2.0)
        if depth is None:
            depth = np.random.uniform(0.3, 0.6)

        length = len(waveform)
        t = np.arange(length) / self.sample_rate

        # Create LFO (low frequency oscillator)
        lfo = 0.5 * (1 + np.sin(2 * np.pi * rate * t))

        # Modulated delay (all-pass simulation)
        max_delay = int(0.002 * self.sample_rate)  # 2ms max delay
        min_delay = int(0.0005 * self.sample_rate)  # 0.5ms min delay

        augmented = waveform.copy()

        for stage in range(4):  # 4-stage phaser
            # Calculate modulated delay for each sample
            modulated_delay = min_delay + (max_delay - min_delay) * lfo * depth

            # Apply variable delay (simplified)
            delay_samples = int(np.mean(modulated_delay))

            if delay_samples > 0 and delay_samples < length:
                delayed = np.zeros_like(waveform)
                delayed[delay_samples:] = waveform[:-delay_samples]
                # Mix with feedback
                augmented = augmented + depth * 0.3 * delayed

        return augmented.astype(waveform.dtype)

    def add_echo(
        self,
        waveform: np.ndarray,
        delay_time: Optional[float] = None,
        decay: Optional[float] = None,
        num_echoes: Optional[int] = None
    ) -> np.ndarray:
        """
        Add echo effect

        Args:
            waveform: Input audio waveform
            delay_time: Delay time in seconds (default: 0.1-0.3)
            decay: Echo decay factor (default: 0.3-0.6)
            num_echoes: Number of echoes (default: 2-4)

        Returns:
            Waveform with echo
        """
        if delay_time is None:
            delay_time = np.random.uniform(0.1, 0.3)
        if decay is None:
            decay = np.random.uniform(0.3, 0.6)
        if num_echoes is None:
            num_echoes = np.random.randint(2, 5)

        augmented = waveform.copy()
        delay_samples = int(delay_time * self.sample_rate)

        for i in range(1, num_echoes + 1):
            echo_delay = delay_samples * i
            echo_decay = decay ** i

            if echo_delay < len(waveform):
                echo = np.zeros_like(waveform)
                echo[echo_delay:] = waveform[:-echo_delay] * echo_decay
                augmented = augmented + echo

        return augmented.astype(waveform.dtype)


def mixup(
    waveform1: np.ndarray,
    waveform2: np.ndarray,
    label1: int,
    label2: int,
    alpha: float = 0.4
) -> Tuple[np.ndarray, int]:
    """
    Mixup augmentation for two waveforms (same length expected)

    - Label follows OR logic: if either is cry (1), result is cry (1)

    Args:
        waveform1: First waveform
        waveform2: Second waveform (same length as waveform1)
        label1: Label of first waveform (0 or 1)
        label2: Label of second waveform (0 or 1)
        alpha: Beta distribution parameter for mixup

    Returns:
        Tuple of (mixed waveform, mixed label)
    """
    # Sample mixup coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Mix waveforms
    mixed = lam * waveform1 + (1 - lam) * waveform2

    # OR logic for label: if either is cry, result is cry
    mixed_label = label1 | label2  # Assuming labels are binary (0 or 1)

    return mixed.astype(waveform1.dtype), mixed_label


def mixup_batch(
    waveforms: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch-level mixup augmentation (all waveforms same length)

    Args:
        waveforms: Batch of waveforms, shape (batch_size, seq_len)
        labels: Batch of labels, shape (batch_size,) - integer class indices (0 or 1)
        alpha: Beta distribution parameter for mixup

    Returns:
        Tuple of (mixed waveforms, mixed labels)
        - mixed waveforms: shape (batch_size, seq_len)
        - mixed labels: shape (batch_size,) - OR of original labels
    """
    batch_size = len(waveforms)

    # Sample mixup coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Shuffle indices
    shuffle_idx = np.random.permutation(batch_size)

    # Mix waveforms
    mixed_waveforms = lam * waveforms + (1 - lam) * waveforms[shuffle_idx]

    # OR logic for labels
    mixed_labels = np.maximum(labels, labels[shuffle_idx])

    return mixed_waveforms.astype(waveforms.dtype), mixed_labels


class MixupCollateFn:
    """
    Collate function with mixup augmentation for DataLoader

    Binary classification: cry (1) vs non-cry (0)
    - Any label containing 'cry' is treated as positive class
    - Mixup label follows OR logic: if either sample is cry, result is cry
    - All waveforms expected to have same length (padded by dataset)

    Example:
        >>> collate_fn = MixupCollateFn(alpha=0.4, mixup_prob=0.5)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> for waveforms, labels in dataloader:
        ...     loss = criterion(model(waveforms), labels)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        mixup_prob: float = 0.5
    ):
        """
        Initialize MixupCollateFn

        Args:
            alpha: Beta distribution parameter for mixup (default: 0.4)
            mixup_prob: Probability of applying mixup to each batch (default: 0.5)
        """
        self.alpha = alpha
        self.mixup_prob = mixup_prob

    def _label_to_idx(self, label: str) -> int:
        """Convert string label to binary index: cry=1, others=0"""
        return 1 if 'cry' in label.lower() else 0

    def __call__(
        self,
        batch: List[Tuple[np.ndarray, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a batch of samples with optional mixup

        Args:
            batch: List of (waveform, label) tuples

        Returns:
            (waveforms, labels)
            - waveforms: torch.Tensor, shape (batch_size, seq_len)
            - labels: torch.Tensor, shape (batch_size,) - binary labels (0 or 1)
        """
        # Separate waveforms and labels
        waveforms = [item[0] for item in batch]
        labels_str = [item[1] for item in batch]

        # Stack waveforms (all same length from dataset)
        waveforms_stacked = np.stack(waveforms, axis=0).astype(np.float32)

        # Convert string labels to binary indices (cry=1, non-cry=0)
        labels = np.array([self._label_to_idx(label) for label in labels_str], dtype=np.int64)

        # Decide whether to apply mixup
        if np.random.random() < self.mixup_prob and self.alpha > 0:
            mixed_waveforms, mixed_labels = mixup_batch(
                waveforms_stacked, labels, self.alpha
            )
            return (
                torch.from_numpy(mixed_waveforms),
                torch.from_numpy(mixed_labels)
            )
        else:
            return (
                torch.from_numpy(waveforms_stacked),
                torch.from_numpy(labels)
            )


def spec_augment(
    features: np.ndarray,
    time_mask_param: int = 40,
    freq_mask_param: int = 20,
    num_time_masks: int = 2,
    num_freq_masks: int = 2
) -> np.ndarray:
    """
    SpecAugment: masking in the feature domain

    Args:
        features: Input features of shape (feature_dim, time)
        time_mask_param: Maximum time mask length
        freq_mask_param: Maximum frequency mask length
        num_time_masks: Number of time masks to apply
        num_freq_masks: Number of frequency masks to apply

    Returns:
        Augmented features
    """
    augmented = features.copy()
    feature_dim, time_len = features.shape

    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param, time_len))
        t0 = np.random.randint(0, time_len - t)
        augmented[:, t0:t0 + t] = 0

    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, feature_dim))
        f0 = np.random.randint(0, feature_dim - f)
        augmented[f0:f0 + f, :] = 0

    return augmented


# Standalone functions for backward compatibility
def add_noise(waveform: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add random noise to the waveform (standalone function)

    Args:
        waveform: Input audio waveform
        noise_factor: Scaling factor for noise

    Returns:
        Noisy waveform
    """
    noise = np.random.randn(len(waveform))
    augmented = waveform + noise_factor * noise
    return augmented.astype(waveform.dtype)
