"""
Audio Augmentation Utilities
Supports various augmentation techniques for audio classification
"""

import numpy as np
import torch
import random
from typing import Optional, Tuple, List, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .audio_reader import AudioReader

from scipy import signal

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioAugmenter:
    """
    Audio augmentation with label-aware enhancement chains

    Cry samples:
        - cry_aug_rate: probability of applying pitch/reverb/phaser
        - cry_mix_rate: probability of mixing with non-cry sample
        - When mixing, mix_rate is sampled from Gaussian(cry_mix_mean, cry_mix_std)

    Non-cry samples:
        - non_cry_aug_rate: probability of applying pitch/reverb/phaser/echo

    The augmenter encapsulates all augmentation logic internally, including
    loading non-cry samples for mixup when needed.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        audio_reader: Optional['AudioReader'] = None,
        # Cry augmentation
        cry_aug_rate: float = 0.5,
        cry_mix_rate: float = 0.3,
        cry_mix_mean: float = 0.5,
        cry_mix_std: float = 0.15,
        # Non-cry augmentation
        non_cry_aug_rate: float = 0.5,
        # Pitch shift range
        pitch_shift: float = 2.0,
        # Individual effect probabilities
        pitch_prob: float = 0.5,
        reverb_prob: float = 0.5,
        phaser_prob: float = 0.5,
        echo_prob: float = 0.5
    ):
        """
        Initialize augmenter

        Args:
            sample_rate: Audio sample rate
            audio_reader: AudioReader instance for loading audio files
            cry_aug_rate: Probability of applying augmentation to cry samples
            cry_mix_rate: Probability of mixing cry with non-cry sample
            cry_mix_mean: Mean of Gaussian distribution for mix_rate
            cry_mix_std: Std of Gaussian distribution for mix_rate
            non_cry_aug_rate: Probability of applying augmentation to non-cry samples
            pitch_shift: Pitch shift range in semitones
            pitch_prob: Probability of applying pitch shift
            reverb_prob: Probability of applying reverb
            phaser_prob: Probability of applying phaser
            echo_prob: Probability of applying echo
        """
        self.sample_rate = sample_rate
        self._audio_reader = audio_reader
        self.cry_aug_rate = cry_aug_rate
        self.cry_mix_rate = cry_mix_rate
        self.cry_mix_mean = cry_mix_mean
        self.cry_mix_std = cry_mix_std
        self.non_cry_aug_rate = non_cry_aug_rate
        self.pitch_shift = pitch_shift
        self.pitch_prob = pitch_prob
        self.reverb_prob = reverb_prob
        self.phaser_prob = phaser_prob
        self.echo_prob = echo_prob

        # Internal state for mixup
        self._file_schedule_dict: Dict[str, List] = {}
        self._non_cry_labels: List[str] = []
        self._num_samples: Dict[str, int] = {}
        self._slice_len: float = 1.0  # Default slice length in seconds

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
        self._non_cry_labels = [label for label in value if label != 'cry']
        self._num_samples = {label: len(schedules) for label, schedules in value.items()}

    def set_slice_len(self, slice_len: float):
        """Set the slice length for padding"""
        self._slice_len = slice_len

    def augment(
        self,
        waveform: np.ndarray,
        label: str
    ) -> np.ndarray:
        """
        Apply augmentation based on label (encapsulated method)

        This method handles all augmentation logic internally, including
        loading non-cry samples for mixup when needed.

        Args:
            waveform: Input audio waveform
            label: Label string (e.g., 'cry', 'animal_world', 'news')

        Returns:
            Augmented waveform
        """
        is_cry = label is not None and 'cry' in label.lower()

        if is_cry and self._non_cry_labels and self._audio_reader is not None:
            # Try mixup for cry samples
            if np.random.random() < self.cry_mix_rate:
                # Load a random non-cry sample
                non_cry_waveform = self._load_random_non_cry(waveform.shape[0])
                if non_cry_waveform is not None:
                    mix_rate = self._sample_mix_rate()
                    # Apply effects first
                    if np.random.random() < self.cry_aug_rate:
                        effects = [
                            ('pitch', self.pitch_prob),
                            ('reverb', self.reverb_prob),
                            ('phaser', self.phaser_prob)
                        ]
                        waveform = self._apply_effects(waveform, effects)
                    # Apply mixup
                    waveform = self.apply_mixup(waveform, non_cry_waveform, mix_rate)
                    return self._normalize(waveform)

            # No mixup, just apply effects
            if np.random.random() < self.cry_aug_rate:
                effects = [
                    ('pitch', self.pitch_prob),
                    ('reverb', self.reverb_prob),
                    ('phaser', self.phaser_prob)
                ]
                waveform = self._apply_effects(waveform, effects)
        else:
            # Non-cry augmentation
            if np.random.random() < self.non_cry_aug_rate:
                effects = [
                    ('pitch', self.pitch_prob),
                    ('reverb', self.reverb_prob),
                    ('phaser', self.phaser_prob),
                    ('echo', self.echo_prob)
                ]
                waveform = self._apply_effects(waveform, effects)

        return self._normalize(waveform)

    def _load_random_non_cry(self, target_length: int) -> Optional[np.ndarray]:
        """
        Load a random non-cry sample for mixup

        Args:
            target_length: Target length in samples for padding

        Returns:
            Non-cry waveform or None if not available
        """
        if not self._non_cry_labels or not self._audio_reader or not self._file_schedule_dict:
            return None

        try:
            non_cry_label = random.choice(self._non_cry_labels)
            num_samples = self._num_samples.get(non_cry_label, 0)
            if num_samples == 0:
                return None

            non_cry_idx = random.randint(0, num_samples - 1)
            non_cry_schedule = self._file_schedule_dict[non_cry_label][non_cry_idx]
            non_cry_path, non_cry_start, non_cry_len, non_cry_need_pad = non_cry_schedule

            non_cry_waveform, _ = self._audio_reader.load_by_time(
                non_cry_path, non_cry_start, non_cry_start + non_cry_len
            )

            if non_cry_need_pad:
                non_cry_waveform = self._pad_waveform(non_cry_waveform, target_length)

            return non_cry_waveform
        except Exception:
            return None

    def _pad_waveform(self, waveform: np.ndarray, target_length: int) -> np.ndarray:
        """Pad waveform to target length"""
        current_length = len(waveform)
        pad_length = target_length - current_length

        if pad_length <= 0:
            return waveform

        # Random choice: pad with zeros or noise
        if random.random() < 0.5:
            padding = np.zeros(pad_length, dtype=np.float32)
        else:
            noise_level = 0.01 * np.std(waveform) if np.std(waveform) > 0 else 0.001
            padding = np.random.randn(pad_length).astype(np.float32) * noise_level

        # Random choice: pad at start or end
        if random.random() < 0.5:
            waveform = np.concatenate([padding, waveform])
        else:
            waveform = np.concatenate([waveform, padding])

        return waveform

    def _normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform to prevent clipping"""
        max_val = np.max(np.abs(waveform))
        if max_val > 1.0:
            waveform = waveform / max_val
        return waveform.astype(np.float32)

    def __call__(
        self,
        waveform: np.ndarray,
        label: str = None,
        non_cry_waveform: np.ndarray = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Apply augmentation based on label (backward compatible method)

        Args:
            waveform: Input audio waveform
            label: Label string (e.g., 'cry', 'animal_world', 'news')
            non_cry_waveform: Optional non-cry waveform for mixup

        Returns:
            Tuple of (augmented waveform, needs_mixup, mix_rate)
            - augmented waveform: The augmented audio
            - needs_mixup: Whether mixup is needed (only for cry samples)
            - mix_rate: The mixing rate (sampled from Gaussian, in (0,1))
        """
        is_cry = label is not None and 'cry' in label.lower()

        needs_mixup = False
        mix_rate = 0.0

        if is_cry:
            # Check if mixup is needed
            if non_cry_waveform is not None and np.random.random() < self.cry_mix_rate:
                needs_mixup = True
                # Sample mix_rate from Gaussian, resample if not in (0, 1)
                mix_rate = self._sample_mix_rate()

            # Check if augmentation is needed
            if np.random.random() < self.cry_aug_rate:
                effects = [
                    ('pitch', self.pitch_prob),
                    ('reverb', self.reverb_prob),
                    ('phaser', self.phaser_prob)
                ]
                waveform = self._apply_effects(waveform, effects)
        else:
            # Non-cry augmentation
            if np.random.random() < self.non_cry_aug_rate:
                effects = [
                    ('pitch', self.pitch_prob),
                    ('reverb', self.reverb_prob),
                    ('phaser', self.phaser_prob),
                    ('echo', self.echo_prob)
                ]
                waveform = self._apply_effects(waveform, effects)

        # Normalize to prevent clipping
        waveform = self._normalize(waveform)

        return waveform, needs_mixup, mix_rate

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
