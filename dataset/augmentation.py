"""
Audio Augmentation Utilities
Supports various augmentation techniques for audio classification
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Callable
from scipy import signal

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioAugmenter:
    """
    Audio augmentation with configurable parameters

    Supports:
    - Additive noise
    - Time masking
    - Pitch shift
    - Reverb
    - Random gain
    - Mixup (applied at batch level)
    """

    def __init__(
        self,
        noise_rate: Optional[float] = None,
        mask_rate: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        reverb_rate: Optional[float] = None,
        gain_db: Optional[float] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize augmenter with parameters

        Args:
            noise_rate: Noise addition probability (0-1)
            mask_rate: Time masking probability (0-1)
            pitch_shift: Pitch shift range in semitones (e.g., 2.0 means -2 to +2)
            reverb_rate: Reverb probability (0-1)
            gain_db: Random gain range in dB (e.g., 6.0 means -6 to +6 dB)
            sample_rate: Audio sample rate
        """
        self.noise_rate = noise_rate
        self.mask_rate = mask_rate
        self.pitch_shift = pitch_shift
        self.reverb_rate = reverb_rate
        self.gain_db = gain_db
        self.sample_rate = sample_rate

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply all enabled augmentations to waveform"""
        augmented = waveform.copy()

        if self.noise_rate is not None and np.random.random() < self.noise_rate:
            augmented = self.add_noise(augmented)

        if self.mask_rate is not None and np.random.random() < self.mask_rate:
            augmented = self.time_mask(augmented)

        if self.pitch_shift is not None and np.random.random() < 0.5:
            augmented = self.pitch_shift_audio(augmented)

        if self.reverb_rate is not None and np.random.random() < self.reverb_rate:
            augmented = self.add_reverb(augmented)

        if self.gain_db is not None:
            augmented = self.random_gain(augmented)

        return augmented

    def add_noise(
        self,
        waveform: np.ndarray,
        noise_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Add random Gaussian noise to waveform

        Args:
            waveform: Input audio waveform
            noise_factor: Noise scaling factor (default: random 0.001-0.01)

        Returns:
            Noisy waveform
        """
        if noise_factor is None:
            noise_factor = np.random.uniform(0.001, 0.01)

        # Scale noise relative to signal
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms > 0:
            noise = np.random.randn(len(waveform)).astype(waveform.dtype)
            noise = noise * rms * noise_factor
            augmented = waveform + noise
        else:
            augmented = waveform

        return augmented.astype(waveform.dtype)

    def time_mask(
        self,
        waveform: np.ndarray,
        max_mask_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Apply time masking (randomly zero out segments)

        Args:
            waveform: Input audio waveform
            max_mask_ratio: Maximum ratio of audio to mask (0-1)

        Returns:
            Masked waveform
        """
        augmented = waveform.copy()
        length = len(waveform)

        # Random number of masks (1-3)
        num_masks = np.random.randint(1, 4)

        for _ in range(num_masks):
            # Random mask length
            mask_len = int(length * np.random.uniform(0.01, max_mask_ratio))
            mask_start = np.random.randint(0, length - mask_len)
            augmented[mask_start:mask_start + mask_len] = 0

        return augmented

    def pitch_shift_audio(
        self,
        waveform: np.ndarray,
        n_steps: Optional[float] = None
    ) -> np.ndarray:
        """
        Shift pitch of audio

        Args:
            waveform: Input audio waveform
            n_steps: Number of semitones to shift (default: random within config range)

        Returns:
            Pitch-shifted waveform
        """
        if n_steps is None:
            if self.pitch_shift is None:
                return waveform
            n_steps = np.random.uniform(-self.pitch_shift, self.pitch_shift)

        if HAS_LIBROSA:
            # Use librosa for high-quality pitch shifting
            try:
                shifted = librosa.effects.pitch_shift(
                    waveform,
                    sr=self.sample_rate,
                    n_steps=n_steps
                )
                return shifted.astype(waveform.dtype)
            except Exception:
                pass

        # Fallback: simple resampling-based pitch shift
        return self._pitch_shift_resample(waveform, n_steps)

    def _pitch_shift_resample(
        self,
        waveform: np.ndarray,
        n_steps: float
    ) -> np.ndarray:
        """
        Pitch shift via resampling (lower quality but no librosa dependency)

        Args:
            waveform: Input audio waveform
            n_steps: Number of semitones to shift

        Returns:
            Pitch-shifted waveform
        """
        # Calculate speed change factor
        factor = 2 ** (n_steps / 12.0)

        # Resample to change pitch
        new_length = int(len(waveform) / factor)
        if new_length < 100:  # Safety check
            return waveform

        resampled = signal.resample(waveform, new_length)

        # Pad or truncate to original length
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
        Add simple reverb effect using delayed echoes

        Args:
            waveform: Input audio waveform
            room_scale: Room size (affects delay time, 0-1)
            decay: Decay factor for echoes (0-1)

        Returns:
            Waveform with reverb
        """
        if room_scale is None:
            room_scale = np.random.uniform(0.1, 0.5)
        if decay is None:
            decay = np.random.uniform(0.2, 0.5)

        augmented = waveform.copy()

        # Create multiple delayed echoes
        delay_times = [0.03, 0.05, 0.08, 0.12]  # seconds
        decay_factors = [decay, decay * 0.7, decay * 0.5, decay * 0.3]

        for delay_time, decay_factor in zip(delay_times, decay_factors):
            delay_samples = int(delay_time * self.sample_rate)
            if delay_samples < len(waveform):
                delayed = np.zeros_like(waveform)
                delayed[delay_samples:] = waveform[:-delay_samples] * decay_factor
                augmented = augmented + delayed

        # Normalize to prevent clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val

        return augmented.astype(waveform.dtype)

    def random_gain(
        self,
        waveform: np.ndarray,
        gain_range_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply random gain to waveform

        Args:
            waveform: Input audio waveform
            gain_range_db: Gain range in dB (e.g., 6.0 means -6 to +6 dB)

        Returns:
            Gain-adjusted waveform
        """
        if gain_range_db is None:
            gain_range_db = self.gain_db if self.gain_db is not None else 6.0

        # Random gain in dB
        gain_db = np.random.uniform(-gain_range_db, gain_range_db)

        # Convert to linear scale
        gain_linear = 10 ** (gain_db / 20.0)

        augmented = waveform * gain_linear

        # Soft clip to prevent harsh clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = np.tanh(augmented)

        return augmented.astype(waveform.dtype)

    def time_stretch(
        self,
        waveform: np.ndarray,
        rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Time stretch audio without changing pitch

        Args:
            waveform: Input audio waveform
            rate: Stretch rate (e.g., 1.2 means 20% faster)

        Returns:
            Time-stretched waveform
        """
        if rate is None:
            rate = np.random.uniform(0.9, 1.1)

        if HAS_LIBROSA:
            try:
                stretched = librosa.effects.time_stretch(waveform, rate=rate)
                # Pad or truncate to original length
                if len(stretched) < len(waveform):
                    pad_len = len(waveform) - len(stretched)
                    stretched = np.pad(stretched, (0, pad_len), mode='constant')
                else:
                    stretched = stretched[:len(waveform)]
                return stretched.astype(waveform.dtype)
            except Exception:
                pass

        # Fallback: simple resampling (changes pitch too)
        new_length = int(len(waveform) * rate)
        if new_length < 100:
            return waveform

        resampled = signal.resample(waveform, new_length)

        if len(resampled) < len(waveform):
            pad_len = len(waveform) - len(resampled)
            resampled = np.pad(resampled, (0, pad_len), mode='constant')
        else:
            resampled = resampled[:len(waveform)]

        return resampled.astype(waveform.dtype)


def mixup(
    waveform1: np.ndarray,
    waveform2: np.ndarray,
    alpha: float = 0.4
) -> Tuple[np.ndarray, float]:
    """
    Mixup augmentation for two waveforms

    Args:
        waveform1: First waveform
        waveform2: Second waveform
        alpha: Beta distribution parameter for mixup

    Returns:
        Tuple of (mixed waveform, mixup coefficient)
    """
    # Sample mixup coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Ensure same length
    min_len = min(len(waveform1), len(waveform2))

    mixed = lam * waveform1[:min_len] + (1 - lam) * waveform2[:min_len]

    return mixed.astype(waveform1.dtype), lam


def mixup_batch(
    waveforms: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.4,
    num_classes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch-level mixup augmentation

    Randomly shuffles batch and mixes with original, returning mixed samples
    and soft labels (for use with cross-entropy loss).

    Args:
        waveforms: Batch of waveforms, shape (batch_size, seq_len)
        labels: Batch of labels, shape (batch_size,) - integer class indices
        alpha: Beta distribution parameter for mixup
        num_classes: Number of classes (inferred from labels if None)

    Returns:
        Tuple of (mixed waveforms, soft labels)
        - mixed waveforms: shape (batch_size, seq_len)
        - soft labels: shape (batch_size, num_classes) - one-hot mixed labels
    """
    batch_size = len(waveforms)

    if num_classes is None:
        num_classes = int(np.max(labels)) + 1

    # Sample mixup coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Shuffle indices
    shuffle_idx = np.random.permutation(batch_size)

    # Mix waveforms
    mixed_waveforms = lam * waveforms + (1 - lam) * waveforms[shuffle_idx]

    # Create soft labels (one-hot then mix)
    # For efficiency, we return (labels, shuffled_labels, lam) tuple
    # The loss function should compute: lam * CE(pred, labels) + (1-lam) * CE(pred, shuffled_labels)
    # This avoids creating large one-hot matrices
    shuffled_labels = labels[shuffle_idx]

    return mixed_waveforms.astype(waveforms.dtype), labels, shuffled_labels, lam


def mixup_batch_soft(
    waveforms: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.4,
    num_classes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch-level mixup with explicit soft labels

    Args:
        waveforms: Batch of waveforms, shape (batch_size, seq_len)
        labels: Batch of labels, shape (batch_size,) - integer class indices
        alpha: Beta distribution parameter for mixup
        num_classes: Number of classes (inferred from labels if None)

    Returns:
        Tuple of (mixed waveforms, soft labels)
        - mixed waveforms: shape (batch_size, seq_len)
        - soft labels: shape (batch_size, num_classes)
    """
    batch_size = len(waveforms)

    if num_classes is None:
        num_classes = int(np.max(labels)) + 1

    # Sample mixup coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Shuffle indices
    shuffle_idx = np.random.permutation(batch_size)

    # Mix waveforms
    mixed_waveforms = lam * waveforms + (1 - lam) * waveforms[shuffle_idx]

    # Create one-hot labels
    one_hot_labels = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot_labels[np.arange(batch_size), labels] = 1.0

    # Create shuffled one-hot labels
    one_hot_shuffled = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot_shuffled[np.arange(batch_size), labels[shuffle_idx]] = 1.0

    # Mix labels
    soft_labels = lam * one_hot_labels + (1 - lam) * one_hot_shuffled

    return mixed_waveforms.astype(waveforms.dtype), soft_labels


class MixupCollateFn:
    """
    Collate function with mixup augmentation for DataLoader

    Binary classification: cry (1) vs non-cry (0)
    Any label containing 'cry' is treated as positive class.

    Example:
        >>> collate_fn = MixupCollateFn(alpha=0.4, mixup_prob=0.5)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> for waveforms, labels_a, labels_b, lam in dataloader:
        ...     loss = lam * criterion(model(waveforms), labels_a) + \
        ...            (1 - lam) * criterion(model(waveforms), labels_b)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Collate a batch of samples with optional mixup

        Args:
            batch: List of (waveform, label) tuples

        Returns:
            (waveforms, labels_a, labels_b, lam)
            - waveforms: torch.Tensor, shape (batch_size, seq_len)
            - labels_a: torch.Tensor, shape (batch_size,) - original labels
            - labels_b: torch.Tensor, shape (batch_size,) - shuffled labels
            - lam: float - mixup coefficient (1.0 if no mixup)
        """
        # Separate waveforms and labels
        waveforms = [item[0] for item in batch]
        labels_str = [item[1] for item in batch]

        # Stack waveforms into array (pad to max length)
        max_len = max(len(w) for w in waveforms)
        waveforms_padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for i, w in enumerate(waveforms):
            waveforms_padded[i, :len(w)] = w

        # Convert string labels to binary indices (cry=1, non-cry=0)
        labels = np.array([self._label_to_idx(label) for label in labels_str], dtype=np.int64)

        # Decide whether to apply mixup
        if np.random.random() < self.mixup_prob and self.alpha > 0:
            mixed_waveforms, labels_a, labels_b, lam = mixup_batch(
                waveforms_padded, labels, self.alpha, num_classes=2
            )
            return (
                torch.from_numpy(mixed_waveforms),
                torch.from_numpy(labels_a),
                torch.from_numpy(labels_b),
                lam
            )
        else:
            # No mixup - return original batch with lam=1.0
            return (
                torch.from_numpy(waveforms_padded),
                torch.from_numpy(labels),
                torch.from_numpy(labels),
                1.0
            )


def mixup_criterion(
    criterion: Callable,
    predictions: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute mixup loss from standard criterion

    Args:
        criterion: Loss function that takes (predictions, labels)
        predictions: Model outputs, shape (batch_size, num_classes)
        labels_a: First set of labels, shape (batch_size,)
        labels_b: Second set of labels (shuffled), shape (batch_size,)
        lam: Mixup coefficient

    Returns:
        Mixed loss value

    Example:
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    """
    return lam * criterion(predictions, labels_a) + (1 - lam) * criterion(predictions, labels_b)


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
