import random

import numpy as np


def get_db(y: np.ndarray, eps: float = 1e-8) -> np.float64:
    """
    Convert amplitude to decibels

    Args:
        y: Input audio signal
        eps: Small value to avoid log of zero (default: 1e-8)

    Returns:
        Decibel values
    """
    return 10 * np.log10(np.maximum(np.mean(y ** 2), eps))


def get_p(y: np.ndarray, y_ref: np.ndarray, ratio: float = 0.5, eps: float = 1e-8) -> float:
    """
    Compute power ratio between two signals

    Args:
        y: Input audio signal
        y_ref: Reference audio signal
        eps: Small value to avoid division by zero (default: 1e-8)

    Returns:
        Power ratio in decibels
    """
    y_db = get_db(y, eps)
    y_ref_db = get_db(y_ref, eps)
    scale = 10 ** ((y_db - y_ref_db) / 20)
    scale_r = (1 - ratio) / ratio
    return 1 / (1 + scale * scale_r)


def pad_pcm(
    y: np.ndarray,
    target_length: int,
    pad_silence_prob: float = 0.5,
    pad_front_prob: float = 0.5,
    truncate: bool = True
) -> np.ndarray:
    """
    Pad or truncate PCM audio to target length

    Args:
        y: Input audio signal
        target_length: Desired length in samples
        pad_silence_prob: Probability of padding with silence (default: 0.5)
        pad_front_prob: Probability of padding at the front (default: 0.5)
        truncate: If True, truncate when y is longer than target_length.
                 If False, return y as-is when pad_length <= 0.

    Returns:
        Padded or truncated audio signal
    """
    pad_length = target_length - len(y)
    if pad_length <= 0:
        return y[:target_length] if truncate else y
    if random.random() < pad_silence_prob:
        padding = np.zeros(pad_length, dtype=y.dtype)
    else:
        noise_level = 0.01 * np.std(y) if np.std(y) > 0 else 0.001
        padding = np.random.randn(pad_length).astype(y.dtype) * noise_level

    if random.random() < pad_front_prob:
        return np.concatenate([padding, y])
    return np.concatenate([y, padding])


def add_noise(y: np.ndarray, snr: int = 5, return_noise: bool = False, silent_rate: float = 0.5, abs: bool = False) -> np.ndarray:
    if y.ndim == 2 and (not y.shape[0] or not y.shape[1]):
        return y
    if return_noise and random.random() < silent_rate:
        return np.zeros_like(y)
    noise = np.random.randn(*y.shape)
    noise = np.nan_to_num(noise)
    noise = noise - np.mean(noise)
    y_power = np.linalg.norm(y) ** 2 / y.size
    noise_var = y_power / np.power(10, (snr / 10))
    noise = noise * (np.sqrt(noise_var) / np.std(noise))
    result = noise if return_noise else y + noise
    if abs:
        result = np.abs(result)
    result = np.nan_to_num(result)
    return result


def gain(y: np.ndarray, ref_db: float, abs: bool = False) -> np.ndarray:
    gain_energy = 10 ** (ref_db / 20)
    if abs:
        ori_energy = max(np.mean(y ** 2), 1e-8) ** 0.5
        y_gain = y / ori_energy * gain_energy
    else:
        y_gain = y * gain_energy
    y_gain = np.clip(y_gain, -1, 1)
    return y_gain


def generate_pink_noise(length: int, alpha: float = 1.0) -> np.ndarray:
    """
    Generate pink noise (1/f^alpha noise).

    Pink noise has equal energy per octave, making it sound more natural
    than white noise for audio augmentation.

    Args:
        length: Length of the noise signal in samples
        alpha: Spectral decay exponent (default: 1.0 for classic pink noise)
               alpha=0 produces white noise, alpha=2 produces brown noise

    Returns:
        Pink noise signal normalized to [-1, 1]
    """
    # Generate white noise in frequency domain
    white = np.random.randn(length)

    # Compute FFT
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)

    # Avoid division by zero at DC
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0

    # Apply 1/f^alpha filter
    pink_filter = 1.0 / (freqs ** (alpha / 2.0))
    pink_filter[0] = pink_filter[1]  # Match DC to lowest frequency

    # Apply filter and inverse FFT
    pink_fft = fft * pink_filter
    pink = np.fft.irfft(pink_fft, n=length)

    # Normalize
    pink = pink / np.max(np.abs(pink) + 1e-8)

    return pink.astype(np.float32)


def generate_brown_noise(length: int) -> np.ndarray:
    """
    Generate brown noise (1/f^2 noise).

    Brown noise has more low-frequency content than pink noise,
    useful for simulating low-frequency ambient sounds.

    Args:
        length: Length of the noise signal in samples

    Returns:
        Brown noise signal normalized to [-1, 1]
    """
    return generate_pink_noise(length, alpha=2.0)


def add_pink_noise(y: np.ndarray, snr: float = 15.0, alpha: float = 1.0) -> np.ndarray:
    """
    Add pink noise to audio signal at specified SNR level.

    Args:
        y: Input audio signal
        snr: Signal-to-noise ratio in dB (default: 15)
        alpha: Pink noise spectral decay (default: 1.0)

    Returns:
        Noisy audio signal
    """
    pink = generate_pink_noise(len(y), alpha=alpha)

    # Calculate signal and noise power
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(pink ** 2)

    # Calculate required noise scaling for target SNR
    # SNR = 10 * log10(signal_power / noise_power)
    target_noise_power = signal_power / (10 ** (snr / 10))
    noise_scale = np.sqrt(target_noise_power / (noise_power + 1e-8))

    # Mix signal and scaled noise
    noisy = y + pink * noise_scale

    return np.clip(noisy, -1, 1)


def load_ambient_noise(
    noise_files: list,
    target_length: int,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Load and prepare ambient noise from file(s).

    Args:
        noise_files: List of paths to ambient noise audio files
        target_length: Target length in samples
        sample_rate: Expected sample rate

    Returns:
        Ambient noise segment of target_length
    """
    import librosa

    if not noise_files:
        # Fallback to pink noise if no files provided
        return generate_pink_noise(target_length)

    # Randomly select a noise file
    noise_file = random.choice(noise_files)

    try:
        # Load noise file
        noise, _ = librosa.load(noise_file, sr=sample_rate, mono=True)

        # Extend or truncate to target length
        if len(noise) < target_length:
            # Loop the noise if too short
            repeats = (target_length // len(noise)) + 1
            noise = np.tile(noise, repeats)

        # Random offset for variety
        max_offset = max(0, len(noise) - target_length)
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
        noise = noise[offset:offset + target_length]

        # Normalize
        noise = noise / (np.max(np.abs(noise)) + 1e-8)

        return noise.astype(np.float32)

    except Exception:
        # Fallback to pink noise on error
        return generate_pink_noise(target_length)


def add_ambient_noise(
    y: np.ndarray,
    noise_files: list,
    snr: float = 15.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Add ambient/environmental noise to audio signal.

    Args:
        y: Input audio signal
        noise_files: List of paths to ambient noise files
        snr: Signal-to-noise ratio in dB (default: 15)
        sample_rate: Sample rate for loading noise files

    Returns:
        Noisy audio signal
    """
    noise = load_ambient_noise(noise_files, len(y), sample_rate)

    # Calculate signal and noise power
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2)

    # Calculate required noise scaling for target SNR
    target_noise_power = signal_power / (10 ** (snr / 10))
    noise_scale = np.sqrt(target_noise_power / (noise_power + 1e-8))

    # Mix signal and scaled noise
    noisy = y + noise * noise_scale

    return np.clip(noisy, -1, 1)


def mix_with_noise(
    y: np.ndarray,
    noise_type: str = 'white',
    snr: float = 15.0,
    noise_files: list = None,
    sample_rate: int = 16000,
    pink_alpha: float = 1.0
) -> np.ndarray:
    """
    Mix audio with various types of noise.

    Args:
        y: Input audio signal
        noise_type: Type of noise ('white', 'pink', 'brown', 'ambient')
        snr: Signal-to-noise ratio in dB
        noise_files: List of ambient noise files (for noise_type='ambient')
        sample_rate: Sample rate for loading noise files
        pink_alpha: Spectral decay for pink noise (1.0 = classic pink)

    Returns:
        Noisy audio signal
    """
    if noise_type == 'pink':
        return add_pink_noise(y, snr=snr, alpha=pink_alpha)
    elif noise_type == 'brown':
        return add_pink_noise(y, snr=snr, alpha=2.0)
    elif noise_type == 'ambient' and noise_files:
        return add_ambient_noise(y, noise_files, snr=snr, sample_rate=sample_rate)
    else:
        # Default to white noise
        return add_noise(y, snr=int(snr))
