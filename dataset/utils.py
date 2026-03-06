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


def pad_pcm(y: np.ndarray, target_length: int, pad_silence_prob: float = 0.5, pad_front_prob: float = 0.5) -> np.ndarray:
    """
    Pad or truncate PCM audio to target length

    Args:
        y: Input audio signal
        target_length: Desired length in samples
        pad_silence_prob: Probability of padding with silence (default: 0.5)
        pad_front_prob: Probability of padding at the front (default: 0.5)

    Returns:
        Padded or truncated audio signal
    """
    pad_length = target_length - len(y)
    if pad_length <= 0:
        return y[:target_length]
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
