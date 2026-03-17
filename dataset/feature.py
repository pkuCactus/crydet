"""
Feature Extraction for Baby Cry Detection
Supports MFCC, Filter Bank, FFT and Energy features with normalization
Reference: docs/feature_extraction_flow.md
"""

import numpy as np
import librosa
from typing import Dict

from utils.config import FeatureConfig


class FeatureExtractor:
    """
    Feature extractor for audio classification

    Supports:
    - FFT (magnitude spectrum)
    - FBank (Mel filter bank / log Mel spectrum)
    - MFCC (Mel-frequency cepstral coefficients)
    - DB (Energy features: average and weighted)
    - Delta features (time and frequency derivatives)

    Processing pipeline:
    1. Preemphasis filtering
    2. Framing with Hanning window
    3. FFT computation
    4. Mel filter bank application
    5. Log transformation
    6. MFCC computation (optional)
    7. Energy feature computation
    8. Normalization

    Configuration options (via FeatureConfig):
    - feature_type: 'fbank', 'mfcc', or 'all' (fbank+mfcc)
    - use_delta: Add time derivative features (computed on combined base)
    - use_freq_delta: Add frequency derivative features (computed on combined base)
    - use_db_feature: Add energy (dB) as additional channel
    - use_db_norm: Normalize db features to [0, 1] range
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize feature extractor

        Args:
            config: FeatureConfig instance with extraction parameters
        """
        self.config = config

        # Pre-compute Hanning window
        self._window = np.hanning(config.n_fft)
        self._window_area = np.sum(self._window ** 2)

        # Mel filter bank matrix (computed lazily)
        self._mel_matrix = None
        self._sr = 16000  # Default sample rate

    def _get_mel_matrix(self, sr: int) -> np.ndarray:
        """Get or compute Mel filter bank matrix"""
        if self._mel_matrix is None or self._sr != sr:
            self._mel_matrix = librosa.filters.mel(
                sr=sr,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )
            self._sr = sr
        return self._mel_matrix

    def preemphasis(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply preemphasis filter to signal

        y[n] = x[n] - coeff * x[n-1]

        Args:
            signal: Input audio signal

        Returns:
            Preemphasized signal
        """
        coeff = self.config.preemphasis
        if coeff <= 0:
            return signal
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    def extract(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract all features from audio waveform

        Args:
            y: Audio waveform (1D numpy array)
            sr: Sample rate

        Returns:
            Dictionary containing:
            - 'fbank': Log Mel filter bank features (n_mels, frames)
            - 'mfcc': MFCC features (n_mfcc, frames)
            - 'db': Energy features (2, frames) [average, weighted]
              Values are normalized to [0, 1] if use_db_norm=True, else raw energy
        """
        # Ensure mono
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # Normalize amplitude
        y = librosa.util.normalize(y)

        # Apply preemphasis
        y_preemph = self.preemphasis(y)

        # Pad signal for framing
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        y_padded = np.pad(y_preemph, (n_fft - hop_length, 0), mode='constant')

        # Frame the signal
        frames = librosa.util.frame(y_padded, frame_length=n_fft, hop_length=hop_length)

        # Compute energy from original signal (before preemphasis)
        y_padded_orig = np.pad(y, (n_fft - hop_length, 0), mode='constant')
        y_squared = y_padded_orig ** 2
        energy_frames = librosa.util.frame(y_squared, frame_length=n_fft, hop_length=hop_length)

        # Apply Hanning window
        windowed_frames = frames * self._window[:, np.newaxis]

        # Compute FFT
        fft_result = np.fft.rfft(windowed_frames, axis=0)
        fft_magnitude = np.abs(fft_result)  # (n_fft//2+1, frames)

        # Compute Mel spectrum
        mel_matrix = self._get_mel_matrix(sr)
        mel_spectrum = np.dot(mel_matrix, fft_magnitude)  # (n_mels, frames)

        # Log transformation (avoid log(0))
        log_mel = np.log(np.maximum(mel_spectrum, 1e-8))

        # Compute MFCC from log Mel spectrum
        mfcc = librosa.feature.mfcc(
            S=log_mel,
            n_mfcc=self.config.n_mfcc,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )  # (n_mfcc, frames)

        # Compute energy features (in dB)
        # 1. Average energy per frame
        avg_energy = np.mean(energy_frames, axis=0)
        db_avg = energy_to_db(avg_energy)

        # 2. Hanning-windowed energy
        weighted_energy = np.sum(energy_frames * (self._window[:, np.newaxis] ** 2), axis=0) / self._window_area
        db_weighted = energy_to_db(weighted_energy)

        # Stack energy features
        db_features = np.stack([db_avg, db_weighted], axis=0)  # (2, frames)

        # Apply FBank normalization
        fbank = log_mel
        if self.config.use_fbank_norm:
            fbank = self._normalize_fbank(fbank)

        # Transpose to (feature_dim, frames) format
        features = {
            'fbank': fbank,
            'mfcc': mfcc,
            'db': db_features
        }

        return features

    def _normalize_fbank(self, fbank: np.ndarray) -> np.ndarray:
        """
        Normalize FBank features with exponential smoothing

        Args:
            fbank: Log Mel filter bank features (n_mels, frames)

        Returns:
            Normalized features in [0, 1] range
        """
        decay = self.config.fbank_decay

        # Compute global max/min across frequency axis
        fbank_max = np.max(fbank, axis=0, keepdims=True)  # (1, frames)
        fbank_min = np.min(fbank, axis=0, keepdims=True)  # (1, frames)

        # Exponential smoothing of max values along time axis (attack-decay envelope)
        if decay > 0 and fbank_max.shape[1] > 1:
            smoothed = np.zeros_like(fbank_max)
            smoothed[0, 0] = fbank_max[0, 0]
            for t in range(1, fbank_max.shape[1]):
                prev, curr = smoothed[0, t-1], fbank_max[0, t]
                smoothed[0, t] = decay * prev + (1 - decay) * curr if curr > prev else curr
            fbank_max = smoothed

        # Normalize to [0, 1]
        range_val = fbank_max - fbank_min
        range_val = np.where(range_val > 0, range_val, 1.0)

        fbank_norm = (fbank - fbank_min) / range_val
        fbank_norm = np.clip(fbank_norm, 0.0, 1.0)

        return fbank_norm

    def _compute_delta(self, features: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Compute delta (differential) features using central difference.

        Args:
            features: Input features array
            axis: Axis along which to compute delta (0=time, 1=frequency)

        Returns:
            Delta features with same shape as input
        """
        # Pad at boundaries
        pad_width = [(0, 0)] * features.ndim
        pad_width[axis] = (1, 1)
        padded = np.pad(features, pad_width, mode='edge')

        # Central difference: delta[t] = (feat[t+1] - feat[t-1]) / 2
        delta = (np.take(padded, range(2, padded.shape[axis]), axis=axis) -
                 np.take(padded, range(0, padded.shape[axis] - 2), axis=axis)) / 2.0

        return delta

    def _get_base_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get base features based on feature_type configuration.

        Returns base features with shape [T, F] where F depends on feature_type:
        - 'fbank': [T, n_mels], 'mfcc': [T, n_mfcc], 'all': [T, n_mels + n_mfcc]
        """
        feature_type = self.config.feature_type

        if feature_type == 'mfcc':
            return features['mfcc'].T

        if feature_type == 'all':
            return np.concatenate([features['fbank'].T, features['mfcc'].T], axis=1)

        # Default: 'fbank'
        return features['fbank'].T

    def extract_with_deltas(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract features with optional delta and energy (dB) features.

        Base features are selected by feature_type, then deltas are computed
        on the combined base features.

        Returns features in shape [T, F] format:
        - Base: [T, base_dim] where base_dim = n_mels or n_mfcc or n_mels+n_mfcc
        - +time delta: [T, base_dim * 2]
        - +freq delta: [T, base_dim * 3]
        - +db feature: adds 2 channel

        Args:
            y: Audio waveform
            sr: Sample rate

        Returns:
            Feature array of shape [time_frames, feature_dim]
        """
        # Extract all features
        features = self.extract(y, sr)

        # Get base features based on feature_type
        base = self._get_base_features(features)  # Shape: [frames, base_dim]

        # Build feature components: base + optional deltas + optional db
        deltas = [
            self._compute_delta(base, axis=0) if self.config.use_delta else None,
            self._compute_delta(base, axis=1) if self.config.use_freq_delta else None
        ]
        feature_components = [base] + [d for d in deltas if d is not None]

        # Add energy (dB) feature if enabled
        if self.config.use_db_feature:
            feature_components.append(features['db'].T)

        # Concatenate all features along feature dimension
        return np.concatenate(feature_components, axis=1) if len(feature_components) > 1 else base

    def extract_single(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract single feature type with optional deltas and db.

        This is an alias for extract_with_deltas() for API consistency.
        The feature_type determines the base features (fbank/mfcc/all).

        Args:
            y: Audio waveform
            sr: Sample rate

        Returns:
            Feature array of shape [time_frames, feature_dim]
        """
        return self.extract_with_deltas(y, sr)

    def __call__(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Alias for extract method"""
        return self.extract(y, sr)


def energy_to_db(energy: np.ndarray, ref: float = 1.0, amin: float = 1e-8) -> np.ndarray:
    """
    Convert energy to decibels

    Args:
        energy: Energy values
        ref: Reference value
        amin: Minimum value to avoid log(0)

    Returns:
        Energy in decibels, normalized to [0, 1] range
    """
    db = 10 * np.log10(np.maximum(energy, amin) / ref)
    # Normalize to [0, 1] range (clip at -8 dB)
    db_normalized = (np.maximum(db, -8) + 8) / 8
    return db_normalized
