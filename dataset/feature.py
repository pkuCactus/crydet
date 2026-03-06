"""
Feature Extraction for Baby Cry Detection
Supports MFCC and Filter Bank features with delta features
"""

import numpy as np
import librosa

from config import FeatureConfig


class FeatureExtractor:
    """
    Feature extractor for audio classification

    Supports:
    - MFCC (Mel-frequency cepstral coefficients)
    - Filter Bank (mel spectrogram)
    - Delta features (time derivative) - concatenated along feature dimension
    - Frequency delta features - concatenated along feature dimension
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize feature extractor

        Args:
            config: FeatureConfig instance with extraction parameters
        """
        self.config = config

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract features from audio waveform

        Args:
            y: Audio waveform (1D numpy array)
            sr: Sample rate

        Returns:
            Feature array with shape (feature_dim, time_frames)
            - feature_dim = base_dim + (delta_dim if use_delta) + (freq_delta_dim if use_freq_delta)
        """
        # Ensure mono
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # Extract base features
        if self.config.feature_type == 'mfcc':
            features = self._extract_mfcc(y, sr)
        else:
            features = self._extract_fbank(y, sr)

        # Collect features to concatenate along feature dimension
        feature_list = [features]

        if self.config.use_delta:
            delta = self._compute_delta(features)
            feature_list.append(delta)

        if self.config.use_freq_delta:
            freq_delta = self._compute_freq_delta(features)
            feature_list.append(freq_delta)

        # Concatenate along feature dimension (axis 0)
        features = np.concatenate(feature_list, axis=0)

        # Normalize if configured
        if self.config.normalize:
            features = self._normalize(features)

        return features.astype(np.float32)

    def _extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        return mfcc

    def _extract_fbank(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract filter bank (mel spectrogram) features"""
        fbank = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        # Convert to log scale
        fbank = librosa.power_to_db(fbank, ref=np.max)
        return fbank

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features per frequency band

        Args:
            features: Feature matrix (feature_dim, time_frames)

        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        std = np.maximum(std, 1e-8)  # Avoid division by zero
        return (features - mean) / std

    def _compute_delta(self, features: np.ndarray) -> np.ndarray:
        """
        Compute delta (time derivative) features

        Args:
            features: Base feature matrix (n_mels, time_frames)

        Returns:
            Delta features with same shape
        """
        delta = librosa.feature.delta(features, width=9)
        return delta

    def _compute_freq_delta(self, features: np.ndarray) -> np.ndarray:
        """
        Compute frequency delta features (derivative along frequency axis)

        Args:
            features: Base feature matrix (n_mels, time_frames)

        Returns:
            Frequency delta features
        """
        # Compute diff along frequency axis (axis 0)
        freq_delta = np.zeros_like(features)
        freq_delta[1:-1, :] = np.diff(features[1:, :] - features[:-1, :], axis=0)
        freq_delta[0, :] = features[1, :] - features[0, :]
        freq_delta[-1, :] = features[-1, :] - features[-2, :]
        return freq_delta

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Alias for extract method"""
        return self.extract(y, sr)

    @property
    def output_dim(self) -> int:
        """
        Return output feature dimension

        Returns:
            Total feature dimension (base + delta + freq_delta)
        """
        dim = self.config.feature_dim
        if self.config.use_delta:
            dim += self.config.feature_dim
        if self.config.use_freq_delta:
            dim += self.config.feature_dim
        return dim
