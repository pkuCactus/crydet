"""
Audio Reader for Cry Detection Dataset
"""

import json
import hashlib
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, Optional, Tuple, List
from scipy import signal


class AudioReader:
    """
    Audio file reader with resampling and caching support

    Example:
        reader = AudioReader(target_sr=16000, cache_dir='.cache/audio')
        waveform, sr = reader.load('audio.wav')
        batch, sr = reader.load_batch(['audio1.wav', 'audio2.wav'])
    """

    def __init__(
        self,
        target_sr: int = 16000,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        force_mono: bool = True
    ):
        """
        Initialize AudioReader

        Args:
            target_sr: Target sample rate (default: 16000)
            cache_dir: Directory to cache resampled audio (default: None)
            use_cache: Whether to use cache (default: True)
            force_mono: Convert stereo to mono (default: True)
        """
        self.target_sr = target_sr
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.force_mono = force_mono

        # Create cache directory if needed
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file with resampling and caching

        Only caches files that need resampling.
        Files with matching sample rate are read directly for best performance.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        file_path = Path(file_path)

        # Get audio info first to check sample rate
        info = sf.info(str(file_path))
        orig_sr = info.samplerate

        # Fast path: sample rate matches, read directly without caching
        if orig_sr == self.target_sr:
            waveform = sf.read(file_path, dtype='float32')[0]
            if self.force_mono and waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            return waveform, self.target_sr

        # Slow path: need resampling, use cache
        if self.use_cache and self.cache_dir:
            cached = self._load_from_cache(file_path)
            if cached is not None:
                return cached, self.target_sr

        # Load and resample
        waveform, _ = sf.read(file_path, dtype='float32')

        if self.force_mono and waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        waveform = self._resample(waveform, orig_sr, self.target_sr)

        # Cache only resampled files
        if self.use_cache and self.cache_dir:
            self._save_to_cache(file_path, waveform, orig_sr)

        return waveform, self.target_sr

    def load_batch(
        self,
        file_paths: List[Union[str, Path]],
        max_length: Optional[int] = None,
        pad_value: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load multiple audio files into a batch

        Args:
            file_paths: List of audio file paths
            max_length: Maximum length in samples (pad/truncate if specified)
            pad_value: Value to use for padding

        Returns:
            Tuple of (batch_waveform, sample_rate)
        """
        waveforms = [self.load(fp)[0] for fp in file_paths]

        if max_length is not None:
            batch = np.full((len(waveforms), max_length), pad_value, dtype=np.float32)
            for i, waveform in enumerate(waveforms):
                length = min(len(waveform), max_length)
                batch[i, :length] = waveform[:length]
        else:
            batch = np.array(waveforms, dtype=object)

        return batch, self.target_sr

    def _resample(
        self,
        waveform: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio using scipy.signal.resample"""
        if orig_sr == target_sr:
            return waveform

        duration = len(waveform) / orig_sr
        new_length = int(duration * target_sr)

        if waveform.ndim > 1:
            # Multi-channel
            resampled = np.stack([
                signal.resample(waveform[ch], new_length)
                for ch in range(waveform.shape[0])
            ], axis=0)
        else:
            # Single channel
            resampled = signal.resample(waveform, new_length)

        return resampled.astype(np.float32)

    def _compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of audio file"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_path(self, file_path: Path, orig_sr: int) -> Path:
        """Get cache file path for resampled audio"""
        audio_hash = self._compute_hash(file_path)
        return self.cache_dir / f"{audio_hash}_{orig_sr}to{self.target_sr}hz.npy"

    def _load_from_cache(self, file_path: Path) -> Optional[np.ndarray]:
        """Load audio from cache if exists and valid"""
        # Need to find cache file by scanning (since we don't know orig_sr)
        if not self.cache_dir.exists():
            return None

        audio_hash = self._compute_hash(file_path)

        # Find matching cache file
        for cache_file in self.cache_dir.glob(f"{audio_hash}_*to{self.target_sr}hz.npy"):
            meta_path = cache_file.with_suffix('.json')
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)

                    if meta.get('source_path') == str(file_path):
                        return np.load(cache_file)
                except (json.JSONDecodeError, KeyError):
                    pass

        return None

    def _save_to_cache(self, file_path: Path, waveform: np.ndarray, orig_sr: int):
        """Save resampled audio to cache"""
        cache_path = self._get_cache_path(file_path, orig_sr)
        meta_path = cache_path.with_suffix('.json')

        np.save(cache_path, waveform)

        meta = {
            'source_path': str(file_path),
            'orig_sr': orig_sr,
            'target_sr': self.target_sr,
            'shape': list(waveform.shape),
            'dtype': str(waveform.dtype)
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

    def clear_cache(self) -> int:
        """Clear all cached audio files"""
        if not self.cache_dir or not self.cache_dir.exists():
            return 0

        count = 0
        for file in self.cache_dir.glob('*.npy'):
            file.unlink()
            count += 1

        for file in self.cache_dir.glob('*.json'):
            file.unlink()

        return count

    def get_cache_info(self) -> dict:
        """Get cache statistics"""
        if not self.cache_dir:
            return {
                'enabled': False,
                'cache_dir': None,
                'file_count': 0,
                'total_size_mb': 0
            }

        if not self.cache_dir.exists():
            return {
                'enabled': True,
                'cache_dir': str(self.cache_dir),
                'file_count': 0,
                'total_size_mb': 0
            }

        npy_files = list(self.cache_dir.glob('*.npy'))
        total_size = sum(f.stat().st_size for f in npy_files)

        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'file_count': len(npy_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

    def __repr__(self) -> str:
        return (
            f"AudioReader(target_sr={self.target_sr}, "
            f"cache_dir={self.cache_dir}, "
            f"use_cache={self.use_cache}, "
            f"force_mono={self.force_mono})"
        )
