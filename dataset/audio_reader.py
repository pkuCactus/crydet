"""
Audio Reader for Cry Detection Dataset
Supports partial reading and caching of resampled audio as WAV files
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

    Caches resampled audio as WAV files to support partial reading.
    Files with matching sample rate are read directly without caching.

    Example:
        reader = AudioReader(target_sr=16000, cache_dir='.cache/audio')
        # Load entire audio
        waveform, sr = reader.load('audio.wav')
        # Load partial audio (from sample 1000 to 5000)
        waveform, sr = reader.load('audio.wav', start=1000, stop=5000)
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
            cache_dir: Directory to cache resampled audio as WAV (default: None)
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

    def load(
        self,
        file_path: Union[str, Path],
        start: int = 0,
        stop: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with resampling and caching

        - Files with matching sample rate: read directly without caching
        - Files needing resampling: resample entire audio, cache as WAV,
          then support partial reading from cache

        Args:
            file_path: Path to audio file
            start: Start sample index (default: 0)
            stop: End sample index, exclusive (default: None = end of file)

        Returns:
            Tuple of (waveform, sample_rate)
        """
        file_path = Path(file_path)

        # Get audio info to check sample rate
        info = sf.info(str(file_path))
        orig_sr = info.samplerate

        # Fast path: sample rate matches, read directly without caching
        if orig_sr == self.target_sr:
            waveform = sf.read(file_path, dtype='float32', start=start, stop=stop)[0]
            if self.force_mono and waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            return waveform, self.target_sr

        # Slow path: need resampling
        cache_path = self._get_cache_path(file_path, orig_sr)

        # Check if cached WAV exists
        if self.use_cache and self.cache_dir and cache_path.exists():
            # Read from cached WAV with partial support
            waveform = sf.read(cache_path, dtype='float32', start=start, stop=stop)[0]
            if self.force_mono and waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            return waveform, self.target_sr

        # Load entire audio, resample, and cache
        waveform, _ = sf.read(file_path, dtype='float32')

        if self.force_mono and waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample entire audio
        waveform = self._resample(waveform, orig_sr, self.target_sr)

        # Save as WAV for partial reading support
        if self.use_cache and self.cache_dir:
            self._save_to_cache(cache_path, waveform, file_path, orig_sr)
            # Now read partial from cache
            if start != 0 or stop is not None:
                waveform = sf.read(cache_path, dtype='float32', start=start, stop=stop)[0]
                if self.force_mono and waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)

        return waveform, self.target_sr

    def load_by_time(
        self,
        file_path: Union[str, Path],
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio segment by time (in seconds)

        Args:
            file_path: Path to audio file
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (default: None = end of file)

        Returns:
            Tuple of (waveform, sample_rate)
        """
        start_sample = int(start_time * self.target_sr)
        stop_sample = int(end_time * self.target_sr) if end_time is not None else None
        return self.load(file_path, start=start_sample, stop=stop_sample)

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
        """Get cache file path for resampled audio (WAV format)"""
        audio_hash = self._compute_hash(file_path)
        return self.cache_dir / f"{audio_hash}_{orig_sr}to{self.target_sr}hz.wav"

    def _save_to_cache(
        self,
        cache_path: Path,
        waveform: np.ndarray,
        source_path: Path,
        orig_sr: int
    ):
        """Save resampled audio as WAV for partial reading support"""
        # Save as WAV
        sf.write(cache_path, waveform, self.target_sr)

        # Save metadata
        meta_path = cache_path.with_suffix('.json')
        meta = {
            'source_path': str(source_path),
            'orig_sr': orig_sr,
            'target_sr': self.target_sr,
            'shape': list(waveform.shape),
            'dtype': str(waveform.dtype),
            'cache_format': 'wav'
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

    def clear_cache(self) -> int:
        """Clear all cached audio files"""
        if not self.cache_dir or not self.cache_dir.exists():
            return 0

        count = 0
        for ext in ['*.wav', '*.npy']:
            for file in self.cache_dir.glob(ext):
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

        # Count both wav and npy files
        cache_files = list(self.cache_dir.glob('*.wav')) + list(self.cache_dir.glob('*.npy'))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'file_count': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

    def __repr__(self) -> str:
        return (
            f"AudioReader(target_sr={self.target_sr}, "
            f"cache_dir={self.cache_dir}, "
            f"use_cache={self.use_cache}, "
            f"force_mono={self.force_mono})"
        )
