"""
Audio Reader for Cry Detection Dataset
Supports partial reading and caching of resampled audio as WAV files
"""

import json
import mmap
import os
import struct
import threading
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Thread-local storage for librosa to avoid import overhead per thread
_thread_local = threading.local()


def _get_librosa():
    """Get thread-local librosa instance"""
    if not hasattr(_thread_local, 'librosa'):
        import librosa
        _thread_local.librosa = librosa
    return _thread_local.librosa


class AudioCache:
    """
    LRU cache for frequently accessed audio files.
    Thread-safe implementation for DataLoader multi-processing.
    """
    def __init__(self, max_size: int = 50, max_memory_mb: float = 500):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
        self._memory_usage = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get item from cache, update access order"""
        with self._lock:
            if key in self._cache:
                # Move to front (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: np.ndarray) -> bool:
        """Add item to cache, evict if necessary"""
        with self._lock:
            value_size = value.nbytes

            # Don't cache if single item exceeds max memory
            if value_size > self.max_memory_bytes * 0.5:
                return False

            # Evict items if necessary
            while (len(self._cache) >= self.max_size or
                   self._memory_usage + value_size > self.max_memory_bytes):
                if not self._evict_lru():
                    break

            self._cache[key] = value
            self._access_order.append(key)
            self._memory_usage += value_size
            return True

    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self._access_order:
            return False
        lru_key = self._access_order.pop(0)
        if lru_key in self._cache:
            self._memory_usage -= self._cache[lru_key].nbytes
            del self._cache[lru_key]
        return True

    def clear(self):
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._memory_usage = 0

    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_mb': self._memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self._hits,
                'misses': self._misses
            }


class AudioReader:
    """
    Audio file reader with resampling, file caching and memory caching support

    Features:
    - Memory LRU cache for frequently accessed files
    - File caching for resampled audio (WAV format)
    - Parallel batch loading
    - Files with matching sample rate are read directly without caching

    Example:
        reader = AudioReader(target_sr=16000, cache_dir='.cache/audio', memory_cache_mb=500)
        # Load entire audio
        waveform, sr = reader.load('audio.wav')
        # Load partial audio (from sample 1000 to 5000)
        waveform, sr = reader.load('audio.wav', start=1000, stop=5000)
    """

    def __init__(
        self,
        target_sr: int = 16000,
        cache_dir: Optional[str] = None,
        force_mono: bool = True,
        memory_cache_mb: float = 500,
        memory_cache_size: int = 100
    ):
        """
        Initialize AudioReader

        Args:
            target_sr: Target sample rate (default: 16000)
            cache_dir: Directory to cache resampled audio as WAV (default: None)
            force_mono: Convert stereo to mono (default: True)
            memory_cache_mb: Maximum memory for in-memory cache in MB (default: 500)
            memory_cache_size: Maximum number of files in memory cache (default: 100)
        """
        self.target_sr = target_sr
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.force_mono = force_mono

        # Create cache directory if needed
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory cache
        self._memory_cache = AudioCache(
            max_size=memory_cache_size,
            max_memory_mb=memory_cache_mb
        )

        # File info cache to avoid repeated sf.info calls
        self._file_info_cache: Dict[str, dict] = {}

        # Import librosa lazily for faster startup
        self._librosa = None

    @property
    def librosa(self):
        """Lazy import librosa using thread-local storage"""
        return _get_librosa()

    def _get_file_info(self, file_path: Path) -> dict:
        """Get file info with caching"""
        cache_key = str(file_path)
        if cache_key not in self._file_info_cache:
            info = sf.info(str(file_path))
            self._file_info_cache[cache_key] = {
                'samplerate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'duration': info.duration,
            }
        return self._file_info_cache[cache_key]

    def load(
        self,
        file_path: Union[str, Path],
        start: int = 0,
        stop: Optional[int] = None,
        use_memory_cache: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with resampling and caching

        - Files with matching sample rate: read directly with memory cache
        - Files needing resampling: resample entire audio, cache as WAV,
          then support partial reading from cache

        Args:
            file_path: Path to audio file
            start: Start sample index (default: 0)
            stop: End sample index, exclusive (default: None = end of file)
            use_memory_cache: Whether to use memory cache (default: True)

        Returns:
            Tuple of (waveform, sample_rate)
        """
        file_path = Path(file_path)
        cache_key = f"{file_path}:{start}:{stop}"

        # Check memory cache first
        if use_memory_cache:
            cached = self._memory_cache.get(cache_key)
            if cached is not None:
                return cached, self.target_sr

        # Get audio info (cached)
        info = self._get_file_info(file_path)
        orig_sr = info['samplerate']

        # Fast path: sample rate matches, read directly without caching
        if orig_sr == self.target_sr:
            waveform = sf.read(file_path, dtype='float32', start=start, stop=stop)[0]
            if self.force_mono and waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            # Cache in memory if requested
            if use_memory_cache:
                self._memory_cache.put(cache_key, waveform)

            return waveform, self.target_sr

        # Slow path: need resampling
        waveform = self._load_with_file_cache(file_path, orig_sr, start, stop)

        # Cache in memory if requested
        if use_memory_cache:
            self._memory_cache.put(cache_key, waveform)

        return waveform, self.target_sr

    def _load_with_file_cache(
        self,
        file_path: Path,
        orig_sr: int,
        start: int,
        stop: Optional[int]
    ) -> np.ndarray:
        """Load audio using file cache for resampled audio"""
        if self.cache_dir:
            cache_path = self._get_cache_path(file_path, orig_sr)
            # Check if cached WAV exists and is valid
            if cache_path.exists():
                # Verify cache is not stale (check mtime)
                cache_meta_path = cache_path.with_suffix('.json')
                if cache_meta_path.exists():
                    with open(cache_meta_path, 'r') as f:
                        meta = json.load(f)
                    # Cache is valid if source hasn't been modified
                    if meta.get('source_mtime') == file_path.stat().st_mtime_ns:
                        waveform = sf.read(cache_path, dtype='float32', start=start, stop=stop)[0]
                        if self.force_mono and waveform.ndim > 1:
                            waveform = np.mean(waveform, axis=1)
                        return waveform

        # Load entire audio, resample, and cache
        waveform, _ = sf.read(file_path, dtype='float32')

        if self.force_mono and waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample using librosa (much faster than scipy)
        waveform = self._resample(waveform, orig_sr, self.target_sr)

        # Save as WAV for partial reading support
        if self.cache_dir is not None:
            self._save_to_cache(cache_path, waveform, file_path, orig_sr)
            # Now read partial from cache
            if start != 0 or stop is not None:
                waveform = sf.read(cache_path, dtype='float32', start=start, stop=stop)[0]
                if self.force_mono and waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)

        return waveform

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
        pad_value: float = 0.0,
        num_workers: int = 4,
        use_memory_cache: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load multiple audio files into a batch with parallel processing

        Args:
            file_paths: List of audio file paths
            max_length: Maximum length in samples (pad/truncate if specified)
            pad_value: Value to use for padding
            num_workers: Number of parallel workers for loading (default: 4)
            use_memory_cache: Whether to use memory cache (default: True)

        Returns:
            Tuple of (batch_waveform, sample_rate)
        """
        # Use ThreadPoolExecutor for parallel I/O
        if num_workers > 1 and len(file_paths) > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(self.load, fp, 0, None, use_memory_cache)
                    for fp in file_paths
                ]
                waveforms = [f.result()[0] for f in futures]
        else:
            waveforms = [self.load(fp, 0, None, use_memory_cache)[0] for fp in file_paths]

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
        """Resample audio using librosa (much faster than scipy)"""
        if orig_sr == target_sr:
            return waveform

        librosa = _get_librosa()

        # Use librosa.resample which is much faster
        if waveform.ndim > 1:
            # Multi-channel: resample each channel
            resampled = np.stack([
                librosa.resample(waveform[ch], orig_sr=orig_sr, target_sr=target_sr)
                for ch in range(waveform.shape[0])
            ], axis=0)
        else:
            resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

        return resampled.astype(np.float32)

    def _get_cache_path(self, file_path: Path, orig_sr: int) -> Path:
        """
        Get cache file path using file mtime (much faster than MD5 hash)

        Uses file stem + mtime + sample rate as cache key
        """
        # Use mtime for cache invalidation (nanosecond precision)
        mtime = file_path.stat().st_mtime_ns
        # Use file stem to make cache filename readable
        safe_stem = "".join(c if c.isalnum() or c in '-_' else '_' for c in file_path.stem)
        return self.cache_dir / f"{safe_stem}_{mtime}_{orig_sr}to{self.target_sr}hz.wav"

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

        # Save metadata with mtime for cache validation
        meta_path = cache_path.with_suffix('.json')
        meta = {
            'source_path': str(source_path),
            'source_mtime': source_path.stat().st_mtime_ns,
            'orig_sr': orig_sr,
            'target_sr': self.target_sr,
            'shape': list(waveform.shape),
            'dtype': str(waveform.dtype),
            'cache_format': 'wav'
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

    def clear_cache(self) -> dict:
        """Clear all cached audio files (file cache and memory cache)"""
        # Clear memory cache
        self._memory_cache.clear()
        self._file_info_cache.clear()

        # Clear file cache
        file_count = 0
        if self.cache_dir and self.cache_dir.exists():
            for ext in ['*.wav', '*.npy']:
                for file in self.cache_dir.glob(ext):
                    file.unlink()
                    file_count += 1

            for file in self.cache_dir.glob('*.json'):
                file.unlink()

        return {
            'memory_cache_cleared': True,
            'file_cache_cleared': file_count,
            'file_info_cache_cleared': True
        }

    def get_cache_info(self) -> dict:
        """Get cache statistics (both memory and file cache)"""
        memory_stats = self._memory_cache.get_stats()

        if not self.cache_dir:
            return {
                'memory_cache': memory_stats,
                'file_cache': {
                    'enabled': False,
                    'cache_dir': None,
                    'file_count': 0,
                    'total_size_mb': 0
                }
            }

        if not self.cache_dir.exists():
            return {
                'memory_cache': memory_stats,
                'file_cache': {
                    'enabled': True,
                    'cache_dir': str(self.cache_dir),
                    'file_count': 0,
                    'total_size_mb': 0
                }
            }

        # Count both wav and npy files
        cache_files = list(self.cache_dir.glob('*.wav')) + list(self.cache_dir.glob('*.npy'))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'memory_cache': memory_stats,
            'file_cache': {
                'enabled': True,
                'cache_dir': str(self.cache_dir),
                'file_count': len(cache_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        }

    def __repr__(self) -> str:
        return (
            f"AudioReader(target_sr={self.target_sr}, "
            f"cache_dir={self.cache_dir}, "
            f"force_mono={self.force_mono}, "
            f"memory_cache_mb={self._memory_cache.max_memory_bytes / (1024*1024):.0f})"
        )
