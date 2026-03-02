"""
Configuration classes for Baby Cry Detection
Defines configuration dataclasses for feature extraction and data loading
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AugmentationConfig:
    mixup_alpha: Optional[float] = None
    mask_rate: Optional[float] = None
    noise_rate: Optional[float] = None
    pitch_shift: Optional[float] = None
    reverb_rate: Optional[float] = None
    gain_db: Optional[float] = None


@dataclass
class FeatureConfig:
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    fmin: int = 250
    fmax: Optional[int] = 8000


@dataclass
class DatasetConfig:
    sample_rate: int = 16000
    duration: float = 10.0
    stride: float = 1
    cache_dir: Optional[str] = './audio_cache'
    use_cache: bool = True
    force_mono: bool = True
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)

