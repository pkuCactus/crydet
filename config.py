"""
Configuration classes for Baby Cry Detection
Defines configuration dataclasses for feature extraction and data loading
"""

from dataclasses import dataclass, field
from typing import Optional


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
    audio_suffixes: tuple = ('.wav', '.mp3', '.flac')
    sample_rate: int = 16000
    duration: float = 10.0
    stride: float = 1
    cache_dir: Optional[str] = './audio_cache'
    use_cache: bool = True
    force_mono: bool = True
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = None
    early_stopping_patience: Optional[int] = 10
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    use_augmentation: bool = True
