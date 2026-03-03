"""
Configuration classes for Baby Cry Detection
Defines configuration dataclasses for feature extraction, model, and training
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    mixup_alpha: Optional[float] = None
    mask_rate: Optional[float] = None
    noise_rate: Optional[float] = None
    pitch_shift: Optional[float] = None
    reverb_rate: Optional[float] = None
    gain_db: Optional[float] = None


@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    feature_type: str = 'fbank'  # 'mfcc' or 'fbank'
    n_mfcc: Optional[int] = 40
    n_mels: Optional[int] = 64
    n_fft: int = 1024
    hop_length: int = 512
    fmin: int = 250
    fmax: Optional[int] = 8000
    normalize: bool = True
    use_delta: bool = True
    use_freq_delta: bool = True

    @property
    def feature_dim(self) -> int:
        """Return feature dimension"""
        return self.n_mfcc if self.feature_type == 'mfcc' else self.n_mels

    @property
    def num_channels(self) -> int:
        """Return number of output channels"""
        channels = 1
        if self.use_delta:
            channels += 1
        if self.use_freq_delta:
            channels += 1
        return channels


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    audio_suffixes: tuple = ('.wav', '.mp3', '.flac')
    sample_rate: int = 16000
    duration: float = 10.0
    stride: float = 1.0
    cry_rate: float = 0.5
    cache_dir: Optional[str] = './audio_cache'
    use_cache: bool = True
    force_mono: bool = True
    aug_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = 'transformer'
    num_classes: int = 2
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    d_ff: int = 512
    dropout: float = 0.1

    def __post_init__(self):
        """Validate model configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class TrainingConfig:
    """Training configuration"""
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
    device: str = 'cuda'
    log_interval: int = 10
    val_interval: int = 1
    save_best_only: bool = True


@dataclass
class Config:
    """Main configuration container"""
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with loaded settings
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    return _dict_to_config(yaml_config)


def _dict_to_config(yaml_dict: Dict[str, Any]) -> Config:
    """Convert dictionary to Config object"""
    # Parse feature config
    feature_dict = yaml_dict.get('feature', {})
    feature_config = FeatureConfig(
        feature_type=feature_dict.get('feature_type', 'mfcc'),
        n_mfcc=feature_dict.get('n_mfcc', 40),
        n_mels=feature_dict.get('n_mels', 64),
        n_fft=feature_dict.get('n_fft', 1024),
        hop_length=feature_dict.get('hop_length', 512),
        fmin=feature_dict.get('fmin', 250),
        fmax=feature_dict.get('fmax', 8000),
        normalize=feature_dict.get('normalize', True),
        use_delta=feature_dict.get('use_delta', True),
        use_freq_delta=feature_dict.get('use_freq_delta', True),
    )

    # Parse augmentation config
    aug_dict = yaml_dict.get('augmentation', {})
    aug_config = AugmentationConfig(
        mixup_alpha=aug_dict.get('mixup_alpha'),
        mask_rate=aug_dict.get('mask_rate'),
        noise_rate=aug_dict.get('noise_rate'),
        pitch_shift=aug_dict.get('pitch_shift'),
        reverb_rate=aug_dict.get('reverb_rate'),
        gain_db=aug_dict.get('gain_db'),
    )

    # Parse dataset config
    data_dict = yaml_dict.get('dataset', {})
    dataset_config = DatasetConfig(
        audio_suffixes=tuple(data_dict.get('audio_suffixes', ('.wav', '.mp3', '.flac'))),
        sample_rate=data_dict.get('sample_rate', 16000),
        duration=data_dict.get('duration', 10.0),
        stride=data_dict.get('stride', 1.0),
        cry_rate=data_dict.get('cry_rate', 0.5),
        cache_dir=data_dict.get('cache_dir', './audio_cache'),
        use_cache=data_dict.get('use_cache', True),
        force_mono=data_dict.get('force_mono', True),
        aug_config=aug_config,
        feature_config=feature_config,
    )

    # Parse model config
    model_dict = yaml_dict.get('model', {})
    model_config = ModelConfig(
        model_type=model_dict.get('model_type', 'transformer'),
        num_classes=model_dict.get('num_classes', 2),
        d_model=model_dict.get('d_model', 256),
        n_heads=model_dict.get('n_heads', 4),
        n_layers=model_dict.get('n_layers', 8),
        d_ff=model_dict.get('d_ff', 512),
        dropout=model_dict.get('dropout', 0.1),
    )

    # Parse training config
    train_dict = yaml_dict.get('training', {})
    training_config = TrainingConfig(
        batch_size=train_dict.get('batch_size', 32),
        num_workers=train_dict.get('num_workers', 4),
        pin_memory=train_dict.get('pin_memory', True),
        num_epochs=train_dict.get('num_epochs', 100),
        learning_rate=train_dict.get('learning_rate', 1e-3),
        weight_decay=train_dict.get('weight_decay', 1e-5),
        grad_clip=train_dict.get('grad_clip'),
        early_stopping_patience=train_dict.get('early_stopping_patience', 10),
        checkpoint_dir=train_dict.get('checkpoint_dir', './checkpoints'),
        log_dir=train_dict.get('log_dir', './logs'),
        use_augmentation=train_dict.get('use_augmentation', True),
        device=train_dict.get('device', 'cuda'),
        log_interval=train_dict.get('log_interval', 10),
        val_interval=train_dict.get('val_interval', 1),
        save_best_only=train_dict.get('save_best_only', True),
    )

    return Config(
        feature=feature_config,
        dataset=dataset_config,
        model=model_config,
        training=training_config,
    )


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Config object to save
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        'feature': asdict(config.feature),
        'dataset': {
            'audio_suffixes': list(config.dataset.audio_suffixes),
            'sample_rate': config.dataset.sample_rate,
            'duration': config.dataset.duration,
            'stride': config.dataset.stride,
            'cry_rate': config.dataset.cry_rate,
            'cache_dir': config.dataset.cache_dir,
            'use_cache': config.dataset.use_cache,
            'force_mono': config.dataset.force_mono,
        },
        'augmentation': asdict(config.dataset.aug_config),
        'model': asdict(config.model),
        'training': asdict(config.training),
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


# Convenience function to get default config
def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
