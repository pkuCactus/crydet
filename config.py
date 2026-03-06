"""
Configuration classes for Baby Cry Detection
Defines configuration dataclasses for feature extraction, model, and training
"""

from dataclasses import dataclass, field, asdict, fields, MISSING
from typing import Optional, Dict, Any, Type, TypeVar, get_origin, get_args
from pathlib import Path
import yaml


T = TypeVar('T')


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Generic function to convert dict to dataclass instance.
    Handles nested dataclasses and Optional types automatically.

    Args:
        cls: Target dataclass type
        data: Dictionary with configuration values

    Returns:
        Instance of cls with values from data
    """
    if data is None:
        data = {}

    if not hasattr(cls, '__dataclass_fields__'):
        return data

    result = {}
    for f in fields(cls):
        if f.name.startswith('_'):
            continue

        field_type = f.type
        value = data.get(f.name, MISSING)

        # Handle Optional types
        origin = get_origin(field_type)
        if origin is Optional:
            field_type = get_args(field_type)[0]

        # Handle nested dataclass
        if hasattr(field_type, '__dataclass_fields__'):
            nested_data = data.get(f.name, {})
            result[f.name] = from_dict(field_type, nested_data)
        elif value is not MISSING:
            result[f.name] = value
        elif f.default is not MISSING:
            result[f.name] = f.default
        elif f.default_factory is not MISSING:
            result[f.name] = f.default_factory()

    return cls(**result)


@dataclass
class MixupConfig:
    """Mixup augmentation configuration"""
    cry_mix_prob: float = 0.3
    cry_mix_rate_mean: float = 0.3
    cry_mix_rate_std: float = 0.15
    other_mix_prob: float = 0.3
    mix_front_prob: float = 0.7


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    mixup: MixupConfig = field(default_factory=MixupConfig)

    # Augmentation probabilities by label type
    cry_aug_prob: float = 0.9
    other_aug_prob: float = 0.6
    other_reverse_prob: float = 0.5

    # Individual effect probabilities
    pitch_prob: float = 0.5
    reverb_prob: float = 0.8
    phaser_prob: float = 0.5
    echo_prob: float = 0.5
    noise_prob: float = 0.1
    gain_prob: float = 0.9

    _key_map: Dict[str, str] = field(
        default_factory=lambda: {
            'pitch': 'pitch_prob',
            'reverb': 'reverb_prob',
            'phaser': 'phaser_prob',
            'echo': 'echo_prob',
            'gain': 'gain_prob',
        },
        repr=False,
        compare=False
    )

    def __getitem__(self, key: str) -> float:
        """Support indexing access for effect probabilities."""
        key = self._key_map.get(key, key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Unknown key: {key}")


@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    feature_type: str = 'fbank'
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
        """Return feature dimension based on feature type"""
        return self.n_mfcc if self.feature_type == 'mfcc' else self.n_mels

    @property
    def num_channels(self) -> int:
        """Return number of output channels (including delta features)"""
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
    slice_len: float = 5.0
    stride: float = 3.0
    cry_rate: float = 0.5
    cache_dir: Optional[str] = './audio_cache'
    force_mono: bool = True


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
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with loaded settings
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f) or {}

    return from_dict(Config, yaml_config)


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file.

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
            'slice_len': config.dataset.slice_len,
            'stride': config.dataset.stride,
            'cry_rate': config.dataset.cry_rate,
            'cache_dir': config.dataset.cache_dir,
            'force_mono': config.dataset.force_mono,
        },
        'augmentation': asdict(config.augmentation),
        'model': asdict(config.model),
        'training': asdict(config.training),
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
