"""
Configuration classes for Baby Cry Detection
Defines configuration dataclasses for feature extraction, model, and training
"""

from dataclasses import dataclass, field, asdict, fields, MISSING
from typing import Optional, Dict, Any, Type, TypeVar, ClassVar, get_origin, get_args
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
class LossConfig:
    """Loss function configuration"""
    # Loss type: 'cross_entropy', 'focal', 'label_smoothing', 'combined', 'ohem_focal', 'ohem_ce'
    loss_type: str = 'combined'

    # Focal loss parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Label smoothing parameters
    label_smoothing: float = 0.1

    # Combined loss weight
    focal_weight: float = 0.5

    # OHEM parameters (used when loss_type starts with 'ohem_')
    ohem_hard_ratio: float = 0.25  # Ratio of hard examples to keep
    ohem_min_hard_num: int = 4  # Minimum number of hard examples per batch


@dataclass
class MixupConfig:
    """Mixup augmentation configuration"""
    cry_mix_prob: float = 0.3
    cry_mix_rate_mean: float = 0.3
    cry_mix_rate_std: float = 0.15
    other_mix_prob: float = 0.3
    mix_front_prob: float = 0.7


@dataclass
class NoiseConfig:
    """Noise augmentation configuration"""
    # Noise type probabilities (relative weights, will be normalized)
    white_noise_prob: float = 0.3
    pink_noise_prob: float = 0.4
    ambient_noise_prob: float = 0.3

    # SNR range for noise mixing (in dB)
    snr_min: float = 5.0
    snr_max: float = 25.0

    # Ambient noise settings
    ambient_noise_dir: Optional[str] = None  # Directory containing ambient noise files
    ambient_noise_files: tuple = field(default_factory=tuple)  # List of ambient noise file paths

    # Pink noise characteristics
    pink_noise_alpha: float = 1.0  # 1/f^alpha, alpha=1 for classic pink noise


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    mixup: MixupConfig = field(default_factory=MixupConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

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

    _EFFECT_MAP: ClassVar[Dict[str, str]] = {
        'pitch': 'pitch_prob',
        'reverb': 'reverb_prob',
        'phaser': 'phaser_prob',
        'echo': 'echo_prob',
        'noise': 'noise_prob',
        'gain': 'gain_prob',
    }

    def __getitem__(self, key: str) -> float:
        """Support indexing access for effect probabilities."""
        attr = self._EFFECT_MAP.get(key, key)
        if hasattr(self, attr):
            return getattr(self, attr)
        raise KeyError(f"Unknown key: {key}")


@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    # Feature type: 'fbank', 'mfcc', 'fft', 'all'
    feature_type: str = 'fbank'

    # FFT parameters
    n_fft: int = 1024
    hop_length: int = 500  # ~31.25ms frame shift at 16kHz

    # Mel filterbank parameters
    n_mels: int = 32
    n_mfcc: int = 16
    fmin: int = 250
    fmax: int = 8000

    # Preemphasis
    preemphasis: float = 0.95

    # Normalization
    use_fbank_norm: bool = True
    fbank_decay: float = 0.9  # Exponential smoothing decay

    # Energy feature
    use_db_feature: bool = False  # Add energy (dB) feature as additional channel
    use_db_norm: bool = False  # Normalize db features to [0, 1] range

    # Output configuration
    use_delta: bool = False  # Time delta features
    use_freq_delta: bool = False  # Frequency delta features

    @property
    def feature_dim(self) -> int:
        """Return total feature dimension based on configuration."""
        # Base feature dimension
        if self.feature_type == 'fbank':
            base_dim = self.n_mels
        elif self.feature_type == 'mfcc':
            base_dim = self.n_mfcc
        else:  # 'all' - both fbank and mfcc
            base_dim = self.n_mels + self.n_mfcc

        # Total dimension = base + optional deltas + optional db features
        dim = base_dim
        if self.use_delta:
            dim += base_dim  # Time delta doubles the dimension
        if self.use_freq_delta:
            dim += base_dim  # Frequency delta adds another base_dim
        if self.use_db_feature:
            dim += 2  # DB feature adds 2 channels (avg + weighted energy)

        return dim

    @property
    def frames_per_second(self) -> int:
        """Return frames per second"""
        return 16000 // self.hop_length


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    audio_suffixes: tuple = ('.wav',)
    sample_rate: int = 16000
    slice_len: float = 5.0
    stride: float = 3.0
    cry_rate: float = 0.5
    cache_dir: Optional[str] = './audio_cache'
    force_mono: bool = True

    def __post_init__(self):
        """Ensure audio_suffixes is always a tuple"""
        if isinstance(self.audio_suffixes, list):
            self.audio_suffixes = tuple(self.audio_suffixes)


@dataclass
class ModelConfig:
    """Model configuration for Transformer-based cry detection

    Model size is controlled by architecture parameters (d_model, n_layers, etc.)
    rather than preset variants. The attention_type and ffn_type are auto-selected
    based on model size unless explicitly specified.

    Size guidelines:
    - Large model: d_model >= 512, n_layers >= 8
    - Medium model: d_model 128-512, n_layers 3-8
    - Tiny model: d_model < 128 or n_layers < 3
    """
    model_type: str = 'transformer'
    num_classes: int = 2

    # Transformer core parameters - control model size with these
    d_model: int = 256          # Hidden dimension (larger = more capacity)
    n_heads: int = 4            # Attention heads (must divide d_model)
    n_layers: int = 6           # Number of transformer layers (deeper = more capacity)
    d_ff: int = 1024            # Feed-forward dimension (typically 2-4x d_model)
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Positional encoding
    max_seq_len: int = 200
    use_relative_pos: bool = True

    # Attention variant: 'standard', 'linear', 'depthwise'
    # If 'auto', will be selected based on model size
    attention_type: str = 'auto'

    # FFN variant: 'standard', 'inverted_bottleneck'
    # If 'auto', will be selected based on model size
    ffn_type: str = 'auto'

    # Pooling: 'mean', 'max', 'attention'
    pool_type: str = 'mean'

    # Label smoothing for training
    label_smoothing: float = 0.1

    def __post_init__(self):
        """Validate model configuration and auto-select efficient settings"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        # Auto-select attention type based on model size
        if self.attention_type == 'auto':
            self.attention_type = self._select_attention_type()

        # Auto-select FFN type based on model size
        if self.ffn_type == 'auto':
            self.ffn_type = self._select_ffn_type()

    def _select_attention_type(self) -> str:
        """Select attention type based on model size"""
        total_params = self.d_model * self.n_layers
        if total_params >= 3000:  # d_model * n_layers >= 3000 (e.g., 512*6, 256*12)
            return 'standard'  # Full attention for large models
        elif total_params >= 1000:  # Medium models
            return 'standard'  # Or 'linear' for efficiency
        else:  # Small models
            return 'depthwise'  # Efficient depthwise separable attention

    def _select_ffn_type(self) -> str:
        """Select FFN type based on model size"""
        _ = self.d_model * self.n_layers  # For potential future use
        if self.d_model >= 512:
            return 'standard'  # Full FFN for large models
        else:
            return 'inverted_bottleneck'  # Efficient FFN for smaller models

    @property
    def variant(self) -> str:
        """Derive variant name from architecture parameters"""
        total_params = self.d_model * self.n_layers
        if self.d_model >= 512 or self.n_layers >= 10:
            return 'large'
        elif self.d_model <= 128 or self.n_layers <= 3:
            return 'tiny'
        else:
            return 'medium'

    @property
    def estimated_params(self) -> int:
        """Estimate total parameter count"""
        # Linear projection + Transformer layers + Classifier
        # Linear projection: in_features (64/128/192) -> d_model
        projection_params = 64 * self.d_model + self.d_model  # weights + bias
        layer_params = self.n_layers * (
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            2 * self.d_model * self.d_ff       # FFN
        )
        classifier_params = self.d_model * self.num_classes
        return projection_params + layer_params + classifier_params

    @property
    def size_category(self) -> str:
        """Return model size category for deployment guidance"""
        params = self.estimated_params
        if params > 20_000_000:
            return 'server/cloud (GPU recommended)'
        elif params > 5_000_000:
            return 'edge device (ARM CPU/GPU)'
        elif params > 1_000_000:
            return 'embedded (Raspberry Pi, Edge TPU)'
        else:
            return 'microcontroller (Cortex-M, ESP32)'

    @classmethod
    def large(cls) -> 'ModelConfig':
        """High-performance configuration for server/cloud deployment"""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=12,
            d_ff=2048,
            dropout=0.1,
            attention_type='standard',
            ffn_type='standard',
        )

    @classmethod
    def medium(cls) -> 'ModelConfig':
        """Lightweight configuration for edge devices (Raspberry Pi, Edge TPU)"""
        return cls(
            d_model=256,
            n_heads=4,
            n_layers=6,
            d_ff=1024,
            dropout=0.1,
            attention_type='standard',
            ffn_type='inverted_bottleneck',
        )

    @classmethod
    def tiny(cls) -> 'ModelConfig':
        """Ultra-lightweight configuration for MCU (Cortex-M, ESP32)"""
        return cls(
            d_model=128,
            n_heads=2,
            n_layers=3,
            d_ff=256,
            dropout=0.1,
            attention_type='depthwise',
            ffn_type='inverted_bottleneck',
        )


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 1.0
    early_stopping_patience: Optional[int] = 10
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    use_augmentation: bool = True
    device: str = 'cuda'
    log_interval: int = 10
    val_interval: int = 1
    save_best_only: bool = True

    # Optimizer settings
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'  # 'step', 'cosine', 'plateau', 'cosine_warmup', 'none'
    warmup_epochs: int = 5
    warmup_steps: int = 0  # If > 0, use steps instead of epochs for warmup
    min_lr: float = 1e-6
    lr_decay_epochs: int = 0  # For cosine_warmup: epochs for cosine decay after warmup (0 = auto)

    # Loss configuration
    loss: LossConfig = field(default_factory=LossConfig)

    # SpecAugment settings
    use_spec_augment: bool = True
    spec_augment_freq_mask: int = 8
    spec_augment_time_mask: int = 20

    # SWA (Stochastic Weight Averaging)
    use_swa: bool = False
    swa_start: int = 70
    swa_lr: float = 1e-4

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.9999  # EMA decay rate (closer to 1 = slower update)


@dataclass
class Config:
    """Main configuration container"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
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


def _as_dict(obj: Any) -> Any:
    """Convert dataclass to dict, handling special types like tuples."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _as_dict(v) for k, v in asdict(obj).items() if not k.startswith('_')}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def save_config(config: Config, save_path: str) -> None:
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = _as_dict(config)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
