"""
Model variants and factory functions for CryTransformer
"""

from typing import Optional, Dict

from config import ModelConfig
from model.transformer import CryTransformer, CryTransformerWithSpecAugment


# Predefined model configurations - convenience presets for common use cases
# These can still be used with the 'variant' parameter for quick prototyping
MODEL_CONFIGS = {
    'large': {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 12,
        'd_ff': 2048,
        'attention_type': 'auto',  # Will be auto-selected
        'ffn_type': 'auto',
        'pool_type': 'mean',
        'patch_size': 3,
        'patch_stride': 2,
        'use_relative_pos': True,
    },
    'medium': {
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 6,
        'd_ff': 1024,
        'attention_type': 'auto',
        'ffn_type': 'auto',
        'pool_type': 'mean',
        'patch_size': 3,
        'patch_stride': 2,
        'use_relative_pos': True,
    },
    'tiny': {
        'd_model': 128,
        'n_heads': 2,
        'n_layers': 3,
        'd_ff': 256,
        'attention_type': 'auto',
        'ffn_type': 'auto',
        'pool_type': 'mean',
        'patch_size': 5,
        'patch_stride': 3,
        'use_relative_pos': False,
    },
    'nano': {
        'd_model': 64,
        'n_heads': 2,
        'n_layers': 2,
        'd_ff': 128,
        'attention_type': 'auto',
        'ffn_type': 'auto',
        'pool_type': 'mean',
        'patch_size': 5,
        'patch_stride': 3,
        'use_relative_pos': False,
    }
}


def create_model(
    config: ModelConfig,
    in_channels: int = 64,
    num_classes: int = 2,
    use_spec_augment: bool = False,
) -> CryTransformer:
    """
    Factory function to create a CryTransformer model.

    Model architecture is fully controlled by the config parameter.
    Use MODEL_CONFIGS presets for quick prototyping, or create custom configs
    for fine-grained control over model size.

    Args:
        config: ModelConfig containing all architecture parameters
        in_channels: Number of input feature channels (e.g., 64 for mel bins)
        num_classes: Number of output classes
        use_spec_augment: Whether to use SpecAugment during training

    Returns:
        CryTransformer model instance

    Examples:
        >>> # Create model with custom config (recommended approach)
        >>> config = ModelConfig(d_model=192, n_layers=4, n_heads=4)
        >>> model = create_model(config)

        >>> # Create with preset config
        >>> config = ModelConfig(**MODEL_CONFIGS['medium'])
        >>> model = create_model(config)
    """
    if use_spec_augment:
        return CryTransformerWithSpecAugment(config, in_channels, num_classes)
    else:
        return CryTransformer(config, in_channels, num_classes)


def create_model_from_variant(
    variant: str = 'medium',
    in_channels: int = 64,
    num_classes: int = 2,
    use_spec_augment: bool = False,
    **kwargs
) -> CryTransformer:
    """
    Convenience function to create a model from a preset variant.

    For production use, prefer create_model() with explicit ModelConfig.

    Args:
        variant: Model variant ('large', 'medium', 'tiny', 'nano')
        in_channels: Number of input feature channels
        num_classes: Number of output classes
        use_spec_augment: Whether to use SpecAugment
        **kwargs: Additional config overrides

    Returns:
        CryTransformer model instance
    """
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}")

    config_dict = MODEL_CONFIGS[variant].copy()
    config_dict.update(kwargs)
    config = ModelConfig(**config_dict)

    return create_model(config, in_channels, num_classes, use_spec_augment)


def list_models() -> Dict[str, Dict]:
    """
    List all available model preset variants.

    Returns:
        Dictionary mapping variant names to their configs
    """
    return MODEL_CONFIGS.copy()


def get_model_info(model: CryTransformer) -> Dict:
    """
    Get information about a model.

    Args:
        model: CryTransformer instance

    Returns:
        Dictionary with model information
    """
    trainable, total = model.count_parameters()
    macs = model.estimate_macs()

    return {
        'variant': model.config.variant,
        'size_category': model.config.size_category,
        'd_model': model.config.d_model,
        'n_heads': model.config.n_heads,
        'n_layers': model.config.n_layers,
        'd_ff': model.config.d_ff,
        'attention_type': model.config.attention_type,
        'ffn_type': model.config.ffn_type,
        'estimated_params': model.config.estimated_params,
        'trainable_params': trainable,
        'total_params': total,
        'model_size_mb': total * 4 / (1024 * 1024),  # Assuming float32
        'estimated_macs': macs,
        'estimated_macs_m': macs / 1e6,
    }


def print_model_summary(model: CryTransformer):
    """Print a summary of the model"""
    info = get_model_info(model)

    print("=" * 60)
    print(f"CryTransformer Model Summary")
    print("=" * 60)
    print(f"Variant:           {info['variant']}")
    print(f"Size Category:     {info['size_category']}")
    print(f"Model Dimension:   {info['d_model']}")
    print(f"Attention Heads:   {info['n_heads']}")
    print(f"Encoder Layers:    {info['n_layers']}")
    print(f"FFN Dimension:     {info['d_ff']}")
    print(f"Attention Type:    {info['attention_type']}")
    print(f"FFN Type:          {info['ffn_type']}")
    print("-" * 60)
    print(f"Estimated Params:  {info['estimated_params']:,}")
    print(f"Trainable Params:  {info['trainable_params']:,}")
    print(f"Total Params:      {info['total_params']:,}")
    print(f"Model Size:        {info['model_size_mb']:.2f} MB (float32)")
    print(f"Estimated MACs:    {info['estimated_macs_m']:.1f} M")
    print("=" * 60)
