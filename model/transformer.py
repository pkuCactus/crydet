"""
CryTransformer - Transformer-based model for baby cry detection

Supports multiple model variants for different deployment scenarios:
- Large: High-performance for server/cloud
- Medium: Lightweight for edge devices (Raspberry Pi, Edge TPU)
- Tiny: Ultra-lightweight for MCU (Cortex-M, ESP32)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from model.layers import (
    PatchEmbedding,
    SinusoidalPositionalEncoding,
    TransformerEncoderLayer,
    PoolingLayer
)


class CryTransformer(nn.Module):
    """
    Transformer-based audio classifier for baby cry detection.

    Architecture:
        1. Patch Embedding: Conv1d projects mel-spectrogram to embeddings
        2. Positional Encoding: Sinusoidal or learned positional encoding
        3. Transformer Encoder: Stack of attention + FFN layers
        4. Pooling: Global average/max/attention pooling
        5. Classifier: Linear projection to class logits

    Args:
        config: ModelConfig containing model hyperparameters
        in_channels: Number of input feature channels (e.g., 64 for mel bins)
        num_classes: Number of output classes (default: 2 for cry/non-cry)
    """

    def __init__(
        self,
        config: ModelConfig,
        in_channels: int = 64,
        num_classes: int = 2
    ):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Patch embedding: (B, C, T) -> (B, T', D)
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            d_model=config.d_model,
            patch_size=config.patch_size,
            stride=config.patch_stride,
            dropout=config.dropout
        )

        # Positional encoding
        if config.use_relative_pos:
            # Relative PE is handled inside attention layers
            self.pos_encoding = None
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
                dropout=config.dropout
            )

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                attention_type=config.attention_type,
                ffn_type=config.ffn_type,
                use_relative_pos=config.use_relative_pos,
                max_seq_len=config.max_seq_len
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)

        # Pooling layer
        self.pooling = PoolingLayer(config.d_model, config.pool_type)

        # Classification head
        self.classifier = nn.Linear(config.d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input spectrogram.

        Args:
            x: Input spectrogram of shape (B, C, T) where
               B = batch size, C = mel bins (e.g., 64), T = time frames (e.g., 157)

        Returns:
            Feature tensor of shape (B, d_model)
        """
        # Patch embedding: (B, C, T) -> (B, T', D)
        x = self.patch_embed(x)

        # Add positional encoding (if not using relative positions)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Final layer norm
        x = self.norm(x)

        # Pooling: (B, T', D) -> (B, D)
        x = self.pooling(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram of shape (B, C, T) or (B, T, C)
               Will automatically handle channel-last inputs

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Handle channel-last input (B, T, C) -> (B, C, T)
        if x.dim() == 3 and x.size(1) > x.size(2):
            # Likely (B, T, C) where T > C
            x = x.transpose(1, 2)

        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Extract attention maps from all layers for visualization.

        Args:
            x: Input spectrogram

        Returns:
            List of attention weight tensors from each layer
        """
        attention_maps = []

        # Patch embedding
        x = self.patch_embed(x)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Extract attention weights from each layer
        for layer in self.encoder_layers:
            # Hook into attention module to get weights
            attn_weights = []

            def hook_fn(module, input, output):
                # Store attention weights (implementation depends on attention type)
                pass

            handle = layer.attn.register_forward_hook(hook_fn)
            x = layer(x)
            handle.remove()

        return attention_maps

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count model parameters.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def estimate_macs(self, input_shape: Tuple[int, ...] = (1, 64, 157)) -> int:
        """
        Estimate MACs (multiply-accumulate operations) for a forward pass.

        Args:
            input_shape: Input tensor shape (B, C, T)

        Returns:
            Estimated number of MACs
        """
        # This is a rough estimation
        B, C, T = input_shape

        # Patch embedding
        T_out = T // self.config.patch_stride
        patch_macs = C * self.config.d_model * self.config.patch_size * T_out

        # Self-attention per layer: O(n²·d) where n = seq_len, d = d_model
        n = T_out
        d = self.config.d_model
        attn_macs_per_layer = 2 * n * n * d  # Q@K.T and attn@V

        # FFN per layer: O(n·d·d_ff)
        ffn_macs_per_layer = 2 * n * d * self.config.d_ff

        # Total per encoder layer
        layer_macs = attn_macs_per_layer + ffn_macs_per_layer

        # All layers + classifier
        total_macs = patch_macs + self.config.n_layers * layer_macs + d * self.num_classes

        return total_macs


class CryTransformerWithSpecAugment(CryTransformer):
    """CryTransformer with built-in SpecAugment for training"""

    def __init__(
        self,
        config: ModelConfig,
        in_channels: int = 64,
        num_classes: int = 2,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
        num_freq_masks: int = 1,
        num_time_masks: int = 1
    ):
        super().__init__(config, in_channels, num_classes)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input spectrogram.

        Args:
            x: Input spectrogram (B, C, T)

        Returns:
            Augmented spectrogram
        """
        if not self.training:
            return x

        B, C, T = x.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, C - f), (1,)).item()
            x[:, f0:f0+f, :] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            x[:, :, t0:t0+t] = 0

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SpecAugment"""
        x = self.spec_augment(x)
        return super().forward(x)
