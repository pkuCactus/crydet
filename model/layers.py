"""
Custom layers for Transformer-based cry detection
Includes various attention mechanisms and FFN variants
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for better sequence modeling"""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Learnable relative position embeddings
        self.rel_pos_embed = nn.Embedding(2 * max_len - 1, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate relative position bias for attention"""
        # Create relative position matrix
        positions = torch.arange(seq_len, device=self.rel_pos_embed.weight.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        rel_pos = rel_pos + self.max_len - 1  # Shift to [0, 2*max_len-1]
        return self.rel_pos_embed(rel_pos)  # (seq_len, seq_len, d_model)


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_relative_pos: bool = False,
        max_seq_len: int = 200
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_relative_pos = use_relative_pos

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        if use_relative_pos:
            self.rel_pos = RelativePositionalEncoding(d_model, max_seq_len)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # (batch, n_heads, seq_len, d_k)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)

        # Add relative positional bias if enabled
        if self.use_relative_pos:
            rel_pos_bias = self.rel_pos(seq_len)  # (seq_len, seq_len, d_model)
            # Reshape for multi-head: (seq_len, seq_len, n_heads, d_k)
            rel_pos_bias = rel_pos_bias.view(seq_len, seq_len, self.n_heads, self.d_k)
            # Compute bias: (batch, n_heads, seq_len, seq_len)
            rel_pos_scores = torch.einsum('bhqd,qkhd->bhqk', q, rel_pos_bias)
            scores = scores + rel_pos_scores

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(output)


class LinearAttention(nn.Module):
    """Linear attention with O(n) complexity using feature maps"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map for linear attention: elu+1"""
        return F.elu(x) + 1

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply feature map
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Linear attention: O(n) instead of O(n²)
        # KV = Σ(k^T * v), Z = Σ(q * k^T)
        kv = torch.matmul(k.transpose(-2, -1), v)  # (batch, n_heads, d_k, d_k)
        z = torch.matmul(q, k.sum(dim=-2).unsqueeze(-1))  # (batch, n_heads, seq_len, 1)

        output = torch.matmul(q, kv) / (z + 1e-6)  # (batch, n_heads, seq_len, d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(output)


class DepthwiseSeparableAttention(nn.Module):
    """Depthwise-separable attention for ultra-lightweight models"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Depthwise convolutions for Q, K, V
        self.q_depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.k_depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.v_depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        # Pointwise projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Apply depthwise conv (transpose for conv1d: batch, channels, seq)
        x_t = x.transpose(1, 2)
        q = self.q_proj(self.q_depthwise(x_t).transpose(1, 2))
        k = self.k_proj(self.k_depthwise(x_t).transpose(1, 2))
        v = self.v_proj(self.v_depthwise(x_t).transpose(1, 2))

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Efficient attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(output)


class FeedForwardNetwork(nn.Module):
    """Standard Feed-Forward Network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class InvertedBottleneckFFN(nn.Module):
    """Inverted Bottleneck FFN (MobileNetV2 style) for efficient computation"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        expansion_factor: int = 4
    ):
        super().__init__()
        hidden_dim = d_model * expansion_factor

        # Expansion -> Depthwise -> Projection
        self.expand = nn.Linear(d_model, hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim
        )
        self.project = nn.Linear(hidden_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = self.expand(x)  # (batch, seq_len, hidden_dim)
        x = self.activation(x)

        # Depthwise conv (transpose for conv1d)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        x = self.dropout(x)
        x = self.project(x)  # (batch, seq_len, d_model)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with configurable attention and FFN"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = 'standard',
        ffn_type: str = 'standard',
        use_relative_pos: bool = False,
        max_seq_len: int = 200
    ):
        super().__init__()

        # Attention mechanism
        if attention_type == 'standard':
            self.attn = MultiHeadSelfAttention(
                d_model, n_heads, dropout, use_relative_pos, max_seq_len
            )
        elif attention_type == 'linear':
            self.attn = LinearAttention(d_model, n_heads, dropout)
        elif attention_type == 'depthwise':
            self.attn = DepthwiseSeparableAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Feed-forward network
        if ffn_type == 'standard':
            self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        elif ffn_type == 'inverted_bottleneck':
            self.ffn = InvertedBottleneckFFN(d_model, d_ff, dropout)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x


class PatchEmbedding(nn.Module):
    """Patch embedding using 1D convolution for audio spectrograms"""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_size: int = 3,
        stride: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

        # Use conv to create patch embeddings
        self.proj = nn.Conv1d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) e.g., (B, 64, 157)
        Returns:
            (batch, time', d_model)
        """
        x = self.proj(x)  # (batch, d_model, time')
        x = x.transpose(1, 2)  # (batch, time', d_model)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class PoolingLayer(nn.Module):
    """Configurable pooling layer"""

    def __init__(self, d_model: int, pool_type: str = 'mean'):
        super().__init__()
        self.pool_type = pool_type

        if pool_type == 'attention':
            self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, d_model)
        """
        if self.pool_type == 'mean':
            return x.mean(dim=1)
        elif self.pool_type == 'max':
            return x.max(dim=1)[0]
        elif self.pool_type == 'attention':
            weights = F.softmax(self.attn_pool(x), dim=1)  # (batch, seq_len, 1)
            return (x * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")
