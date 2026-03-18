"""
Channel-wise attention modules adapted from TabTune for Prithvi EO 2.0.

Three modes:
  - "none"  : no channel attention, pure Prithvi baseline
  - "limix" : LIMIX-style feature attention (einsum QKV, cross-channel per patch)
  - "mitra" : Mitra/Tab2D-style alternating row+feature attention (separate Q,K,V,O linears)

All modules operate on the spectral-channel axis of Prithvi patch tokens.
Prithvi patch token shape after embedding: (B, N_patches, D)
  where each patch was formed from (1, 16, 16) spatiotemporal cube of (C, T, H, W) input.

The channel attention is inserted *after* patch embedding, treating the C=6 spectral
bands as a secondary "feature" axis to attend over within each patch token.

Design:
  - We lift each patch token into (B, N_patches, C, D//C) — C slots of D//C dims each
  - Apply attention across the C slots
  - Flatten back to (B, N_patches, D)

This keeps the module self-contained and requires no changes to PrithviViT internals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline: identity, no channel attention
# ---------------------------------------------------------------------------

class NoChannelAttention(nn.Module):
    """Passthrough — no channel attention, pure Prithvi baseline."""

    def __init__(self, embed_dim: int, num_channels: int = 6, num_heads: int = 2):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# LIMIX-style channel attention
# Adapted from TabTune/tabtune/models/limix/layer.py MultiheadAttention
#
# Key idea: einsum-packed QKV projection over the channel axis.
# Input shape: (B, N, D) — we reshape to (B*N, C, D//C), attend across C, flatten.
# ---------------------------------------------------------------------------

class LIMIXChannelAttention(nn.Module):
    """
    LIMIX-style feature attention applied channel-wise to Prithvi patch tokens.

    Adapted from LIMIX's MultiheadAttention which uses an einsum QKV projection
    of shape (3, num_heads, head_dim, embed_dim) to attend across the feature axis.

    Here 'features' = spectral channels C.
    """

    def __init__(self, embed_dim: int, num_channels: int = 6, num_heads: int = 2):
        super().__init__()
        assert embed_dim % num_channels == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_channels ({num_channels})"
        )
        self.embed_dim = embed_dim
        self.C = num_channels
        self.slot_dim = embed_dim // num_channels  # D per channel slot
        self.num_heads = num_heads
        assert self.slot_dim % num_heads == 0
        self.head_dim = self.slot_dim // num_heads

        # LIMIX-style: packed (3, H, head_dim, slot_dim) weight — Q, K, V jointly
        self.qkv_proj = nn.Parameter(
            torch.empty(3, num_heads, self.head_dim, self.slot_dim)
        )
        self.out_proj = nn.Parameter(
            torch.empty(num_heads, self.head_dim, self.slot_dim)
        )
        nn.init.xavier_uniform_(self.qkv_proj)
        nn.init.xavier_uniform_(self.out_proj)

        self.norm = nn.LayerNorm(embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)  — Prithvi patch tokens post-embedding
        returns: (B, N, D)
        """
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        # Reshape: (B, N, D) -> (B*N, C, slot_dim)
        x_ch = x.view(B * N, self.C, self.slot_dim)

        # LIMIX einsum QKV projection: "... s, j h d s -> ... j h d"
        # x_ch: (B*N, C, slot_dim)  qkv_proj: (3, H, head_dim, slot_dim)
        qkv = torch.einsum("b c s, j h d s -> b c j h d", x_ch, self.qkv_proj)
        # qkv: (B*N, C, 3, H, head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B*N, C, H, head_dim)

        # Scaled dot-product attention across C channels
        q = q.permute(0, 2, 1, 3)  # (B*N, H, C, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B*N, H, C, C)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                    # (B*N, H, C, head_dim)

        # Output projection via LIMIX einsum: "n h c d, h d s -> n c s"
        out = out.permute(0, 2, 1, 3)                    # (B*N, C, H, head_dim)
        out = torch.einsum("b c h d, h d s -> b c s", out, self.out_proj)
        # out: (B*N, C, slot_dim)

        # Flatten back to (B, N, D)
        out = out.reshape(B, N, D)
        return residual + out


# ---------------------------------------------------------------------------
# Mitra/Tab2D-style channel attention
# Adapted from TabTune/tabtune/models/mitra/tab2d.py Layer + MultiheadAttention
#
# Key idea: separate Q, K, V, O linear projections; reshape trick to batch
# the attention across channels.
# Input: (B, N, D) -> reshape to (B*N, C, slot_dim) -> attend across C -> reshape back
# ---------------------------------------------------------------------------

class MitraChannelAttention(nn.Module):
    """
    Mitra/Tab2D-style feature attention applied channel-wise to Prithvi patch tokens.

    Adapted from Mitra's MultiheadAttention which uses separate Q, K, V, O linears
    and the reshape trick: (b, s, f, d) -> (b*s, f, d) for feature attention.

    Here: (B, N, C, slot_dim) — N patches as "observations", C channels as "features".
    """

    def __init__(self, embed_dim: int, num_channels: int = 6, num_heads: int = 2):
        super().__init__()
        assert embed_dim % num_channels == 0
        self.embed_dim = embed_dim
        self.C = num_channels
        self.slot_dim = embed_dim // num_channels
        self.num_heads = num_heads
        assert self.slot_dim % num_heads == 0
        self.head_dim = self.slot_dim // num_heads

        # Mitra-style: separate Q, K, V, O projections
        self.q = nn.Linear(self.slot_dim, self.slot_dim, bias=True)
        self.k = nn.Linear(self.slot_dim, self.slot_dim, bias=True)
        self.v = nn.Linear(self.slot_dim, self.slot_dim, bias=True)
        self.o = nn.Linear(self.slot_dim, self.slot_dim, bias=True)

        # Two-block structure from Mitra: norm before each sub-block
        self.norm1 = nn.LayerNorm(embed_dim)     # before channel attention
        self.norm2 = nn.LayerNorm(self.slot_dim) # inside the attention block
        self.ff1 = nn.Linear(self.slot_dim, self.slot_dim * 4)
        self.ff2 = nn.Linear(self.slot_dim * 4, self.slot_dim)

        self.scale = self.head_dim ** -0.5

    def _mha(self, x: torch.Tensor) -> torch.Tensor:
        """Standard MHA on input (M, L, slot_dim) -> (M, L, slot_dim)."""
        M, L, S = x.shape
        q = self.q(x).view(M, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(M, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(M, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(M, L, S)
        return self.o(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        returns: (B, N, D)
        """
        B, N, D = x.shape
        residual = x
        x = self.norm1(x)

        # Mitra reshape trick: (B, N, D) -> (B, N, C, slot_dim) -> (B*N, C, slot_dim)
        x_ch = x.view(B, N, self.C, self.slot_dim)
        x_flat = x_ch.reshape(B * N, self.C, self.slot_dim)  # (B*N, C, slot_dim)

        # Channel attention (attention across C, independent per patch)
        x_flat = self.norm2(x_flat)
        att_out = self._mha(x_flat)
        x_flat = x_flat + att_out

        # FFN
        ff_in = x_flat
        x_flat = self.ff2(F.gelu(self.ff1(self.norm2(x_flat))))
        x_flat = ff_in + x_flat

        # Reshape back: (B*N, C, slot_dim) -> (B, N, D)
        out = x_flat.reshape(B, N, D)
        return residual + out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ATTENTION_REGISTRY = {
    "none":  NoChannelAttention,
    "limix": LIMIXChannelAttention,
    "mitra": MitraChannelAttention,
}


def build_channel_attention(
    attn_type: str,
    embed_dim: int,
    num_channels: int = 6,
    num_heads: int = 2,
) -> nn.Module:
    """
    Factory function.  attn_type must be one of: 'none', 'limix', 'mitra'.
    """
    if attn_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type '{attn_type}'. Choose from: {list(ATTENTION_REGISTRY)}"
        )
    return ATTENTION_REGISTRY[attn_type](embed_dim, num_channels, num_heads)
