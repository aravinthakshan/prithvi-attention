"""
Channel-wise attention modules adapted from TabTune for Prithvi EO 2.0.

Four modes:
  - "none"  : no channel attention, pure Prithvi baseline
  - "limix" : LIMIX-style feature attention (einsum QKV, cross-channel per patch)
  - "mitra" : Mitra/Tab2D-style alternating row+feature attention (separate Q,K,V,O linears)
  - "orion" : Orion-BIX Bi-Axial attention (4 stacked patterns: standard, grouped, hierarchical, relational)

Architecture:
  Input: (B, N, D) — Prithvi patch tokens after patch_embed
  We project D -> C * slot_dim (where C = num_channels = 6, slot_dim is chosen to be divisible by heads)
  Attend across the C channel slots
  Project back C * slot_dim -> D
  Residual connection

This avoids the D % C == 0 constraint entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _slot_dim(embed_dim: int, num_channels: int, num_heads: int) -> int:
    """Pick a slot_dim close to embed_dim // num_channels, divisible by num_heads."""
    raw = embed_dim // num_channels
    # round down to nearest multiple of num_heads
    return max(num_heads, (raw // num_heads) * num_heads)


# ---------------------------------------------------------------------------
# Baseline: identity
# ---------------------------------------------------------------------------

class NoChannelAttention(nn.Module):
    def __init__(self, embed_dim: int, num_channels: int = 6, num_heads: int = 2):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# LIMIX-style channel attention
# ---------------------------------------------------------------------------

class LIMIXChannelAttention(nn.Module):
    """
    LIMIX-style feature attention: einsum-packed QKV over C channel slots.

    Project D -> C * slot_dim, attend across C, project back.
    """

    def __init__(self, embed_dim: int, num_channels: int = 6, num_heads: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.C = num_channels
        self.slot_dim = _slot_dim(embed_dim, num_channels, num_heads)
        self.num_heads = num_heads
        self.head_dim = self.slot_dim // num_heads
        self.inner_dim = self.C * self.slot_dim

        # Project in/out between D and C*slot_dim
        self.proj_in = nn.Linear(embed_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, embed_dim)

        # LIMIX-style packed QKV: (3, H, head_dim, slot_dim)
        self.qkv_proj = nn.Parameter(torch.empty(3, num_heads, self.head_dim, self.slot_dim))
        self.out_attn = nn.Parameter(torch.empty(num_heads, self.head_dim, self.slot_dim))
        nn.init.xavier_uniform_(self.qkv_proj)
        nn.init.xavier_uniform_(self.out_attn)

        self.norm = nn.LayerNorm(embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        # Project to channel-slot space: (B, N, D) -> (B*N, C, slot_dim)
        x_in = self.proj_in(x).view(B * N, self.C, self.slot_dim)

        # LIMIX einsum QKV
        qkv = torch.einsum("b c s, j h d s -> b c j h d", x_in, self.qkv_proj)
        q, k, v = qkv.unbind(dim=2)  # each: (B*N, C, H, head_dim)

        q = q.permute(0, 2, 1, 3)  # (B*N, H, C, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B*N, H, C, head_dim)

        out = out.permute(0, 2, 1, 3)  # (B*N, C, H, head_dim)
        out = torch.einsum("b c h d, h d s -> b c s", out, self.out_attn)

        # Project back: (B*N, C, slot_dim) -> (B, N, D)
        out = out.reshape(B, N, self.inner_dim)
        out = self.proj_out(out)
        return residual + out


# ---------------------------------------------------------------------------
# Mitra/Tab2D-style channel attention
# ---------------------------------------------------------------------------

class MitraChannelAttention(nn.Module):
    """
    Mitra-style feature attention: separate Q,K,V,O linears + FFN over C channel slots.

    Project D -> C * slot_dim, attend across C, FFN, project back.
    """

    def __init__(self, embed_dim: int, num_channels: int = 6, num_heads: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.C = num_channels
        self.slot_dim = _slot_dim(embed_dim, num_channels, num_heads)
        self.num_heads = num_heads
        self.head_dim = self.slot_dim // num_heads
        self.inner_dim = self.C * self.slot_dim

        # Project in/out
        self.proj_in = nn.Linear(embed_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, embed_dim)

        # Mitra-style separate Q, K, V, O
        self.q = nn.Linear(self.slot_dim, self.slot_dim, bias=True)
        self.k = nn.Linear(self.slot_dim, self.slot_dim, bias=True)
        self.v = nn.Linear(self.slot_dim, self.slot_dim, bias=True)
        self.o = nn.Linear(self.slot_dim, self.slot_dim, bias=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(self.slot_dim)
        self.norm3 = nn.LayerNorm(self.slot_dim)

        # FFN (Mitra includes this after attention)
        self.ff1 = nn.Linear(self.slot_dim, self.slot_dim * 4)
        self.ff2 = nn.Linear(self.slot_dim * 4, self.slot_dim)

        self.scale = self.head_dim ** -0.5

    def _mha(self, x: torch.Tensor) -> torch.Tensor:
        M, L, S = x.shape
        q = self.q(x).view(M, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(M, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(M, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(M, L, S)
        return self.o(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        residual = x
        x = self.norm1(x)

        # Project to channel-slot space
        x_in = self.proj_in(x).view(B * N, self.C, self.slot_dim)

        # Attention across channels
        x_normed = self.norm2(x_in)
        att_out = self._mha(x_normed)
        x_in = x_in + att_out

        # FFN
        ff_in = x_in
        x_in = self.ff2(F.gelu(self.ff1(self.norm3(x_in))))
        x_in = ff_in + x_in

        # Project back
        out = x_in.reshape(B, N, self.inner_dim)
        out = self.proj_out(out)
        return residual + out


# ---------------------------------------------------------------------------
# Orion Bi-Axial channel attention
# Adapted from TabTune/tabtune/models/orion_bix/model/layers.py
#
# 4 stacked feature-axis attention patterns from Orion-BIX:
#   1. Standard cross-feature self-attention
#   2. Grouped feature attention (split channels into groups, attend within)
#   3. Hierarchical feature attention (first half <-> second half cross-attend)
#   4. Relational feature attention (project then self-attend)
# ---------------------------------------------------------------------------

class _MHABlock(nn.Module):
    """Minimal MHA + FFN block (Orion-BIX style, pre-norm)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
        if v is None:
            v = k
        # Pre-norm attention
        q_n, k_n, v_n = self.norm1(q), self.norm1(k), self.norm1(v)
        out, _ = self.attn(q_n, k_n, v_n)
        x = q + out
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class OrionBiAxialChannelAttention(nn.Module):
    """
    Orion-BIX Bi-Axial attention applied channel-wise to Prithvi patch tokens.

    Stacks 4 complementary feature attention patterns on the C=6 channel slots:
      1. Standard: all channels attend to all channels
      2. Grouped: split into 2 groups of 3, attend within groups
      3. Hierarchical: first 3 channels <-> last 3 channels cross-attend
      4. Relational: project then full self-attention

    Project D -> C*slot_dim, apply 4 attention blocks across C, project back.
    """

    def __init__(self, embed_dim: int, num_channels: int = 6, num_heads: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.C = num_channels
        self.slot_dim = _slot_dim(embed_dim, num_channels, num_heads)
        self.num_heads = num_heads
        self.inner_dim = self.C * self.slot_dim

        # Project in/out
        self.proj_in = nn.Linear(embed_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # 1. Standard cross-channel self-attention
        self.standard_attn = _MHABlock(self.slot_dim, num_heads)

        # 2. Grouped channel attention (2 groups of C//2)
        self.num_groups = min(2, num_channels)
        self.group_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.group_norm = nn.LayerNorm(self.slot_dim)
        self.grouped_attn = _MHABlock(self.slot_dim, num_heads)

        # 3. Hierarchical: first-half <-> second-half cross-attention
        self.hier_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.hier_norm = nn.LayerNorm(self.slot_dim)
        self.hier_attn = _MHABlock(self.slot_dim, num_heads)

        # 4. Relational: project then self-attend
        self.rel_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.rel_norm = nn.LayerNorm(self.slot_dim)
        self.rel_attn = _MHABlock(self.slot_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        # Project to channel-slot space: (B*N, C, slot_dim)
        x_ch = self.proj_in(x).view(B * N, self.C, self.slot_dim)

        # 1. Standard cross-channel attention
        x_ch = self.standard_attn(x_ch)

        # 2. Grouped: split channels into groups, attend within each
        cpg = self.C // self.num_groups  # channels per group
        x_g = x_ch[:, :cpg * self.num_groups, :].view(
            B * N * self.num_groups, cpg, self.slot_dim
        )
        x_g = self.grouped_attn(self.group_norm(self.group_proj(x_g)))
        x_ch[:, :cpg * self.num_groups, :] = x_g.view(B * N, cpg * self.num_groups, self.slot_dim)

        # 3. Hierarchical: first half <-> second half cross-attend
        mid = self.C // 2
        first = self.hier_norm(self.hier_proj(x_ch[:, :mid, :]))
        second = self.hier_norm(self.hier_proj(x_ch[:, mid:, :]))
        first = self.hier_attn(first, second, second)
        second = self.hier_attn(second, first, first)
        x_ch = torch.cat([first, second], dim=1)

        # 4. Relational: project then self-attend
        x_ch = self.rel_attn(self.rel_norm(self.rel_proj(x_ch)))

        # Project back: (B*N, C, slot_dim) -> (B, N, D)
        out = x_ch.reshape(B, N, self.inner_dim)
        out = self.proj_out(out)
        return residual + out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ATTENTION_REGISTRY = {
    "none":   NoChannelAttention,
    "limix":  LIMIXChannelAttention,
    "mitra":  MitraChannelAttention,
    "orion":  OrionBiAxialChannelAttention,
}


def build_channel_attention(
    attn_type: str,
    embed_dim: int,
    num_channels: int = 6,
    num_heads: int = 2,
) -> nn.Module:
    if attn_type not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type '{attn_type}'. Choose from: {list(ATTENTION_REGISTRY)}")
    return ATTENTION_REGISTRY[attn_type](embed_dim, num_channels, num_heads)
