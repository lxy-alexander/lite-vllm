"""Rotary Positional Embedding (RoPE) with YaRN scaling support."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Standard RoPE with optional YaRN / linear / dynamic NTK scaling."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling

        self._build_cache(device or "cpu")

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        if self.rope_scaling is not None:
            scaling_type = self.rope_scaling.get("type", "linear")
            factor = self.rope_scaling.get("factor", 1.0)

            if scaling_type == "linear":
                inv_freq = inv_freq / factor
            elif scaling_type == "dynamic":
                base = self.base * (
                    (factor * self.max_position_embeddings / self.max_position_embeddings)
                    - (factor - 1)
                ) ** (self.head_dim / (self.head_dim - 2))
                inv_freq = 1.0 / (
                    base
                    ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
                )
            elif scaling_type == "yarn":
                inv_freq = self._yarn_scaling(inv_freq, factor)

        return inv_freq

    def _yarn_scaling(
        self, inv_freq: torch.Tensor, factor: float
    ) -> torch.Tensor:
        """YaRN: Yet another RoPE extensioN."""
        beta_fast = 32.0
        beta_slow = 1.0
        dim = self.head_dim

        low = math.floor(dim * math.log(1.0 / (beta_fast * 2 * math.pi)) / (2 * math.log(self.base)))
        high = math.ceil(dim * math.log(1.0 / (beta_slow * 2 * math.pi)) / (2 * math.log(self.base)))
        low = max(low, 0)
        high = min(high, dim // 2 - 1)

        freqs = torch.arange(0, dim // 2, dtype=torch.float32)
        ramp = (freqs - low) / max(high - low, 1)
        ramp = ramp.clamp(0, 1)

        scaled = inv_freq / factor
        inv_freq = (1 - ramp) * scaled + ramp * inv_freq

        return inv_freq

    def _build_cache(self, device: str) -> None:
        inv_freq = self._compute_inv_freq()
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim // 2)
        cos = freqs.cos().to(device)
        sin = freqs.sin().to(device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for the given position ids.

        Returns:
            cos: (num_tokens, head_dim // 2)
            sin: (num_tokens, head_dim // 2)
        """
        return (
            self.cos_cached[positions],
            self.sin_cached[positions],
        )


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to the input tensor.

    Args:
        x: (..., head_dim) – last dim is split into two halves.
        cos: (..., head_dim // 2)
        sin: (..., head_dim // 2)
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
