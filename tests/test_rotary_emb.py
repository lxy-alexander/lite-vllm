"""Tests for rotary positional embedding."""

from __future__ import annotations

import torch

from litevllm.layers.rotary_emb import RotaryEmbedding, apply_rotary_emb


def test_rope_position_zero_is_identity() -> None:
    head_dim = 32
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=128)
    positions = torch.tensor([0])
    cos, sin = rope(positions)

    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin))


def test_rope_norm_preserved() -> None:
    head_dim = 32
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=128)
    positions = torch.arange(4)

    x = torch.randn(4, 1, head_dim)
    cos, sin = rope(positions)
    rotated = apply_rotary_emb(x, cos.unsqueeze(1), sin.unsqueeze(1))

    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)
