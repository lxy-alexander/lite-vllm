"""Linear layer wrappers.

The single-GPU build keeps the ``ColumnParallelLinear`` / ``RowParallelLinear``
class names so model code in ``llama.py`` / ``qwen3.py`` stays unchanged, but
they now reduce to plain ``nn.Linear`` (no weight sharding, no quantization).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear (degenerate to nn.Linear on a single GPU)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        **_unused,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Row-parallel linear (degenerate to nn.Linear on a single GPU)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        **_unused,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
