"""KV Tensor lifecycle management.

Manages the raw GPU/CPU tensors that hold key-value cache data.  Each layer
gets a pair of tensors ``(key_cache, value_cache)`` of shape
``(num_blocks, block_size, num_kv_heads, head_dim)``.
"""

from __future__ import annotations

from typing import Optional

import torch


class KVCache:
    """Holds the physical KV tensors for all layers on one device."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.key_caches: list[torch.Tensor] = []
        self.value_caches: list[torch.Tensor] = []

        self._allocate()

    def _allocate(self) -> None:
        shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        for _ in range(self.num_layers):
            k = torch.zeros(shape, dtype=self.dtype, device=self.device)
            v = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.key_caches.append(k)
            self.value_caches.append(v)

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    @property
    def total_bytes(self) -> int:
        elem = self.key_caches[0].element_size()
        per_layer = 2 * self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim * elem
        return per_layer * self.num_layers

    @classmethod
    def profile_num_blocks(
        cls,
        available_memory_bytes: int,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """Estimate max number of blocks that fit in *available_memory_bytes*."""
        elem = torch.tensor([], dtype=dtype).element_size()
        per_block_per_layer = 2 * block_size * num_kv_heads * head_dim * elem
        per_block_total = per_block_per_layer * num_layers
        return max(1, int(available_memory_bytes // per_block_total))

    def swap_in(
        self,
        src_cpu_cache: "KVCache",
        block_mapping: dict[int, int],
    ) -> None:
        """Copy blocks from CPU cache to this GPU cache."""
        for src_block, dst_block in block_mapping.items():
            for layer_idx in range(self.num_layers):
                self.key_caches[layer_idx][dst_block].copy_(
                    src_cpu_cache.key_caches[layer_idx][src_block]
                )
                self.value_caches[layer_idx][dst_block].copy_(
                    src_cpu_cache.value_caches[layer_idx][src_block]
                )

    def swap_out(
        self,
        dst_cpu_cache: "KVCache",
        block_mapping: dict[int, int],
    ) -> None:
        """Copy blocks from this GPU cache to CPU cache."""
        for src_block, dst_block in block_mapping.items():
            for layer_idx in range(self.num_layers):
                dst_cpu_cache.key_caches[layer_idx][dst_block].copy_(
                    self.key_caches[layer_idx][src_block]
                )
                dst_cpu_cache.value_caches[layer_idx][dst_block].copy_(
                    self.value_caches[layer_idx][src_block]
                )
