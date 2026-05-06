"""Tests for the prefix cache (hash-based KV reuse)."""

from __future__ import annotations

from litevllm.cache.block_manager import BlockManager
from litevllm.cache.prefix_cache import PrefixCache


def test_prefix_cache_lookup_returns_blocks_for_repeated_prefix() -> None:
    block_size = 4
    bm = BlockManager(block_size=block_size, num_gpu_blocks=8)
    bm.allocate(seq_id=1, num_blocks=3)

    pc = PrefixCache(block_size=block_size)
    tokens = list(range(12))
    pc.insert(tokens, bm.get_block_table(1))

    matched, num_cached = pc.lookup(tokens, bm)
    assert num_cached == 12
    assert len(matched) == 3


def test_prefix_cache_partial_overlap() -> None:
    block_size = 4
    bm = BlockManager(block_size=block_size, num_gpu_blocks=8)
    bm.allocate(seq_id=1, num_blocks=3)

    pc = PrefixCache(block_size=block_size)
    tokens_a = list(range(12))
    pc.insert(tokens_a, bm.get_block_table(1))

    tokens_b = list(range(8)) + [99, 100, 101, 102]
    matched, num_cached = pc.lookup(tokens_b, bm)
    assert num_cached == 8
    assert len(matched) == 2


def test_prefix_cache_no_match_returns_empty() -> None:
    pc = PrefixCache(block_size=4)
    bm = BlockManager(block_size=4, num_gpu_blocks=4)
    matched, num_cached = pc.lookup([1, 2, 3, 4], bm)
    assert matched == []
    assert num_cached == 0
