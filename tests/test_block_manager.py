"""Unit tests for the BlockManager (PagedAttention block allocator)."""

from __future__ import annotations

import pytest

from litevllm.cache.block_manager import BlockManager


def test_allocate_and_free_releases_blocks() -> None:
    bm = BlockManager(block_size=16, num_gpu_blocks=8)
    assert bm.num_free_gpu_blocks == 8

    bm.allocate(seq_id=1, num_blocks=3)
    assert bm.num_free_gpu_blocks == 5
    assert len(bm.get_block_table(1)) == 3

    bm.free(seq_id=1)
    assert bm.num_free_gpu_blocks == 8
    assert 1 not in bm.block_tables


def test_can_allocate_returns_false_when_full() -> None:
    bm = BlockManager(block_size=16, num_gpu_blocks=4)
    bm.allocate(seq_id=1, num_blocks=4)
    assert bm.num_free_gpu_blocks == 0
    assert not bm.can_allocate(1)


def test_append_slot_extends_block_table() -> None:
    bm = BlockManager(block_size=16, num_gpu_blocks=4)
    bm.allocate(seq_id=1, num_blocks=1)
    bm.append_slot(1)
    assert len(bm.get_block_table(1)) == 2
    assert bm.num_free_gpu_blocks == 2


def test_fork_shares_blocks_via_refcount() -> None:
    bm = BlockManager(block_size=16, num_gpu_blocks=4)
    bm.allocate(seq_id=1, num_blocks=2)
    bm.fork(src_seq_id=1, dst_seq_id=2)

    assert bm.num_free_gpu_blocks == 2
    for blk in bm.get_block_table(1).blocks:
        assert blk.ref_count == 2

    bm.free(seq_id=1)
    assert bm.num_free_gpu_blocks == 2
    bm.free(seq_id=2)
    assert bm.num_free_gpu_blocks == 4


def test_allocate_raises_when_no_blocks_left() -> None:
    bm = BlockManager(block_size=16, num_gpu_blocks=2)
    bm.allocate(seq_id=1, num_blocks=2)
    with pytest.raises(RuntimeError):
        bm.allocate(seq_id=2, num_blocks=1)
