"""PagedAttention Block allocator / deallocator.

Tracks which physical blocks are free or occupied, allocates new blocks for
sequences, and supports copy-on-write for forked sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    is_cached: bool = False  # used by prefix cache
    hash_value: Optional[int] = None


class BlockAllocator:
    """Free-list based allocator for a single device (GPU or CPU).

    ``reserved_blocks`` makes the first ``reserved_blocks`` block ids
    permanently unallocatable. CUDA-graph decode uses block 0 as a scratch
    slot so padded lanes can write their KV without corrupting any real
    sequence's cache.
    """

    def __init__(
        self,
        num_blocks: int,
        device: str = "gpu",
        reserved_blocks: int = 0,
    ) -> None:
        self.device = device
        self.num_blocks = num_blocks
        self.reserved_blocks = reserved_blocks
        self.blocks: list[PhysicalBlock] = [
            PhysicalBlock(block_id=i) for i in range(num_blocks)
        ]
        self.free_list: list[int] = list(range(reserved_blocks, num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def allocate(self) -> PhysicalBlock:
        if not self.free_list:
            raise RuntimeError(f"No free blocks on {self.device}")
        block_id = self.free_list.pop()
        block = self.blocks[block_id]
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            block.is_cached = False
            block.hash_value = None
            self.free_list.append(block.block_id)

    def ref(self, block: PhysicalBlock) -> None:
        """Increment reference count (copy-on-write)."""
        block.ref_count += 1


@dataclass
class BlockTable:
    """Per-sequence mapping of logical block index -> physical block."""

    blocks: list[PhysicalBlock] = field(default_factory=list)

    @property
    def physical_block_ids(self) -> list[int]:
        return [b.block_id for b in self.blocks]

    def __len__(self) -> int:
        return len(self.blocks)


class BlockManager:
    """Manages block allocation for all active sequences."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int = 0,
        reserved_blocks: int = 0,
    ) -> None:
        self.block_size = block_size
        self.gpu_allocator = BlockAllocator(
            num_gpu_blocks, device="gpu", reserved_blocks=reserved_blocks,
        )
        self.cpu_allocator = (
            BlockAllocator(num_cpu_blocks, device="cpu")
            if num_cpu_blocks > 0
            else None
        )
        self.block_tables: dict[int, BlockTable] = {}  # seq_id -> BlockTable

    @property
    def num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.num_free_blocks

    def can_allocate(self, num_required_blocks: int) -> bool:
        return self.gpu_allocator.num_free_blocks >= num_required_blocks

    def allocate(self, seq_id: int, num_blocks: int) -> BlockTable:
        table = BlockTable()
        for _ in range(num_blocks):
            block = self.gpu_allocator.allocate()
            table.blocks.append(block)
        self.block_tables[seq_id] = table
        return table

    def append_slot(self, seq_id: int) -> Optional[PhysicalBlock]:
        """Allocate one more block for *seq_id*, or return None if the last
        block still has room (caller checks token position within block)."""
        block = self.gpu_allocator.allocate()
        self.block_tables[seq_id].blocks.append(block)
        return block

    def free(self, seq_id: int) -> None:
        table = self.block_tables.pop(seq_id, None)
        if table is None:
            return
        for block in table.blocks:
            self.gpu_allocator.free(block)

    def get_block_table(self, seq_id: int) -> BlockTable:
        return self.block_tables[seq_id]

    def fork(self, src_seq_id: int, dst_seq_id: int) -> BlockTable:
        """Copy-on-write: share physical blocks between two sequences."""
        src_table = self.block_tables[src_seq_id]
        dst_table = BlockTable()
        for block in src_table.blocks:
            self.gpu_allocator.ref(block)
            dst_table.blocks.append(block)
        self.block_tables[dst_seq_id] = dst_table
        return dst_table
