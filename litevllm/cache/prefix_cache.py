"""Prefix cache: detect and reuse shared prompt KV blocks.

Uses content-hash of token sequences to identify identical prefixes across
different requests, avoiding redundant prefill computation.
"""

from __future__ import annotations

from typing import Optional

from litevllm.cache.block_manager import BlockManager, BlockTable, PhysicalBlock


def _hash_block(token_ids: tuple[int, ...]) -> int:
    return hash(token_ids)


class PrefixCache:
    """Maps token prefix hashes to cached physical blocks."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.cache: dict[int, PhysicalBlock] = {}  # hash -> block

    def lookup(
        self,
        token_ids: list[int],
        block_manager: BlockManager,
    ) -> tuple[list[PhysicalBlock], int]:
        """Return (cached_blocks, num_cached_tokens) for the given prompt.

        Finds the longest matching prefix that is already in cache.
        """
        matched_blocks: list[PhysicalBlock] = []
        num_cached_tokens = 0

        num_full_blocks = len(token_ids) // self.block_size
        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            chunk = tuple(token_ids[start:end])
            h = _hash_block(chunk)

            if h in self.cache:
                block = self.cache[h]
                matched_blocks.append(block)
                num_cached_tokens = end
            else:
                break

        return matched_blocks, num_cached_tokens

    def insert(
        self,
        token_ids: list[int],
        block_table: BlockTable,
    ) -> None:
        """Register the blocks of a completed sequence into the prefix cache."""
        num_full_blocks = len(token_ids) // self.block_size
        for i in range(min(num_full_blocks, len(block_table))):
            start = i * self.block_size
            end = start + self.block_size
            chunk = tuple(token_ids[start:end])
            h = _hash_block(chunk)
            if h not in self.cache:
                block = block_table.blocks[i]
                block.is_cached = True
                block.hash_value = h
                self.cache[h] = block

    def evict(self, block: PhysicalBlock) -> None:
        if block.hash_value is not None and block.hash_value in self.cache:
            del self.cache[block.hash_value]
            block.is_cached = False
            block.hash_value = None
