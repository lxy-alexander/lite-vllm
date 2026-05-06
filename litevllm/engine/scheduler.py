"""Continuous Batching scheduler with Chunked Prefill support.

The scheduler decides which sequences to run in each step, balancing between
prefill and decode batches to maximize GPU utilization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..cache.block_manager import BlockManager
from ..config import CacheConfig, SchedulerConfig
from .sequence import Sequence, SequenceGroup, SequenceStatus


@dataclass
class SchedulerOutput:
    """What the scheduler decided for one step."""

    scheduled_seq_groups: list[SequenceGroup]
    num_prefill_groups: int
    num_decode_groups: int
    blocks_to_swap_in: dict[int, int] = field(default_factory=dict)
    blocks_to_swap_out: dict[int, int] = field(default_factory=dict)
    blocks_to_copy: dict[int, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_seq_groups) == 0

    @property
    def num_batched_tokens(self) -> int:
        total = 0
        for sg in self.scheduled_seq_groups:
            for seq in sg.get_unfinished_seqs():
                total += seq.num_new_tokens
        return total


class Scheduler:
    """FCFS scheduler with continuous batching and chunked prefill."""

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        block_manager: BlockManager,
    ) -> None:
        self.config = scheduler_config
        self.cache_config = cache_config
        self.block_manager = block_manager

        self.waiting: deque[SequenceGroup] = deque()
        self.running: list[SequenceGroup] = []
        self.swapped: list[SequenceGroup] = []

    def add_request(self, seq_group: SequenceGroup) -> None:
        self.waiting.append(seq_group)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def abort_request(self, request_id: str) -> None:
        for queue in [self.waiting, self.running, self.swapped]:
            for sg in list(queue):
                if sg.request_id == request_id:
                    for seq in sg.seqs:
                        seq.status = SequenceStatus.FINISHED_ABORT
                        self.block_manager.free(seq.seq_id)
                    queue.remove(sg)

    def schedule(self) -> SchedulerOutput:
        """Select sequences for the next forward step."""
        scheduled: list[SequenceGroup] = []
        num_prefill = 0
        num_decode = 0
        budget_tokens = self.config.max_num_batched_tokens
        budget_seqs = self.config.max_num_seqs
        used_tokens = 0
        used_seqs = 0

        # 1) Schedule running (decode) sequences first — they each need 1 token
        still_running: list[SequenceGroup] = []
        for sg in self.running:
            seqs = sg.get_unfinished_seqs()
            if not seqs:
                continue
            num_new = sum(s.num_new_tokens for s in seqs)
            if used_seqs + len(seqs) > budget_seqs or used_tokens + num_new > budget_tokens:
                still_running.append(sg)
                continue

            # Ensure blocks are available for new tokens
            can_run = True
            for seq in seqs:
                needed = seq.get_num_blocks(self.cache_config.block_size) - len(
                    self.block_manager.get_block_table(seq.seq_id)
                )
                if needed > 0 and not self.block_manager.can_allocate(needed):
                    can_run = False
                    break
            if not can_run:
                still_running.append(sg)
                continue

            for seq in seqs:
                needed = seq.get_num_blocks(self.cache_config.block_size) - len(
                    self.block_manager.get_block_table(seq.seq_id)
                )
                if needed > 0:
                    self.block_manager.append_slot(seq.seq_id)

            scheduled.append(sg)
            num_decode += 1
            used_tokens += num_new
            used_seqs += len(seqs)

        self.running = still_running

        # 2) Schedule waiting (prefill) sequences
        while self.waiting and used_seqs < budget_seqs and used_tokens < budget_tokens:
            sg = self.waiting[0]
            seq = sg.seqs[0]

            # Calculate how many tokens to prefill (chunked)
            remaining_prefill = seq.data.prompt_len - seq.num_computed_tokens
            if self.config.chunked_prefill:
                chunk = min(remaining_prefill, self.config.chunk_size, budget_tokens - used_tokens)
            else:
                chunk = remaining_prefill

            if chunk <= 0 or used_seqs + 1 > budget_seqs:
                break

            num_blocks_needed = seq.get_num_blocks(self.cache_config.block_size)
            existing = len(self.block_manager.block_tables.get(seq.seq_id, []))
            new_blocks_needed = num_blocks_needed - existing

            if new_blocks_needed > 0 and not self.block_manager.can_allocate(new_blocks_needed):
                break

            self.waiting.popleft()

            if seq.seq_id not in self.block_manager.block_tables:
                self.block_manager.allocate(seq.seq_id, max(new_blocks_needed, 1))
            else:
                for _ in range(new_blocks_needed):
                    self.block_manager.append_slot(seq.seq_id)

            seq.status = SequenceStatus.RUNNING
            scheduled.append(sg)
            num_prefill += 1
            used_tokens += chunk
            used_seqs += 1

        return SchedulerOutput(
            scheduled_seq_groups=scheduled,
            num_prefill_groups=num_prefill,
            num_decode_groups=num_decode,
        )

    def free_finished(self) -> list[SequenceGroup]:
        """Move finished groups out of running and free their blocks."""
        finished: list[SequenceGroup] = []
        still_running: list[SequenceGroup] = []
        for sg in self.running:
            if sg.is_finished:
                for seq in sg.seqs:
                    self.block_manager.free(seq.seq_id)
                finished.append(sg)
            else:
                still_running.append(sg)
        self.running = still_running
        return finished

    def update_running(self, scheduled: list[SequenceGroup]) -> None:
        """After a step, move scheduled groups into the running queue."""
        for sg in scheduled:
            if sg not in self.running:
                self.running.append(sg)
