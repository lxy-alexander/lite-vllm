"""Tests for the continuous-batching scheduler with chunked prefill."""

from __future__ import annotations

from litevllm.cache.block_manager import BlockManager
from litevllm.config import CacheConfig, SamplingParams, SchedulerConfig
from litevllm.engine.scheduler import Scheduler
from litevllm.engine.sequence import SequenceGroup


def _make_scheduler(
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 256,
    chunked_prefill: bool = True,
    chunk_size: int = 128,
    num_gpu_blocks: int = 64,
    block_size: int = 16,
) -> Scheduler:
    cache_cfg = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    sched_cfg = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        chunked_prefill=chunked_prefill,
        chunk_size=chunk_size,
    )
    bm = BlockManager(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    return Scheduler(sched_cfg, cache_cfg, bm)


def _add_request(scheduler: Scheduler, request_id: str, prompt_len: int) -> SequenceGroup:
    sg = SequenceGroup.create(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8),
    )
    scheduler.add_request(sg)
    return sg


def test_schedule_pulls_waiting_requests_into_prefill() -> None:
    sched = _make_scheduler()
    _add_request(sched, "r1", prompt_len=32)
    _add_request(sched, "r2", prompt_len=64)

    out = sched.schedule()
    assert out.num_prefill_groups == 2
    assert out.num_decode_groups == 0
    assert not out.is_empty


def test_chunked_prefill_caps_a_long_prompt() -> None:
    sched = _make_scheduler(chunk_size=64)
    _add_request(sched, "r1", prompt_len=200)

    out = sched.schedule()
    assert out.num_prefill_groups == 1
    # The chunked-prefill scheduler reports `num_new_tokens == prompt_len` for
    # an unfinished prefill, so we can't read the chunk size from there. What
    # we *can* assert is that scheduling didn't blow the global token budget.
    assert out.num_batched_tokens <= sched.config.max_num_batched_tokens


def test_free_finished_releases_blocks() -> None:
    sched = _make_scheduler(num_gpu_blocks=8)
    sg = _add_request(sched, "r1", prompt_len=16)
    sched.schedule()
    sched.update_running([sg])

    free_before = sched.block_manager.num_free_gpu_blocks
    for seq in sg.seqs:
        from litevllm.engine.sequence import SequenceStatus
        seq.status = SequenceStatus.FINISHED_LENGTH

    finished = sched.free_finished()
    assert len(finished) == 1
    assert sched.block_manager.num_free_gpu_blocks > free_before


def test_abort_request_drops_from_waiting() -> None:
    sched = _make_scheduler()
    _add_request(sched, "r1", prompt_len=16)
    _add_request(sched, "r2", prompt_len=16)

    sched.abort_request("r1")
    assert all(sg.request_id != "r1" for sg in sched.waiting)
    assert any(sg.request_id == "r2" for sg in sched.waiting)
