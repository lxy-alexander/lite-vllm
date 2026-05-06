#!/usr/bin/env python3
"""Prefix Cache ON vs OFF module-level experiment.

Goal:
  Compare estimated prefill cost with and without Prefix Cache.

OFF:
  - no prefix cache
  - every request prefill = shared prefix + unique suffix

ON first pass:
  - cache starts cold
  - first request in each shared-prefix group inserts prefix blocks
  - later requests reuse cached prefix
  - includes lookup + insert overhead

ON steady state:
  - cache is already warm
  - all shared prefixes are already cached
  - includes lookup overhead only

This is a module-level estimate, not end-to-end engine speedup.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from litevllm.cache.block_manager import BlockManager
from litevllm.cache.prefix_cache import PrefixCache


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_speedup(baseline_ms: float, actual_ms: float) -> float:
    if actual_ms <= 0:
        return 0.0
    return baseline_ms / actual_ms


def _build_shared_prefix_prompts(
    *,
    num_requests: int,
    vocab_size: int,
    prefix_tokens: int,
    suffix_tokens: int,
    shared_prefix_groups: int,
    seed: int,
) -> list[list[int]]:
    random.seed(seed)

    groups = max(1, shared_prefix_groups)

    group_prefixes: list[list[int]] = []
    for _ in range(groups):
        prefix = [random.randint(10, vocab_size - 1) for _ in range(prefix_tokens)]
        group_prefixes.append(prefix)

    prompts: list[list[int]] = []
    for i in range(num_requests):
        prefix = group_prefixes[i % groups]
        suffix = [random.randint(10, vocab_size - 1) for _ in range(suffix_tokens)]
        prompts.append(prefix + suffix)

    return prompts


def _allocate_block_table_for_prompt(
    bm: BlockManager,
    seq_id: int,
    token_ids: list[int],
    block_size: int,
) -> None:
    num_blocks = max(1, (len(token_ids) + block_size - 1) // block_size)
    bm.allocate(seq_id, num_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefix Cache ON/OFF module-level experiment")

    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=50000)

    # Important defaults:
    # Long shared prefix + short unique suffix.
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--shared-prefix-groups", type=int, default=4)

    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=200000)
    parser.add_argument("--prefill-us-per-token", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="benchmark_results/prefix_cache")

    args = parser.parse_args()

    if args.num_requests <= 0:
        raise ValueError("--num-requests must be > 0")
    if args.prefix_tokens < 0:
        raise ValueError("--prefix-tokens must be >= 0")
    if args.suffix_tokens < 0:
        raise ValueError("--suffix-tokens must be >= 0")
    if args.shared_prefix_groups <= 0:
        raise ValueError("--shared-prefix-groups must be > 0")
    if args.block_size <= 0:
        raise ValueError("--block-size must be > 0")
    if args.prefill_us_per_token < 0:
        raise ValueError("--prefill-us-per-token must be >= 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _build_shared_prefix_prompts(
        num_requests=args.num_requests,
        vocab_size=args.vocab_size,
        prefix_tokens=args.prefix_tokens,
        suffix_tokens=args.suffix_tokens,
        shared_prefix_groups=args.shared_prefix_groups,
        seed=args.seed,
    )

    total_prompt_tokens = sum(len(p) for p in prompts)

    # ----------------------------------------------------------------------
    # Prefix Cache OFF baseline
    # ----------------------------------------------------------------------
    # Without Prefix Cache, every prompt token is prefilled normally.
    prefix_cache_off_ms = total_prompt_tokens * args.prefill_us_per_token / 1000.0

    # ----------------------------------------------------------------------
    # Prefix Cache ON: first pass
    # ----------------------------------------------------------------------
    bm = BlockManager(
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=0,
    )
    pc = PrefixCache(args.block_size)

    first_lookup_total_s = 0.0
    first_insert_total_s = 0.0
    first_hit_requests = 0
    first_cached_tokens = 0
    first_hit_blocks = 0
    trace_rows: list[dict] = []

    for i, token_ids in enumerate(prompts):
        seq_id = i + 1

        t0 = time.perf_counter()
        matched_blocks, num_cached_tokens = pc.lookup(token_ids, bm)
        lookup_s = time.perf_counter() - t0
        first_lookup_total_s += lookup_s

        if num_cached_tokens > 0:
            first_hit_requests += 1
            first_cached_tokens += num_cached_tokens
            first_hit_blocks += len(matched_blocks)

        _allocate_block_table_for_prompt(
            bm=bm,
            seq_id=seq_id,
            token_ids=token_ids,
            block_size=args.block_size,
        )

        t1 = time.perf_counter()
        pc.insert(token_ids, bm.get_block_table(seq_id))
        insert_s = time.perf_counter() - t1
        first_insert_total_s += insert_s

        trace_rows.append(
            {
                "request_idx": i,
                "prompt_tokens": len(token_ids),
                "cached_tokens": num_cached_tokens,
                "hit_blocks": len(matched_blocks),
                "lookup_ms": lookup_s * 1000.0,
                "insert_ms": insert_s * 1000.0,
            }
        )

    first_lookup_overhead_ms = first_lookup_total_s * 1000.0
    first_insert_overhead_ms = first_insert_total_s * 1000.0
    first_cache_overhead_ms = first_lookup_overhead_ms + first_insert_overhead_ms
    first_saved_prefill_ms = first_cached_tokens * args.prefill_us_per_token / 1000.0

    # ON first-pass cost:
    # original full prefill cost - saved cached-token prefill + cache operation overhead.
    prefix_cache_on_first_pass_ms = (
        prefix_cache_off_ms - first_saved_prefill_ms + first_cache_overhead_ms
    )

    # ----------------------------------------------------------------------
    # Prefix Cache ON: steady state
    # ----------------------------------------------------------------------
    # Cache has already been warmed by first pass.
    steady_lookup_total_s = 0.0
    steady_hit_requests = 0
    steady_cached_tokens = 0
    steady_hit_blocks = 0

    for token_ids in prompts:
        t0 = time.perf_counter()
        matched_blocks, num_cached_tokens = pc.lookup(token