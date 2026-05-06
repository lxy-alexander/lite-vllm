#!/usr/bin/env python3
"""Experiment: with/without PagedAttention (no-paged is an estimate baseline).

This script runs the real LiteVLLM engine (with PagedAttention), then computes a
"no-paged" proxy baseline by assuming each active request reserves a contiguous
KV region sized for (avg_prompt_tokens + max_tokens) for its whole lifetime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")

from examples.system_metrics_bench import make_prompts, run_one_experiment


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="PagedAttention experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--prompt-mode", choices=["short", "long", "mixed"], default="mixed")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out-dir", type=str, default="benchmark_results/pagedattention")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = make_prompts(args.num_requests, args.prompt_mode)

    real = run_one_experiment(
        model=args.model,
        dtype=args.dtype,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        chunked_prefill=True,
        chunk_size=args.chunk_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_sample_interval=0.2,
        out_dir=out_dir / "with_pagedattention",
    )

    # "No PagedAttention" proxy: contiguous/static reserve estimate.
    avg_prompt_tokens = real["throughput"]["total_prompt_tokens"] / max(
        real["throughput"]["num_requests"], 1
    )
    reserve_tokens_per_req = avg_prompt_tokens + args.max_tokens
    reserve_blocks_per_req = math.ceil(reserve_tokens_per_req / args.block_size)
    peak_active_seqs = real["batching"]["active_sequences_p95"]
    required_blocks_no_paged = reserve_blocks_per_req * peak_active_seqs
    total_blocks = real["kv"]["total_blocks"]
    util_no_paged = required_blocks_no_paged / total_blocks if total_blocks > 0 else 0.0
    estimated_admitted_active_seqs = (
        total_blocks / reserve_blocks_per_req if reserve_blocks_per_req > 0 else 0.0
    )

    compare = {
        "note": "no-paged baseline is an analytical estimate, not a real engine mode",
        "with_pagedattention": {
            "peak_block_utilization": real["kv"]["peak_block_utilization"],
            "avg_block_utilization": real["kv"]["avg_block_utilization"],
            "internal_fragmentation_peak_ratio": real["fragmentation"][
                "internal_fragmentation_peak_ratio"
            ],
            "external_fragmentation_rate": real["fragmentation"][
                "external_fragmentation_rate"
            ],
        },
        "without_pagedattention_estimated": {
            "reserve_tokens_per_request": reserve_tokens_per_req,
            "reserve_blocks_per_request": reserve_blocks_per_req,
            "required_blocks_at_p95_active_seqs": required_blocks_no_paged,
            "required_block_utilization": util_no_paged,
            "estimated_max_admitted_active_seqs": estimated_admitted_active_seqs,
        },
    }
    _write_json(out_dir / "pagedattention_compare.json", compare)
    print(f"[done] wrote: {out_dir / 'pagedattention_compare.json'}")


if __name__ == "__main__":
    main()

