#!/usr/bin/env python3
"""Experiment: chunked prefill ON vs OFF.

Focus metric: TPOT stability (std / p99), plus throughput and TTFT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from examples.system_metrics_bench import make_prompts, run_one_experiment


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked Prefill on/off experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--prompt-mode", choices=["short", "long", "mixed"], default="mixed")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out-dir", type=str, default="benchmark_results/chunked_prefill")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = make_prompts(args.num_requests, args.prompt_mode)

    off = run_one_experiment(
        model=args.model,
        dtype=args.dtype,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        chunked_prefill=False,
        chunk_size=args.chunk_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_sample_interval=0.2,
        out_dir=out_dir / "chunked_off",
    )
    on = run_one_experiment(
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
        out_dir=out_dir / "chunked_on",
    )

    compare = {
        "focus": "TPOT stability: lower std/p99 is better",
        "chunked_off": {
            "tpot_step_std_s": off["latency"]["tpot_step_std_s"],
            "tpot_step_p99_s": off["latency"]["tpot_step_p99_s"],
            "ttft_p99_s": off["latency"]["ttft_p99_s"],
            "request_throughput_req_per_s": off["throughput"]["request_throughput_req_per_s"],
            "output_token_throughput_tok_per_s": off["throughput"][
                "output_token_throughput_tok_per_s"
            ],
        },
        "chunked_on": {
            "tpot_step_std_s": on["latency"]["tpot_step_std_s"],
            "tpot_step_p99_s": on["latency"]["tpot_step_p99_s"],
            "ttft_p99_s": on["latency"]["ttft_p99_s"],
            "request_throughput_req_per_s": on["throughput"]["request_throughput_req_per_s"],
            "output_token_throughput_tok_per_s": on["throughput"][
                "output_token_throughput_tok_per_s"
            ],
        },
    }
    _write_json(out_dir / "chunked_prefill_compare.json", compare)
    print(f"[done] wrote: {out_dir / 'chunked_prefill_compare.json'}")


if __name__ == "__main__":
    main()

