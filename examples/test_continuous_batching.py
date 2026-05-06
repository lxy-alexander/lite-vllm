#!/usr/bin/env python3
"""Experiment: with vs without continuous batching.

With continuous batching:
  - submit all requests together
  - allow many active sequences

Without continuous batching baseline:
  - submit the same requests together
  - but set max_num_seqs=1, so the engine can only process one active sequence at a time

This avoids reloading the model once per request.

Use --mode to run only one side (useful when one config OOMs / hangs on small
GPUs like T4):
  --mode on    : only run the continuous-batching ON experiment
  --mode off   : only run the max_num_seqs=1 baseline
  --mode both  : run both and write the compare file (default)

T4 (16 GB) friendly suggestion:
  --num-requests 16 --max-tokens 64 --prompt-mode short \\
  --max-num-seqs 32 --max-num-batched-tokens 2048 \\
  --gpu-memory-utilization 0.80
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


def _pick_metrics(result: dict) -> dict:
    return {
        "request_throughput_req_per_s": result["throughput"]["request_throughput_req_per_s"],
        "output_token_throughput_tok_per_s": result["throughput"][
            "output_token_throughput_tok_per_s"
        ],
        "total_token_throughput_tok_per_s": result["throughput"][
            "total_token_throughput_tok_per_s"
        ],
        "ttft_p99_s": result["latency"]["ttft_p99_s"],
        "tpot_step_p99_s": result["latency"]["tpot_step_p99_s"],
        "active_sequences_mean": result["batching"]["active_sequences_mean"],
        "idle_slot_ratio_mean": result["batching"]["idle_slot_ratio_mean"],
        "queue_wait_p99_s": result["batching"]["queue_wait_p99_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous batching experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-mode", choices=["short", "long", "mixed"], default="mixed")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out-dir", type=str, default="benchmark_results/continuous_batching")
    parser.add_argument(
        "--mode",
        choices=["on", "off", "both"],
        default="both",
        help=(
            "Which experiment to run. 'on' = continuous batching only, "
            "'off' = max_num_seqs=1 baseline only, 'both' = run both and "
            "write the compare file (default)."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = make_prompts(args.num_requests, args.prompt_mode)

    cont = None
    off = None

    if args.mode in ("on", "both"):
        # Continuous batching ON:
        # All requests are submitted together, and the engine can keep many
        # sequences active.
        print(f"[run] continuous batching ON  (max_num_seqs={args.max_num_seqs})")
        cont = run_one_experiment(
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
            out_dir=out_dir / "continuous_on",
        )
        print(f"[done] wrote: {out_dir / 'continuous_on'}")

    if args.mode in ("off", "both"):
        # Continuous batching OFF baseline:
        # Submit the same requests, but only allow one active sequence.
        # This approximates sequential serving while avoiding model reload per request.
        print("[run] continuous batching OFF (max_num_seqs=1 baseline)")
        off = run_one_experiment(
            model=args.model,
            dtype=args.dtype,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            chunked_prefill=True,
            chunk_size=args.chunk_size,
            max_num_seqs=1,
            max_num_batched_tokens=args.max_num_batched_tokens,
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            gpu_sample_interval=0.2,
            out_dir=out_dir / "continuous_off_max_num_seqs_1",
        )
        print(f"[done] wrote: {out_dir / 'continuous_off_max_num_seqs_1'}")

    # Only emit the compare file when both sides actually ran in this invocation.
    # Running only one side is the intended workflow on small GPUs (e.g. T4)
    # where the other side may OOM / hang.
    if cont is not None and off is not None:
        compare = {
            "note": (
                "continuous_batching_off is a max_num_seqs=1 sequential baseline, "
                "not a separate engine mode that disables continuous batching internally"
            ),
            "continuous_batching_on": _pick_metrics(cont),
            "continuous_batching_off_max_num_seqs_1_baseline": _pick_metrics(off),
        }
        _write_json(out_dir / "continuous_batching_compare.json", compare)
        print(f"[done] wrote: {out_dir / 'continuous_batching_compare.json'}")
    else:
        which = "on" if cont is not None else "off"
        print(
            f"[skip] compare file not written (mode={args.mode}, only ran '{which}'). "
            f"Run with --mode {'off' if which == 'on' else 'on'} later to produce the comparison."
        )


if __name__ == "__main__":
    main()