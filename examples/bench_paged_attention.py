#!/usr/bin/env python3
"""Benchmark PagedAttention ON vs OFF (throughput + peak GPU memory).

Tuned defaults for: NVIDIA T4 (16 GiB, fp16) + Qwen/Qwen3-0.6B.

Idea
  PagedAttention's value is the *block-paged* KV cache: each sequence
  reserves KV memory in small fixed blocks (e.g. 16 tokens) instead of
  pre-allocating the full ``max_model_len`` per request.

  This bench simulates "PagedAttention OFF" by setting
  ``block_size = max_model_len`` — every sequence then reserves one
  giant block, the same memory pattern as a non-paged KV cache.

Metrics
  1) throughput  : total decode tokens / wall-clock seconds
  2) peak memory : torch.cuda.max_memory_allocated()

Usage
  python examples/bench_paged_attention.py
  python examples/bench_paged_attention.py --num-requests 128 --gpu-memory-utilization 0.4
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, ".")

from litevllm import LLM, SamplingParams


PROMPTS = [
    "Explain in detail what a transformer is.",
    "Write a short story about a robot learning to paint.",
    "List five interesting facts about the deep ocean.",
    "Describe how PagedAttention reduces KV cache fragmentation.",
    "Summarize the plot of Hamlet in one paragraph.",
    "What are the trade-offs between FP16 and BF16 on GPUs?",
    "Give me a recipe for a simple chocolate cake.",
    "Compare CUDA graphs vs eager execution for LLM decoding.",
]


def run_one(label: str, **cfg) -> dict:
    print("\n" + "=" * 70)
    print(f"[{label}] block_size={cfg['block_size']}, "
          f"max_model_len={cfg['max_model_len']}, "
          f"max_num_seqs={cfg['max_num_seqs']}, "
          f"gpu_mem_util={cfg['gpu_memory_utilization']}")
    print("=" * 70)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=cfg["model"],
        block_size=cfg["block_size"],
        max_model_len=cfg["max_model_len"],
        max_num_seqs=cfg["max_num_seqs"],
        max_num_batched_tokens=max(cfg["max_model_len"], 4096),
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prefix_caching=False,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"])

    # Warmup so JIT / kernel autotune isn't billed to the timed run.
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(cfg["prompts"], sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_in = sum(o.num_prompt_tokens for o in outputs)
    total_out = sum(o.num_output_tokens for o in outputs)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    decode_tps = total_out / elapsed if elapsed > 0 else 0.0
    overall_tps = (total_in + total_out) / elapsed if elapsed > 0 else 0.0

    print(f"requests        : {len(outputs)}")
    print(f"prompt tokens   : {total_in}")
    print(f"output tokens   : {total_out}")
    print(f"elapsed         : {elapsed:.2f} s")
    print(f"decode tok/s    : {decode_tps:.1f}")
    print(f"overall tok/s   : {overall_tps:.1f}")
    print(f"peak GPU memory : {peak_mb:.1f} MiB")

    result = {
        "label": label,
        "decode_tps": decode_tps,
        "overall_tps": overall_tps,
        "peak_mb": peak_mb,
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PagedAttention ON vs OFF (T4 + Qwen3-0.6B defaults)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024,
                        help="also used as the OFF-case block_size")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                        help="lower = tighter KV budget = bigger PagedAttention win")
    parser.add_argument("--paged-block-size", type=int, default=16)

    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_requests)]
    common = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompts=prompts,
        max_tokens=args.max_tokens,
    )

    # OFF first, so its numbers aren't advantaged by warm caches.
    off = run_one(
        "PagedAttention OFF (block_size = max_model_len)",
        block_size=args.max_model_len,
        **common,
    )
    on = run_one(
        f"PagedAttention ON  (block_size = {args.paged_block_size})",
        block_size=args.paged_block_size,
        **common,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    header = f"{'config':<48}{'tok/s':>10}{'peak MiB':>14}"
    print(header)
    print("-" * len(header))
    for r in (off, on):
        print(f"{r['label']:<48}{r['decode_tps']:>10.1f}{r['peak_mb']:>14.1f}")

    if off["decode_tps"] > 0:
        print(f"\nthroughput  ON / OFF : {on['decode_tps'] / off['decode_tps']:.2f}x")
        print(f"peak memory ON - OFF : {on['peak_mb'] - off['peak_mb']:+.1f} MiB")


if __name__ == "__main__":
    main()
