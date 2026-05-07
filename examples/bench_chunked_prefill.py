#!/usr/bin/env python3
"""Benchmark Chunked Prefill ON vs OFF.

Final comparison only shows:
  config | tok/s | p99 ttft | p99 tpot
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import math
import random
import statistics
import sys
import time
from typing import Any

import torch

sys.path.insert(0, ".")

from litevllm import AsyncLLM, SamplingParams
from litevllm.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
)


SHORT_PROMPTS = [
    "Explain in one paragraph what a transformer is.",
    "List five interesting facts about the deep ocean.",
    "Summarize the plot of Hamlet in one paragraph.",
    "Give me a recipe for a simple chocolate cake.",
    "What is the capital of France and why is it famous?",
    "Describe the difference between TCP and UDP.",
]

LONG_FILLER = (
    "The following is a long technical document about distributed systems, "
    "consensus protocols, replication, sharding, scheduling, cache management, "
    "fault tolerance, queueing, prefill, decode, throughput, latency, fairness, "
    "tail latency, and GPU utilization. "
)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")

    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]

    rank = (len(ys) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)

    if lo == hi:
        return ys[lo]

    w = rank - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def build_workload(
    *,
    num_short: int,
    num_long: int,
    long_len_words: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    items: list[tuple[str, str]] = []

    for i in range(num_short):
        items.append(("short", SHORT_PROMPTS[i % len(SHORT_PROMPTS)]))

    filler_words = max(1, len(LONG_FILLER.split()))
    repeats = max(1, long_len_words // filler_words)
    long_prompt = (LONG_FILLER * repeats) + "\nQuestion: summarize the above."

    for _ in range(num_long):
        items.append(("long", long_prompt))

    rng.shuffle(items)
    return items


async def one_request(
    engine: AsyncLLM,
    kind: str,
    prompt: str,
    sp: SamplingParams,
) -> dict[str, Any]:
    submit_t = time.perf_counter()
    first_t: float | None = None
    last_token_t: float | None = None

    inter_token_s: list[float] = []
    num_tokens = 0

    async for _delta in engine.stream(prompt, sp):
        now = time.perf_counter()

        if first_t is None:
            first_t = now
        elif last_token_t is not None:
            inter_token_s.append(now - last_token_t)

        last_token_t = now
        num_tokens += 1

    end_t = time.perf_counter()

    return {
        "kind": kind,
        "ttft_s": (first_t - submit_t) if first_t is not None else None,
        "latency_s": end_t - submit_t,
        "tokens": num_tokens,
        "inter_token_s": inter_token_s,
    }


async def driver(
    engine: AsyncLLM,
    items: list[tuple[str, str]],
    sp: SamplingParams,
    request_rate: float,
    seed: int,
) -> tuple[list[dict[str, Any]], float]:
    rng = random.Random(seed)
    tasks: list[asyncio.Task] = []

    t0 = time.perf_counter()

    for i, (kind, prompt) in enumerate(items):
        tasks.append(asyncio.create_task(one_request(engine, kind, prompt, sp)))

        if request_rate > 0 and i < len(items) - 1:
            await asyncio.sleep(rng.expovariate(request_rate))

    results = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - t0

    return results, wall_s


def flatten_inter_token(results: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for r in results:
        vals.extend(r["inter_token_s"])
    return vals


def summarize(
    *,
    label: str,
    trial: int,
    results: list[dict[str, Any]],
    wall_s: float,
) -> dict[str, Any]:
    total_out = sum(r["tokens"] for r in results)
    total_req = len(results)

    n_short = sum(1 for r in results if r["kind"] == "short")
    n_long = sum(1 for r in results if r["kind"] == "long")

    ttft_all_ms = [
        r["ttft_s"] * 1000.0
        for r in results
        if r["ttft_s"] is not None
    ]

    inter_token_ms = [x * 1000.0 for x in flatten_inter_token(results)]

    output_tps = total_out / wall_s if wall_s > 0 else 0.0
    request_tps = total_req / wall_s if wall_s > 0 else 0.0

    summary = {
        "label": label,
        "trial": trial,
        "requests": total_req,
        "num_short": n_short,
        "num_long": n_long,
        "wall_s": wall_s,
        "output_tokens": total_out,
        "request_throughput_req_s": request_tps,
        "output_throughput_tok_s": output_tps,
        "ttft_all_p99_ms": pct(ttft_all_ms, 99),
        "tpot_inter_token_p99_ms": pct(inter_token_ms, 99),
    }

    print("\n" + "=" * 72)
    print(f"{label} | trial={trial}")
    print("=" * 72)
    print(f"requests          : {total_req}  (short={n_short}, long={n_long})")
    print(f"output tokens     : {total_out}")
    print(f"wall clock        : {wall_s:.2f} s")
    print(f"request throughput: {request_tps:.3f} req/s")
    print(f"output throughput : {output_tps:.1f} tok/s")
    print(f"p99 TTFT          : {summary['ttft_all_p99_ms']:.1f} ms")
    print(f"p99 TPOT          : {summary['tpot_inter_token_p99_ms']:.3f} ms")

    return summary


async def run_one(
    *,
    label: str,
    trial: int,
    model: str,
    dtype: str,
    chunked_prefill: bool,
    chunk_size: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    block_size: int,
    gpu_memory_utilization: float,
    items: list[tuple[str, str]],
    sp: SamplingParams,
    request_rate: float,
    seed: int,
) -> dict[str, Any]:
    engine_config = EngineConfig(
        model_config=ModelConfig(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            chunked_prefill=chunked_prefill,
            chunk_size=chunk_size,
        ),
        seed=seed,
    )

    print(
        f"\n>>> Loading engine: {label} | trial={trial} "
        f"(chunked_prefill={chunked_prefill}, chunk_size={chunk_size})"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    engine = AsyncLLM(engine_config)
    await engine.initialize()

    await engine.generate("Hello", SamplingParams(temperature=0.0, max_tokens=8))

    results, wall_s = await driver(
        engine=engine,
        items=items,
        sp=sp,
        request_rate=request_rate,
        seed=seed,
    )

    summary = summarize(
        label=label,
        trial=trial,
        results=results,
        wall_s=wall_s,
    )

    del engine
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def aggregate(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "label": label,
            "trials": 0,
            "output_throughput_tok_s_mean": float("nan"),
            "ttft_all_p99_ms_mean": float("nan"),
            "tpot_inter_token_p99_ms_mean": float("nan"),
        }

    keys = [
        "output_throughput_tok_s",
        "ttft_all_p99_ms",
        "tpot_inter_token_p99_ms",
    ]

    out: dict[str, Any] = {
        "label": label,
        "trials": len(rows),
    }

    for k in keys:
        vals = [r[k] for r in rows if k in r and not math.isnan(r[k])]
        out[k + "_mean"] = mean(vals)

    return out


def print_comparison(off: dict[str, Any], on: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("Comparison")
    print("=" * 64)

    header = (
        f"{'config':<24}"
        f"{'tok/s':>10}"
        f"{'p99 ttft':>14}"
        f"{'p99 tpot':>14}"
    )
    print(header)
    print("-" * len(header))

    rows = [
        (
            "Chunked Prefill OFF",
            off["output_throughput_tok_s_mean"],
            off["ttft_all_p99_ms_mean"],
            off["tpot_inter_token_p99_ms_mean"],
        ),
        (
            "Chunked Prefill ON",
            on["output_throughput_tok_s_mean"],
            on["ttft_all_p99_ms_mean"],
            on["tpot_inter_token_p99_ms_mean"],
        ),
    ]

    for label, tok_s, p99_ttft, p99_tpot in rows:
        print(
            f"{label:<24}"
            f"{tok_s:>10.1f}"
            f"{p99_ttft:>14.1f}"
            f"{p99_tpot:>14.3f}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked Prefill ON vs OFF benchmark."
    )

    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float16")

    parser.add_argument("--num-short", type=int, default=64)
    parser.add_argument("--num-long", type=int, default=32)
    parser.add_argument(
        "--long-len",
        type=int,
        default=1800,
        help="approximate long-prompt length in words",
    )

    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=8.0,
        help="Poisson arrival rate. <=0 submits all requests immediately.",
    )

    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)

    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--randomize-order",
        action="store_true",
        help="Randomize whether OFF or ON runs first in each trial.",
    )

    args = parser.parse_args()

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
    )

    off_rows: list[dict[str, Any]] = []
    on_rows: list[dict[str, Any]] = []

    for trial_idx in range(args.trials):
        trial = trial_idx + 1
        trial_seed = args.seed + trial_idx

        items = build_workload(
            num_short=args.num_short,
            num_long=args.num_long,
            long_len_words=args.long_len,
            seed=trial_seed,
        )

        common = dict(
            trial=trial,
            model=args.model,
            dtype=args.dtype,
            chunk_size=args.chunk_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            items=items,
            sp=sp,
            request_rate=args.request_rate,
            seed=trial_seed,
        )

        configs = [
            ("Chunked Prefill OFF", False),
            ("Chunked Prefill ON", True),
        ]

        if args.randomize_order:
            random.Random(trial_seed).shuffle(configs)

        for label, enabled in configs:
            row = await run_one(
                label=label,
                chunked_prefill=enabled,
                **common,
            )

            if enabled:
                on_rows.append(row)
            else:
                off_rows.append(row)

    off_agg = aggregate("Chunked Prefill OFF", off_rows)
    on_agg = aggregate("Chunked Prefill ON", on_rows)

    print_comparison(off_agg, on_agg)


if __name__ == "__main__":
    asyncio.run(main())