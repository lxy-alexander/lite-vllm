#!/usr/bin/env python3
"""Benchmark continuous vs static batching (throughput + TTFT).

Tuned defaults for: NVIDIA T4 (16 GiB, fp16) + Qwen/Qwen3-0.6B.

  1) Continuous batching ON
     Requests arrive over time (Poisson). Each arrival immediately calls
     ``stream()`` so the scheduler may add new sequences while earlier
     requests are still decoding — classic continuous batching.

  2) Continuous batching OFF (static batching)
     Simulate static batch inference: accumulate a batch of ``batch_size``
     requests (still using the same arrival process to fill each batch slot),
     submit them all together, ``await asyncio.gather`` until *every*
     request in that batch finishes — only then start the next batch.
     Nothing from the next batch touches the engine during the previous run.

Metrics
  • throughput — total output tokens / wall-clock
  • TTFT — time from local ``submit`` (when ``stream()`` is entered) until
           first streamed token (mean / p50 / p95 / max).

Usage
  python examples/bench_continuous_batching.py
      # default: OFF (static) then ON (continuous), then Comparison table

  python examples/bench_continuous_batching.py --mode continuous
  python examples/bench_continuous_batching.py --mode static --batch-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time

sys.path.insert(0, ".")

from litevllm import AsyncLLM, SamplingParams
from litevllm.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
)


PROMPTS = [
    "Explain in one paragraph what a transformer is.",
    "Write a short story about a robot learning to paint.",
    "List five interesting facts about the deep ocean.",
    "Describe how continuous batching improves LLM throughput.",
    "Summarize the plot of Hamlet in one paragraph.",
    "What are the trade-offs between FP16 and BF16 on GPUs?",
    "Give me a recipe for a simple chocolate cake.",
    "Compare CUDA graphs vs eager execution for LLM decoding.",
]


async def one_request(engine: AsyncLLM, prompt: str,
                      sp: SamplingParams) -> dict:
    submit = time.perf_counter()
    first: float | None = None
    num_tokens = 0
    async for _delta in engine.stream(prompt, sp):
        if first is None:
            first = time.perf_counter()
        num_tokens += 1
    end = time.perf_counter()
    return {
        "ttft": (first - submit) if first is not None else None,
        "latency": end - submit,
        "tokens": num_tokens,
    }


async def driver_continuous(
    engine: AsyncLLM,
    prompts: list[str],
    sp: SamplingParams,
    request_rate: float,
) -> tuple[list[dict], float]:
    tasks: list[asyncio.Task] = []
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        tasks.append(asyncio.create_task(one_request(engine, p, sp)))
        if request_rate > 0 and i < len(prompts) - 1:
            await asyncio.sleep(random.expovariate(request_rate))
    results = await asyncio.gather(*tasks)
    return results, time.perf_counter() - t0


async def driver_static(
    engine: AsyncLLM,
    prompts: list[str],
    sp: SamplingParams,
    batch_size: int,
    request_rate: float,
) -> tuple[list[dict], float]:
    """Static batching: fill a batch (with Poisson waits between arrivals),
    then run that batch to completion before starting another.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    results: list[dict] = []
    t0 = time.perf_counter()
    idx = 0
    batch_idx = 0

    while idx < len(prompts):
        batch_prompts: list[str] = []
        # Simulate arrivals into a buffer — no engine work until batch is launched.
        for _ in range(batch_size):
            if idx >= len(prompts):
                break
            batch_prompts.append(prompts[idx])
            idx += 1
            if request_rate > 0 and idx < len(prompts) and len(batch_prompts) < batch_size:
                await asyncio.sleep(random.expovariate(request_rate))

        batch_idx += 1
        tasks = [
            asyncio.create_task(one_request(engine, p, sp))
            for p in batch_prompts
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    return results, time.perf_counter() - t0


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def summarize_run(results: list[dict], wall: float, label: str) -> dict[str, float]:
    """Compute metrics (and print blocks). Used for comparison table when ON+OFF."""
    total_out = sum(r["tokens"] for r in results)
    ttfts_ms = [r["ttft"] * 1000 for r in results if r["ttft"] is not None]
    latencies_s = [r["latency"] for r in results]

    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"requests        : {len(results)}")
    print(f"output tokens   : {total_out}")
    print(f"wall clock      : {wall:.2f} s")
    print(f"throughput      : {total_out / wall:.1f} tok/s")
    print(f"req throughput  : {len(results) / wall:.2f} req/s")

    print("\nLatency — TTFT (ms)")
    if ttfts_ms:
        print(f"mean : {statistics.fmean(ttfts_ms):8.1f}")
        print(f"p50  : {pct(ttfts_ms, 50):8.1f}")
        print(f"p95  : {pct(ttfts_ms, 95):8.1f}")
        print(f"max  : {max(ttfts_ms):8.1f}")
    else:
        print("n/a")

    print("\nEnd-to-end latency (s)")
    if latencies_s:
        print(f"mean : {statistics.fmean(latencies_s):.2f}   "
              f"p95 : {pct(latencies_s, 95):.2f}   "
              f"max : {max(latencies_s):.2f}")
    else:
        print("n/a")

    tp = total_out / wall if wall > 0 else 0.0
    return {
        "throughput_tok_s": tp,
        "ttft_mean_ms": statistics.fmean(ttfts_ms) if ttfts_ms else 0.0,
        "ttft_p95_ms": pct(ttfts_ms, 95) if ttfts_ms else float("nan"),
        "latency_p95_s": pct(latencies_s, 95) if latencies_s else float("nan"),
    }


def print_comparison(rows: list[tuple[str, dict[str, float]]]) -> None:
    """``rows``: (display_name, summary dict)."""
    print("\n" + "=" * 60)
    print("Comparison  (aggregate tok/s over all concurrent streams)")
    print("=" * 60)
    hdr = (
        f"{'config':<42}{'tok/s':>10}{'TTFT μ ms':>12}{'TTFT p95 ms':>14}"
        f"{'e2e p95 s':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, s in rows:
        print(
            f"{name:<42}{s['throughput_tok_s']:>10.1f}"
            f"{s['ttft_mean_ms']:>12.1f}"
            f"{s['ttft_p95_ms']:>14.1f}"
            f"{s['latency_p95_s']:>12.2f}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous vs static batching (T4 + Qwen3-0.6B defaults)"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--mode",
        choices=("continuous", "static", "both"),
        default="both",
        help="static=OFF (sequential batches); continuous=ON; "
             "both=OFF then ON + comparison table (default)",
    )
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=4.0,
        help="mean req/s (Poisson) between *launches* (continuous) or between "
             "buffered arrivals within a batch (static). 0 = no sleep.",
    )
    parser.add_argument("--batch-size", type=int, default=8,
                        help="static mode: parallel width per wave; wave N+1 starts "
                             "only after wave N finishes")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_requests)]

    engine_config = EngineConfig(
        model_config=ModelConfig(model=args.model, max_model_len=args.max_model_len),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=max(args.max_model_len, 4096),
        ),
        seed=args.seed,
    )

    print(f"Loading model: {args.model}")
    engine = AsyncLLM(engine_config)
    await engine.initialize()

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print("Warmup...")
    await engine.generate("Hello", SamplingParams(temperature=0.0, max_tokens=8))

    modes_to_run = (
        ["static", "continuous"] if args.mode == "both" else [args.mode]
    )

    summaries: list[tuple[str, dict[str, float]]] = []

    for mi, mode in enumerate(modes_to_run):
        random.seed(args.seed)

        print(f"\n{'#' * 60}")
        on_off = (
            "ON (continuous)"
            if mode == "continuous"
            else f"OFF (static, batch={min(args.batch_size, args.max_num_seqs)})"
        )
        print(f"Continuous batching {on_off}")
        print(
            f"num_requests={args.num_requests}, request_rate={args.request_rate}, "
            f"max_tokens={args.max_tokens}, max_num_seqs={args.max_num_seqs}"
        )
        print(f"{'#' * 60}")

        if mode == "continuous":
            results, wall = await driver_continuous(
                engine, prompts, sp, args.request_rate
            )
        else:
            results, wall = await driver_static(
                engine,
                prompts,
                sp,
                batch_size=min(args.batch_size, args.max_num_seqs),
                request_rate=args.request_rate,
            )

        title = (
            mode
            if len(modes_to_run) > 1
            else ("continuous batching ON" if mode == "continuous" else "continuous batching OFF")
        )
        summ = summarize_run(results, wall, title)
        display = (
            "ON  (continuous)"
            if mode == "continuous"
            else f"OFF (static bs={min(args.batch_size, args.max_num_seqs)})"
        )
        summaries.append((display, summ))

        if mi < len(modes_to_run) - 1:
            for _ in range(50):
                if not engine.has_unfinished:
                    break
                await asyncio.sleep(0.01)

    if len(summaries) == 2:
        print_comparison(summaries)


if __name__ == "__main__":
    asyncio.run(main())
