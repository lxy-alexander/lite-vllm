#!/usr/bin/env python3
"""litevllm async streaming inference example — single GPU.

Streams the output token-by-token. Run with:
    python examples/async_streaming.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")

from litevllm import AsyncLLM, SamplingParams
from litevllm.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig


async def main() -> None:
    parser = argparse.ArgumentParser(description="litevllm async streaming example")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", type=str, default="auto")
    args = parser.parse_args()

    engine_config = EngineConfig(
        model_config=ModelConfig(model=args.model, dtype=args.dtype),
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig(),
    )

    print(f"Loading model: {args.model}")
    engine = AsyncLLM(engine_config)
    await engine.initialize()
    print("Model loaded. Streaming output:\n")

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"> {args.prompt}", end="", flush=True)
    t_first = None
    t_start = time.time()
    num_chunks = 0
    async for delta in engine.stream(args.prompt, params):
        if t_first is None:
            t_first = time.time() - t_start
        num_chunks += 1
        print(delta, end="", flush=True)
    elapsed = time.time() - t_start

    print("\n")
    print(f"[ttft] {t_first*1000:.1f} ms" if t_first is not None else "[ttft] n/a")
    print(f"[total] {elapsed:.2f}s, {num_chunks} chunks streamed")


if __name__ == "__main__":
    asyncio.run(main())
