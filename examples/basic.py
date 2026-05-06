#!/usr/bin/env python3
"""litevllm offline inference example (basic mode) — synchronous, single-GPU, minimal runnable setup

Usage:
    python examples/basic.py --model Qwen/Qwen3-0.6B

Local model path:
    python examples/basic.py --model ~/models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from litevllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description="litevllm basic offline inference")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    # ── 1. Load model ──
    print(f"Loading model: {args.model}")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── 2. Define sampling params ──
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # ── 3. Batch inference ──
    prompts = [
        "Hello, my name is",
    ]

    print(f"\nGenerating {len(prompts)} prompts with max_tokens={args.max_tokens}...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t1
    total_output_tokens = sum(o.num_output_tokens for o in outputs)

    # ── 4. Print results ──
    print("\n" + "=" * 70)
    for i, output in enumerate(outputs):
        print(f"\n[Prompt {i+1}] {output.prompt_text}")
        print(f"[Output]  {output.output_text}")
        print(f"[Tokens]  {output.num_prompt_tokens} tokens + {output.num_output_tokens} output")
        print(f"[Finish]  {output.finish_reason}")
        print("-" * 70)

    print(f"\nTotal: {total_output_tokens} tokens in {elapsed:.2f}s "
          f"({total_output_tokens / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
