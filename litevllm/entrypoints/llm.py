"""LLM — offline synchronous inference entrypoint (single GPU, basic mode).

Usage:
    from litevllm import LLM, SamplingParams

    llm = LLM("Qwen/Qwen3-0.6B")
    outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=128))
    for out in outputs:
        print(out.output_text)
"""

from __future__ import annotations

import argparse
from typing import Optional

from litevllm.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SamplingParams,
    SchedulerConfig,
)
from litevllm.engine.llm_engine import LLMEngine
from litevllm.engine.sequence import SequenceGroupOutput


class LLM:
    """High-level offline LLM inference class.

    Loads the model once and runs batch generation synchronously on a single GPU.
    """

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.85,
        block_size: int = 16,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        enable_prefix_caching: bool = False,
        chunked_prefill: bool = True,
        chunk_size: int = 512,
        trust_remote_code: bool = True,
        seed: int = 0,
    ) -> None:
        engine_config = EngineConfig(
            model_config=ModelConfig(
                model=model,
                dtype=dtype,
                max_model_len=max_model_len,
                trust_remote_code=trust_remote_code,
            ),
            cache_config=CacheConfig(
                block_size=block_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=enable_prefix_caching,
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                chunked_prefill=chunked_prefill,
                chunk_size=chunk_size,
            ),
            seed=seed,
        )
        self.engine = LLMEngine(engine_config)
        self.engine.initialize()

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> list[SequenceGroupOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single string or list of prompt strings.
            sampling_params: Sampling configuration. Defaults to greedy.

        Returns:
            List of ``SequenceGroupOutput``, one per prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

        for prompt in prompts:
            self.engine.add_request(prompt, sampling_params)
        return self.engine.run_to_completion()

    def __repr__(self) -> str:
        return f"LLM(model={self.engine.config.model_config.model!r})"


def main() -> None:
    """CLI entrypoint: ``python -m litevllm --model ... --prompt ...``"""
    parser = argparse.ArgumentParser(description="litevllm offline inference")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello!")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    llm = LLM(model=args.model, dtype=args.dtype, seed=args.seed)
    outputs = llm.generate([args.prompt], params)
    for out in outputs:
        print(f"\nPrompt: {out.prompt_text}")
        print(f"Output: {out.output_text}")
        print(f"Tokens: {out.num_prompt_tokens} prompt + {out.num_output_tokens} generated")
