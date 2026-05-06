# litevllm

A minimal, single-GPU LLM inference engine in the style of vLLM. The goal is
to expose the few mechanisms that actually matter for serving (PagedAttention,
Continuous Batching, Chunked Prefill) in a small, readable codebase you can
profile, modify, and benchmark on a single A100.

## What's implemented

- **PagedAttention** — block-based KV cache with per-sequence block tables
  and copy-on-write fork support.
- **Continuous Batching** — every step the scheduler reconsiders the active
  batch: finished sequences free their blocks, new requests slot in.
- **Chunked Prefill** — long prompts are sliced into fixed-size chunks with
  decode-first token-budget allocation (Running sequences each get 1-token
  generation budget before any prefill is admitted).
- **Prefix Cache (optional)** — content-hashed KV blocks reused across
  identical prompt prefixes.
- **Triton Sampler** — fused argmax / temperature-scaled softmax kernel for
  the hot path, with a PyTorch fallback for CPU and unsupported sizes.
- **Models** — Llama and Qwen3 (loaded from HF safetensors).

## What's *not* here on purpose

- No tensor / pipeline parallelism (single GPU only).
- No quantization (AWQ / GPTQ / FP8).
- No speculative decoding, no PD-disaggregation, no router.
- No CUDA-graph capture for decode.

If you need any of those, this isn't the right project — go to vLLM.

## Project layout

```text
litevllm/
├── config/          # ModelConfig, EngineConfig, CacheConfig,
│                    # SchedulerConfig, SamplingParams
├── tokenizer/       # Process-safe AutoTokenizer wrapper
├── cache/           # KVCache, BlockManager, PrefixCache
├── layers/          # PagedAttention, RoPE, Sampler (Triton),
│                    # plain nn.Linear wrappers
├── models/          # Llama, Qwen3
├── engine/          # Sequence state machine, Scheduler,
│                    # ModelRunner, LLMEngine
└── entrypoints/     # LLM (sync), AsyncLLM (streaming)
examples/            # basic.py, async_streaming.py
tests/               # pytest unit tests
bench.py             # throughput benchmark
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[test]"   # for pytest
```

Targeted at any single CUDA GPU with PyTorch ≥ 2.1. Triton is a hard
dependency (used by the sampler).

### GPU support matrix

| GPU class | dtype (auto) | Notes |
|---|---|---|
| T4  (sm_75) | float16 | bf16 emulated → slow, auto-detected and avoided; FlashAttention 2 unsupported, PyTorch SDPA falls back to memory-efficient impl |

### T4 (16 GB) suggested config

```python
LLM(
    "Qwen/Qwen3-0.6B",
    dtype="float16",                # explicit, avoids any bf16 path
    gpu_memory_utilization=0.80,    # 16 GB is tight; leave headroom
    max_num_batched_tokens=2048, 
    max_num_seqs=64,
)
```

Don't install `flash-attn` on T4 — it requires sm_80+.

## Usage

### Python API — sync

```python
from litevllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B")
outputs = llm.generate(
    ["Hello, my name is", "The future of AI is"],
    SamplingParams(temperature=0.7, max_tokens=128),
)
for out in outputs:
    print(f"{out.prompt_text} -> {out.output_text}")
```

### Python API — async streaming

```python
import asyncio
from litevllm import AsyncLLM, SamplingParams
from litevllm.config import EngineConfig, ModelConfig

async def main():
    engine = AsyncLLM(EngineConfig(model_config=ModelConfig(model="Qwen/Qwen3-0.6B")))
    await engine.initialize()
    async for chunk in engine.stream("Once upon a time,", SamplingParams(max_tokens=64)):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### CLI

```bash
python -m litevllm --model Qwen/Qwen3-0.6B --prompt "Hello!" --max-tokens 64
```

### Examples

```bash
python examples/basic.py --model Qwen/Qwen3-0.6B
python examples/async_streaming.py --model Qwen/Qwen3-0.6B
```

### Benchmark

```bash
python bench.py --model Qwen/Qwen3-0.6B --num-seqs 256 --max-output-len 1024
```

### Tests

```bash
pytest -q
```

The unit tests run on CPU only (no GPU or model download required) and cover
the BlockManager, PrefixCache, Scheduler (chunked prefill / continuous
batching / abort), Sampler fallback, and RoPE invariants.

## Tuning knobs

Exposed on `LLM(...)` and `EngineConfig`:

| Knob                     | Default | Effect                                    |
|--------------------------|---------|-------------------------------------------|
| `block_size`             | 16      | Tokens per KV block                       |
| `gpu_memory_utilization` | 0.85    | Fraction of GPU memory reserved for KV    |
| `max_num_seqs`           | 256     | Concurrent sequences cap                  |
| `max_num_batched_tokens` | 8192    | Token budget per scheduler step           |
| `chunk_size`             | 512     | Prefill slice length (chunked prefill)    |
| `enable_prefix_caching`  | False   | Enable hash-based prefix KV reuse         |

## License

Apache 2.0
