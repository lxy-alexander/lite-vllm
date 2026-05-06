"""ModelConfig / EngineConfig / SamplingParams.

Single-GPU (A100) configuration. No tensor / pipeline parallel, no quantization,
no speculative decoding — just the knobs needed to drive the engine on one card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelConfig:
    model: str
    dtype: str = "auto"
    trust_remote_code: bool = True
    revision: Optional[str] = None
    max_model_len: Optional[int] = None

    hf_config: Optional[object] = field(default=None, repr=False)
    vocab_size: int = 0
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 4096
    model_type: str = ""
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_qk_norm: bool = False

    def resolve(self) -> None:
        """Load the HF config and populate architecture fields."""
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self.hf_config = cfg
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size
        self.num_hidden_layers = cfg.num_hidden_layers
        self.num_attention_heads = cfg.num_attention_heads
        self.num_key_value_heads = getattr(
            cfg, "num_key_value_heads", cfg.num_attention_heads
        )
        self.head_dim = getattr(
            cfg, "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.intermediate_size = getattr(cfg, "intermediate_size", 0)

        raw_rope_scaling = getattr(cfg, "rope_scaling", None)
        self.rope_theta = getattr(cfg, "rope_theta", None)
        if self.rope_theta is None and isinstance(raw_rope_scaling, dict):
            self.rope_theta = raw_rope_scaling.get("rope_theta")
        if self.rope_theta is None:
            self.rope_theta = 10000.0

        if isinstance(raw_rope_scaling, dict):
            rope_type = raw_rope_scaling.get("type", raw_rope_scaling.get("rope_type"))
            factor = raw_rope_scaling.get("factor")
            if rope_type in (None, "default"):
                self.rope_scaling = None
            else:
                self.rope_scaling = {
                    "type": rope_type,
                    "factor": float(factor) if factor is not None else 1.0,
                }
        else:
            self.rope_scaling = None

        self.max_position_embeddings = getattr(cfg, "max_position_embeddings", 4096)
        self.model_type = getattr(cfg, "model_type", "")
        # Qwen2 hardcodes attention bias=True even though its config doesn't
        # expose `attention_bias`. Other archs default to False.
        self.attention_bias = bool(
            getattr(cfg, "attention_bias", self.model_type == "qwen2")
        )
        self.rms_norm_eps = getattr(cfg, "rms_norm_eps", 1e-6)
        self.tie_word_embeddings = getattr(cfg, "tie_word_embeddings", False)
        self.use_qk_norm = bool(
            getattr(
                cfg,
                "use_qk_norm",
                getattr(cfg, "qk_norm", self.model_type == "qwen3"),
            )
        )

        if self.max_model_len is None:
            self.max_model_len = self.max_position_embeddings

    @property
    def torch_dtype(self) -> torch.dtype:
        """Resolve a dtype string to a ``torch.dtype``.

        Auto rule:
          - Ampere+ (sm_80+, e.g. A100/H100) → bfloat16 (native tensor cores,
            no fp16 overflow risk on long sequences).
          - Turing (sm_75, e.g. T4) → float16. T4 reports
            ``torch.cuda.is_bf16_supported() == True`` because of software
            emulation, but bf16 there is multiple times slower than fp16.
          - CPU → float32.
        """
        if self.dtype == "auto":
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                if cap >= (8, 0):
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.dtype, torch.float16)


@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: Optional[int] = None
    num_cpu_blocks: int = 0
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = False


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    chunked_prefill: bool = True
    chunk_size: int = 512


@dataclass
class EngineConfig:
    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device: str = "auto"
    seed: int = 0

    def resolve(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_config.resolve()


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    min_tokens: int = 0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    skip_special_tokens: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
