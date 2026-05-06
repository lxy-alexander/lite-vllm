"""Model base class / interface.

Every architecture (Llama, Qwen3) must implement ``BaseModelForCausalLM``
so the engine can drive prefill / decode uniformly.
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
import torch.nn as nn

from litevllm.config import ModelConfig


class BaseModelForCausalLM(nn.Module, abc.ABC):
    """Abstract base class for all causal-LM architectures."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids:   (num_tokens,)
            positions:   (num_tokens,) – absolute position ids
            kv_caches:   per-layer (key_cache, value_cache) tensors
            slot_mapping: (num_tokens,) mapping to physical KV slots
            block_tables: (batch, max_blocks) for decode phase
            context_lens: (batch,) context length per sequence (decode)
            is_prefill:   True during prefill, False during decode

        Returns:
            logits: (num_tokens, vocab_size) or (batch, vocab_size) for decode
        """
        ...

    @abc.abstractmethod
    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load pre-trained weights from a state dict (e.g. from safetensors)."""
        ...

    @classmethod
    def get_model_cls(cls, model_type: str) -> type["BaseModelForCausalLM"]:
        """Registry lookup for model classes."""
        from .llama import LlamaForCausalLM
        from .qwen3 import Qwen3ForCausalLM

        _registry: dict[str, type[BaseModelForCausalLM]] = {
            "llama": LlamaForCausalLM,
            "qwen2": Qwen3ForCausalLM,
            "qwen3": Qwen3ForCausalLM,
        }
        model_cls = _registry.get(model_type)
        if model_cls is None:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {list(_registry.keys())}"
            )
        return model_cls
