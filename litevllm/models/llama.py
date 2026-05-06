"""LLaMA-family model (LLaMA 2/3, CodeLlama, etc.)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from litevllm.config import ModelConfig
from litevllm.layers.attention import Attention
from litevllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from litevllm.layers.rotary_emb import RotaryEmbedding, apply_rotary_emb
from litevllm.models.base import BaseModelForCausalLM


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size)

        self.rotary_emb = RotaryEmbedding(
            head_dim, max_position_embeddings, rope_theta, rope_scaling
        )
        self.attn = Attention(self.num_heads, head_dim, self.num_kv_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        q = self.q_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)

        cos, sin = self.rotary_emb(positions)
        q = apply_rotary_emb(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = apply_rotary_emb(k, cos.unsqueeze(1), sin.unsqueeze(1))

        key_cache, value_cache = kv_cache if kv_cache is not None else (None, None)

        attn_output = self.attn(
            q, k, v,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            is_prefill=is_prefill,
        )
        return self.o_proj(attn_output.reshape(num_tokens, -1))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_bias=config.attention_bias,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, kv_cache, slot_mapping,
            block_table, context_lens, is_prefill,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden_states = layer(
                hidden_states, positions, kv, slot_mapping,
                block_tables, context_lens, is_prefill,
            )
        return self.norm(hidden_states)


class LlamaForCausalLM(BaseModelForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
        hidden_states = self.model(
            input_ids, positions, kv_caches, slot_mapping,
            block_tables, context_lens, is_prefill,
        )
        return self.lm_head(hidden_states)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        for name, param in self.named_parameters():
            key = _resolve_weight_key(name, weights)
            if key is not None and weights[key].shape == param.shape:
                param.data.copy_(weights[key])
        for name, buf in self.named_buffers():
            key = _resolve_weight_key(name, weights)
            if key is not None:
                buf.copy_(weights[key])


def _resolve_weight_key(
    name: str, weights: dict[str, torch.Tensor]
) -> Optional[str]:
    """Map a model parameter name to the matching checkpoint key.

    Our ``ColumnParallelLinear`` / ``RowParallelLinear`` wrappers expose the
    actual ``nn.Linear`` as ``.linear``; HF checkpoints don't have that prefix.
    """
    if name in weights:
        return name
    simplified = name.replace(".linear.", ".")
    if simplified in weights:
        return simplified
    return None
