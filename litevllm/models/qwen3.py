"""Qwen3 model.

Architecture is close to Llama (GQA + RoPE + SwiGLU) but with:
- Explicit head_dim (may differ from hidden_size // num_heads)
- QK RMSNorm (per-head normalization on Q and K before RoPE)
- No attention bias by default
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from litevllm.config import ModelConfig
from litevllm.layers.attention import Attention
from litevllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from litevllm.layers.rotary_emb import RotaryEmbedding, apply_rotary_emb
from litevllm.models.base import BaseModelForCausalLM
from litevllm.models.llama import LlamaRMSNorm, LlamaMLP


class Qwen3Attention(nn.Module):
    """Qwen3 attention with QK RMSNorm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = LlamaRMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = LlamaRMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rotary_emb = RotaryEmbedding(
            head_dim, max_position_embeddings, rope_theta, rope_scaling,
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

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rotary_emb(positions)
        q = apply_rotary_emb(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = apply_rotary_emb(k, cos.unsqueeze(1), sin.unsqueeze(1))

        key_cache, value_cache = kv_cache if kv_cache is not None else (None, None)
        attn_output = self.attn(
            q, k, v,
            key_cache=key_cache, value_cache=value_cache,
            block_table=block_table, slot_mapping=slot_mapping,
            context_lens=context_lens, is_prefill=is_prefill,
        )
        return self.o_proj(attn_output.reshape(num_tokens, -1))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_bias=config.attention_bias,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=config.use_qk_norm,
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


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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


class Qwen3ForCausalLM(BaseModelForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Qwen2.x checkpoints often omit lm_head and tie it to embeddings.
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
        tied_lm_head_fallback = (
            self.config.tie_word_embeddings
            and "lm_head.weight" not in weights
            and "model.embed_tokens.weight" in weights
        )
        if tied_lm_head_fallback:
            print("[litevllm] lm_head.weight missing; using tied embed_tokens.weight")

        consumed: set[str] = set()
        for name, param in self.named_parameters():
            key = _find_weight_key(name, weights)
            if key is not None:
                if param.shape != weights[key].shape:
                    print(f"[warn] shape mismatch for {name}: "
                          f"model={param.shape} vs ckpt={weights[key].shape}, skipping")
                    continue
                param.data.copy_(weights[key])
                consumed.add(key)
            elif name == "lm_head.weight" and tied_lm_head_fallback:
                continue
            else:
                print(f"[warn] weight not found for param: {name}")
        for name, buf in self.named_buffers():
            key = _find_weight_key(name, weights)
            if key is not None:
                buf.copy_(weights[key])
                consumed.add(key)

        unused = [k for k in weights.keys() if k not in consumed]
        if unused:
            preview = ", ".join(unused[:8])
            suffix = f" (and {len(unused) - 8} more)" if len(unused) > 8 else ""
            print(f"[warn] {len(unused)} checkpoint tensor(s) ignored: {preview}{suffix}")


def _find_weight_key(name: str, weights: dict[str, torch.Tensor]) -> Optional[str]:
    if name in weights:
        return name
    simplified = name.replace(".linear.", ".")
    if simplified in weights:
        return simplified
    return None
