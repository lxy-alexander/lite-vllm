import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class PagedAttention:
    """Efficient PagedAttention helper with batched cache reads."""

    @staticmethod
    def write_to_cache(
        key: torch.Tensor,       # (num_tokens, num_kv_heads, head_dim)
        value: torch.Tensor,     # (num_tokens, num_kv_heads, head_dim)
        # (num_blocks, block_size, num_kv_heads, head_dim)
        key_cache: torch.Tensor,
        # (num_blocks, block_size, num_kv_heads, head_dim)
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,  # (num_tokens,)
    ) -> None:
        block_size = key_cache.shape[1]
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        # Writes usually happen during prefill, where token counts are high.
        key_cache[block_indices, block_offsets] = key
        value_cache[block_indices, block_offsets] = value

    @staticmethod
    def read_from_cache_batched(
        key_cache: torch.Tensor,    # (num_blocks, block_size, H_kv, D)
        value_cache: torch.Tensor,
        block_table: torch.Tensor,  # (batch_size, max_blocks_per_seq)
        context_lens: torch.Tensor,  # (batch_size,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read KV cache in batch to avoid Python loops."""
        batch_size = block_table.shape[0]
        max_ctx_len = context_lens.max().item()
        block_size = key_cache.shape[1]

        # Build per-token position indices for each sequence: (B, max_ctx_len).
        positions = torch.arange(
            max_ctx_len, device=block_table.device).unsqueeze(0)
        positions = positions.expand(batch_size, max_ctx_len)

        # Compute physical block indices and in-block offsets.
        # block_table stores physical block IDs used by each sequence.
        tbl_idx = positions // block_size
        # Clamp indices to avoid out-of-range access (mask still filters later).
        tbl_idx = torch.clamp(tbl_idx, max=block_table.shape[1] - 1)

        physical_block_ids = torch.gather(block_table, 1, tbl_idx)
        block_offsets = positions % block_size

        # Single-pass indexed gather: (Batch, Max_Ctx, Num_KV_Heads, Head_Dim).
        keys = key_cache[physical_block_ids, block_offsets]
        values = value_cache[physical_block_ids, block_offsets]

        return keys, values


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, num_kv_heads, scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = scale or (1.0 / (head_dim**0.5))

    def forward(self, query, key, value, key_cache, value_cache, block_table, slot_mapping, context_lens, is_prefill=True):
        # Batched cache write.
        if slot_mapping is not None:
            block_size = key_cache.shape[1]
            b_idx = slot_mapping // block_size
            b_off = slot_mapping % block_size
            key_cache[b_idx, b_off] = key
            value_cache[b_idx, b_off] = value

        if is_prefill:
            return self._prefill(query, key, value)

        return self._decode_fast(query, key_cache, value_cache, block_table, context_lens)

    def _decode_fast(self, query, key_cache, value_cache, block_table, context_lens):
        """Naive paged-attention decode in pure torch.

        Shapes are fully determined by ``block_table.shape[1]`` (i.e. by the
        number of columns the caller padded the table to), so the function is
        CUDA-graph capturable: no ``.item()``, no shape-dependent branching.
        """
        # query: (B, H, D)
        B, H, D = query.shape
        block_size = key_cache.shape[1]
        max_s = block_table.shape[1] * block_size
        H_kv = self.num_kv_heads
        G = self.num_kv_groups

        # positions: (1, max_s) – broadcast across batch via gather/compare.
        positions = torch.arange(max_s, device=query.device).unsqueeze(0)

        # Map each logical position -> physical (block_idx, block_offset).
        # block_table is int32; gather requires int64 indices + matching dtype on output,
        # but advanced indexing on key_cache works with either, so we cast once.
        b_idx = torch.gather(
            block_table.to(torch.long), 1,
            (positions // block_size).expand(B, max_s),
        )
        b_off = (positions % block_size).expand(B, max_s)

        # (B, max_s, H_kv, D)
        k_history = key_cache[b_idx, b_off]
        v_history = value_cache[b_idx, b_off]

        # Reshape for GQA without repeat_interleave:
        #   q -> (B, H_kv, G, D)
        #   k -> (B, H_kv, max_s, D), v same
        q = query.view(B, H_kv, G, D)
        k = k_history.permute(0, 2, 1, 3).contiguous()
        v = v_history.permute(0, 2, 1, 3).contiguous()

        # Score: (B, H_kv, G, max_s)
        attn = torch.einsum("bhgd,bhsd->bhgs", q, k) * self.scale

        # Mask: (B, 1, 1, max_s); True where the slot is real history.
        mask = positions < context_lens.unsqueeze(1).to(positions.dtype)
        attn = attn.masked_fill(~mask.view(B, 1, 1, max_s), float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)

        # (B, H_kv, G, D) -> (B, H, D)
        out = torch.einsum("bhgs,bhsd->bhgd", attn, v)
        return out.reshape(B, H, D)

    def _prefill(self, q, k, v):
        # Keep prefill path efficient.
        q = q.transpose(0, 1).unsqueeze(0)  # (1, H, S, D)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        return F.scaled_dot_product_attention(q, k, v, is_causal=True).squeeze(0).transpose(0, 1)
