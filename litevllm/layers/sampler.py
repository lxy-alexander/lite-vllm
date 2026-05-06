"""Sampling: temperature / top-p / top-k / repetition penalty.

The hot path (greedy argmax and temperature-scaled softmax) is implemented in
Triton so that it fuses into a single GPU launch instead of multiple PyTorch
ops. Top-p / top-k filtering still uses ``torch.sort`` on the GPU (sorting in
Triton is awkward) and the final ``torch.multinomial`` draws the token id.

A pure-PyTorch fallback is used when Triton is unavailable, the input is on
CPU, or Triton fails to JIT-compile at runtime (e.g. missing ``Python.h``,
no nvcc, etc.). On the first failure we log a one-time warning and switch the
process over to the PyTorch path permanently.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from litevllm.config import SamplingParams

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


_TRITON_DISABLED = os.environ.get("LITEVLLM_DISABLE_TRITON", "0") == "1"
_TRITON_WARNED = False


def _disable_triton(reason: Exception) -> None:
    """Permanently disable Triton for this process and emit a one-shot warning."""
    global _TRITON_DISABLED, _TRITON_WARNED
    _TRITON_DISABLED = True
    if not _TRITON_WARNED:
        _TRITON_WARNED = True
        warnings.warn(
            f"[litevllm] Triton sampler kernel unavailable ({reason!s}); "
            "falling back to PyTorch ops. To re-enable, install Python dev "
            "headers (e.g. `dnf install python3-devel` or `apt install "
            "python3-dev`) and a working C compiler.",
            RuntimeWarning,
            stacklevel=2,
        )


def _triton_active() -> bool:
    return _HAS_TRITON and not _TRITON_DISABLED


if _HAS_TRITON:

    @triton.jit
    def _argmax_kernel(
        logits_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program per row; tile across the vocab axis with reductions."""
        row = tl.program_id(0)
        row_ptr = logits_ptr + row * n_cols

        best_val = tl.full([], float("-inf"), dtype=tl.float32)
        best_idx = tl.zeros([], dtype=tl.int64)

        for start in range(0, n_cols, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n_cols
            x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
            local_max = tl.max(x, axis=0)
            local_idx = start + tl.argmax(x, axis=0).to(tl.int64)
            update = local_max > best_val
            best_val = tl.where(update, local_max, best_val)
            best_idx = tl.where(update, local_idx, best_idx)

        tl.store(out_ptr + row, best_idx)

    @triton.jit
    def _softmax_temp_kernel(
        logits_ptr,
        out_ptr,
        temperature,
        n_cols: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fused (logits / temperature) -> softmax over the last dim."""
        row = tl.program_id(0)
        row_ptr = logits_ptr + row * n_cols
        out_row = out_ptr + row * n_cols
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols

        x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        x = x / temperature
        x_max = tl.max(x, axis=0)
        x = tl.exp(x - x_max)
        x = tl.where(mask, x, 0.0)
        denom = tl.sum(x, axis=0)
        probs = x / denom
        tl.store(out_row + offs, probs, mask=mask)


def _triton_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Triton-fused argmax along dim=-1. Returns int64 indices of shape (B,)."""
    if not _triton_active() or not logits.is_cuda:
        return logits.argmax(dim=-1)
    logits = logits.contiguous()
    batch, n_cols = logits.shape
    out = torch.empty(batch, dtype=torch.int64, device=logits.device)
    BLOCK = 1024
    try:
        _argmax_kernel[(batch,)](logits, out, n_cols=n_cols, BLOCK=BLOCK)
    except Exception as e:
        _disable_triton(e)
        return logits.argmax(dim=-1)
    return out


def _triton_softmax_temp(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Triton-fused (logits / T) -> softmax. Single block per row."""
    if not _triton_active() or not logits.is_cuda:
        return F.softmax(logits / temperature, dim=-1)
    logits = logits.contiguous()
    batch, n_cols = logits.shape
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK > 65536:
        return F.softmax(logits / temperature, dim=-1)
    out = torch.empty_like(logits, dtype=torch.float32)
    try:
        _softmax_temp_kernel[(batch,)](
            logits, out, float(temperature), n_cols=n_cols, BLOCK=BLOCK
        )
    except Exception as e:
        _disable_triton(e)
        return F.softmax(logits / temperature, dim=-1)
    return out


class Sampler:
    """Stateless sampler that turns logits into token ids."""

    @staticmethod
    def sample(
        logits: torch.Tensor,
        sampling_params: SamplingParams,
        past_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample next tokens from logits.

        Args:
            logits: (batch_size, vocab_size)
            sampling_params: parameters controlling the sampling
            past_token_ids: (batch_size, seq_len) for repetition penalty

        Returns:
            (batch_size,) sampled token ids
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        if past_token_ids is not None and sampling_params.repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(
                logits.clone(), past_token_ids, sampling_params.repetition_penalty
            )

        if sampling_params.temperature == 0.0:
            return _triton_argmax(logits)

        # Temperature scaling + softmax fused in Triton.
        probs = _triton_softmax_temp(logits, sampling_params.temperature)

        if sampling_params.top_k > 0 or sampling_params.top_p < 1.0:
            probs = _apply_top_k_top_p(
                probs,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
            )

        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    past_token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    for i in range(logits.shape[0]):
        unique_ids = past_token_ids[i].unique()
        score = logits[i, unique_ids]
        logits[i, unique_ids] = torch.where(
            score > 0, score / penalty, score * penalty
        )
    return logits


def _apply_top_k_top_p(
    probs: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Mask the probability tensor with top-k then top-p, then renormalize."""
    if top_k > 0:
        k = min(top_k, probs.size(-1))
        topk_vals, _ = torch.topk(probs, k, dim=-1)
        threshold = topk_vals[..., -1, None]
        probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # Drop tokens whose cumulative prob exceeds top_p (always keep the top one).
        mask = cum - sorted_probs > top_p
        sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return probs
