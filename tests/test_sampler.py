"""Tests for the Sampler. Runs the PyTorch fallback path on CPU."""

from __future__ import annotations

import torch

from litevllm.config import SamplingParams
from litevllm.layers.sampler import Sampler


def test_greedy_sample_picks_argmax() -> None:
    logits = torch.tensor([
        [1.0, 5.0, 3.0, 0.5],
        [4.0, 0.0, 2.0, 1.0],
    ])
    out = Sampler.sample(logits, SamplingParams(temperature=0.0, max_tokens=8))
    assert out.tolist() == [1, 0]


def test_temperature_sampling_returns_in_vocab_range() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    out = Sampler.sample(logits, SamplingParams(temperature=1.0, max_tokens=8))
    assert out.shape == (4,)
    assert (out >= 0).all() and (out < 32).all()


def test_top_k_filters_low_probability_tokens() -> None:
    torch.manual_seed(0)
    logits = torch.zeros(1, 8)
    logits[0, 3] = 100.0
    out = Sampler.sample(
        logits,
        SamplingParams(temperature=1.0, top_k=1, max_tokens=8),
    )
    assert out.tolist() == [3]


def test_top_p_keeps_only_probable_tokens() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.5, -100.0, -100.0]])
    out = Sampler.sample(
        logits,
        SamplingParams(temperature=1.0, top_p=0.9, max_tokens=8),
    )
    assert out.tolist()[0] in (0, 1)
