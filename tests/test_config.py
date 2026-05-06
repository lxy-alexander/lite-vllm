"""Tests for SamplingParams validation."""

from __future__ import annotations

import pytest

from litevllm.config import SamplingParams


def test_defaults_are_valid() -> None:
    sp = SamplingParams()
    assert sp.temperature == 1.0
    assert sp.top_p == 1.0
    assert sp.max_tokens == 256


def test_negative_temperature_rejected() -> None:
    with pytest.raises(ValueError):
        SamplingParams(temperature=-0.1)


def test_top_p_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)
