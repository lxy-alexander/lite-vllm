"""AutoTokenizer multiprocess-safe wrapper."""

from __future__ import annotations

import os
from typing import Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


class TokenizerGroup:
    """Process-safe tokenizer pool.

    In multi-worker setups each process loads its own copy; the main process
    holds a reference that is shared via the ``get_tokenizer`` helper to avoid
    repeated disk I/O in the single-process case.
    """

    _instances: dict[str, PreTrainedTokenizer | PreTrainedTokenizerFast] = {}

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self._tokenizer = self._load()

    def _load(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        key = self.model_name_or_path
        if key in self._instances:
            return self._instances[key]
        tok = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self._instances[key] = tok
        return tok

    @property
    def tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        tid = self._tokenizer.eos_token_id
        return tid if tid is not None else 0

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def encode(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
    ) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return [
            self._tokenizer.encode(t, add_special_tokens=add_special_tokens)
            for t in text
        ]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        token_ids_batch: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self._tokenizer.batch_decode(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )


def get_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    revision: Optional[str] = None,
) -> TokenizerGroup:
    return TokenizerGroup(model_name_or_path, trust_remote_code, revision)
