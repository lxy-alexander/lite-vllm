"""Request state machine: waiting → decoding → done.

Each user request becomes a ``SequenceGroup`` containing one or more
``Sequence`` objects (for beam search / best-of-n).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional

from litevllm.config import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH = "finished_length"
    FINISHED_EOS = "finished_eos"
    FINISHED_ABORT = "finished_abort"

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_EOS,
            SequenceStatus.FINISHED_ABORT,
        )


@dataclass
class SequenceData:
    """Token-level data for a single sequence."""

    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)


_next_seq_id = 0


def _gen_seq_id() -> int:
    global _next_seq_id
    _next_seq_id += 1
    return _next_seq_id


@dataclass
class Sequence:
    """A single sequence being generated."""

    seq_id: int
    data: SequenceData
    status: SequenceStatus = SequenceStatus.WAITING

    # Scheduling metadata
    num_computed_tokens: int = 0  # prefill progress (for chunked prefill)

    # Streaming detokenization: text already emitted as deltas. Used to compute
    # the next ``delta_text`` as ``new_full_text[len(emitted_text):]``, which
    # correctly handles BPE / SentencePiece partial-byte boundaries.
    emitted_text: str = ""

    @classmethod
    def create(cls, prompt_token_ids: list[int]) -> "Sequence":
        return cls(
            seq_id=_gen_seq_id(),
            data=SequenceData(prompt_token_ids=list(prompt_token_ids)),
        )

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.data.prompt_len

    @property
    def num_new_tokens(self) -> int:
        """Tokens to process in the next step."""
        if self.is_prefill:
            return self.data.prompt_len - self.num_computed_tokens
        return 1  # decode: one token at a time

    def get_num_blocks(self, block_size: int) -> int:
        return (self.data.total_len + block_size - 1) // block_size


@dataclass
class SequenceGroup:
    """A group of sequences sharing the same prompt (beam search / best-of-n)."""

    request_id: str
    seqs: list[Sequence]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.time)
    prompt_text: str = ""

    @classmethod
    def create(
        cls,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        prompt_text: str = "",
    ) -> "SequenceGroup":
        seq = Sequence.create(prompt_token_ids)
        return cls(
            request_id=request_id,
            seqs=[seq],
            sampling_params=sampling_params,
            prompt_text=prompt_text,
        )

    @property
    def is_finished(self) -> bool:
        return all(s.status.is_finished for s in self.seqs)

    def get_unfinished_seqs(self) -> list[Sequence]:
        return [s for s in self.seqs if not s.status.is_finished]


@dataclass
class SequenceGroupOutput:
    """Per-step result for a SequenceGroup.

    Used for both streaming (``finished=False``, ``delta_text`` is the text
    just produced this step) and final outputs (``finished=True``).
    """

    request_id: str
    prompt_text: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_text: str
    finish_reason: str
    delta_text: str = ""
    finished: bool = True

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)
