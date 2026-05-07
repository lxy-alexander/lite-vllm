"""Main LLM Engine — drives the scheduling + model execution loop.

Ties together the Scheduler, ModelRunner, BlockManager, and Tokenizer into
a single step-based execution loop.
"""

from __future__ import annotations

import uuid
from typing import Optional

import torch

from ..cache.block_manager import BlockManager
from ..cache.prefix_cache import PrefixCache
from ..config import EngineConfig, SamplingParams
from ..layers.sampler import Sampler
from ..tokenizer import get_tokenizer, TokenizerGroup
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import (
    Sequence,
    SequenceGroup,
    SequenceGroupOutput,
    SequenceStatus,
)


class LLMEngine:
    """Core engine: adds requests, runs steps, returns completions."""

    def __init__(self, engine_config: EngineConfig) -> None:
        self.config = engine_config

        self.tokenizer: Optional[TokenizerGroup] = None
        self.model_runner: Optional[ModelRunner] = None
        self.scheduler: Optional[Scheduler] = None
        self.block_manager: Optional[BlockManager] = None
        self.prefix_cache: Optional[PrefixCache] = None
        self.sampler = Sampler()

    def initialize(self) -> None:
        """Resolve config, load tokenizer + model, allocate KV cache."""
        self.config.resolve()
        mc = self.config.model_config

        print(f"[litevllm] model_type={mc.model_type}, "
              f"layers={mc.num_hidden_layers}, "
              f"heads={mc.num_attention_heads}/{mc.num_key_value_heads}, "
              f"hidden={mc.hidden_size}")

        self.tokenizer = get_tokenizer(mc.model, mc.trust_remote_code, mc.revision)

        self.model_runner = ModelRunner(self.config)
        self.model_runner.load_model()

        cc = self.config.cache_config
        if cc.num_gpu_blocks is None:
            cc.num_gpu_blocks = self.model_runner.profile_num_gpu_blocks()
        print(f"[litevllm] KV cache: {cc.num_gpu_blocks} blocks "
              f"(block_size={cc.block_size})")

        self.model_runner.init_kv_cache(cc.num_gpu_blocks)

        self.block_manager = BlockManager(
            block_size=cc.block_size,
            num_gpu_blocks=cc.num_gpu_blocks,
            num_cpu_blocks=cc.num_cpu_blocks,
        )

        if cc.enable_prefix_caching:
            self.prefix_cache = PrefixCache(cc.block_size)

        self.scheduler = Scheduler(
            self.config.scheduler_config,
            cc,
            self.block_manager,
        )
        print("[litevllm] Engine ready.")

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> str:
        """Tokenize and enqueue a new generation request."""
        if request_id is None:
            request_id = str(uuid.uuid4())

        token_ids = self.tokenizer.encode(prompt)

        sg = SequenceGroup.create(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            prompt_text=prompt,
        )
        self.scheduler.add_request(sg)
        return request_id

    def step(self) -> list[SequenceGroupOutput]:
        """Run one scheduling + forward + sampling step.

        Returns one ``SequenceGroupOutput`` per active sequence group that
        produced a new token this step. ``finished=False`` for in-flight
        outputs (with ``delta_text`` carrying the text just decoded);
        ``finished=True`` for the final emission of a completed request.
        """
        sched_output = self.scheduler.schedule()
        if sched_output.is_empty:
            return []

        prefill_groups = [
            sg for sg in sched_output.scheduled_seq_groups
            if any(s.is_prefill for s in sg.get_unfinished_seqs())
        ]
        decode_groups = [
            sg for sg in sched_output.scheduled_seq_groups
            if not any(s.is_prefill for s in sg.get_unfinished_seqs())
        ]

        # Decode first so in-flight sequences advance every step even when this
        # batch also includes new or ongoing prefills (chunked or not). Running
        # all prefills before decode would serialize long prefills ahead of every
        # decode batch and largely removes chunked prefill's TTFT benefit.
        if decode_groups:
            metadata = self.model_runner.prepare_inputs(
                decode_groups, self.block_manager
            )
            logits = self.model_runner.execute_model(metadata)
            self._process_outputs(logits, decode_groups)

        # Prefill: one forward pass per group to avoid cross-sequence attention.
        for sg in prefill_groups:
            metadata = self.model_runner.prepare_inputs([sg], self.block_manager)
            logits = self.model_runner.execute_model(metadata)
            self._process_outputs(logits, [sg])

        # Build per-step outputs (with delta_text) before freeing finished
        # groups, so we can mark the last emission as finished=True.
        results = [
            self._build_step_output(sg)
            for sg in sched_output.scheduled_seq_groups
        ]

        self.scheduler.update_running(sched_output.scheduled_seq_groups)
        self.scheduler.free_finished()
        return results

    def _build_step_output(self, sg: SequenceGroup) -> SequenceGroupOutput:
        """Decode the seq group's running text and compute the delta vs. last step."""
        seq = sg.seqs[0]
        skip_special = sg.sampling_params.skip_special_tokens
        full_text = self.tokenizer.decode(
            seq.data.output_token_ids,
            skip_special_tokens=skip_special,
        )
        delta = full_text[len(seq.emitted_text):]
        seq.emitted_text = full_text
        return SequenceGroupOutput(
            request_id=sg.request_id,
            prompt_text=sg.prompt_text,
            prompt_token_ids=seq.data.prompt_token_ids,
            output_token_ids=list(seq.data.output_token_ids),
            output_text=full_text,
            finish_reason=seq.status.value,
            delta_text=delta,
            finished=sg.is_finished,
        )

    def _process_outputs(
        self,
        logits: torch.Tensor,
        seq_groups: list[SequenceGroup],
    ) -> None:
        """Sample tokens and update sequence state.

        For prefill: logits has one row per input token; we pick the last token
        of each sequence for sampling. For decode: one logit row per sequence.
        """
        idx = 0
        for sg in seq_groups:
            for seq in sg.get_unfinished_seqs():
                if seq.is_prefill:
                    num_tokens = seq.data.prompt_len - seq.num_computed_tokens
                    seq.num_computed_tokens = seq.data.prompt_len
                    seq_logits = logits[idx + num_tokens - 1:idx + num_tokens]
                    idx += num_tokens
                else:
                    seq_logits = logits[idx:idx + 1]
                    idx += 1

                if seq_logits.dim() == 1:
                    seq_logits = seq_logits.unsqueeze(0)

                token_id = self.sampler.sample(seq_logits, sg.sampling_params).item()
                seq.data.append_token(token_id)
                self._should_stop(seq, sg.sampling_params, token_id)

    def _should_stop(
        self,
        seq: Sequence,
        params: SamplingParams,
        token_id: int,
    ) -> bool:
        """Check stopping conditions and update sequence status."""
        if token_id == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED_EOS
            return True

        if token_id in params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return True

        if seq.data.output_len >= params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH
            return True

        if params.stop:
            output_text = self.tokenizer.decode(
                seq.data.output_token_ids, skip_special_tokens=False
            )
            for stop_str in params.stop:
                if stop_str in output_text:
                    seq.status = SequenceStatus.FINISHED_STOPPED
                    return True

        return False

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def run_to_completion(self) -> list[SequenceGroupOutput]:
        """Run the engine until all requests are finished.

        Returns the final ``finished=True`` output of each request, in the
        order requests completed.
        """
        finished: list[SequenceGroupOutput] = []
        while self.has_unfinished():
            for out in self.step():
                if out.finished:
                    finished.append(out)
        return finished
