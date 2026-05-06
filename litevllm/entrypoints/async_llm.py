"""AsyncLLM — asynchronous streaming inference entrypoint.

Usage:
    from litevllm import AsyncLLM, SamplingParams
    from litevllm.config import EngineConfig, ModelConfig

    engine = AsyncLLM(EngineConfig(model_config=ModelConfig(model="Qwen/Qwen3-0.6B")))
    await engine.initialize()

    async for delta in engine.stream("Hello, world!", SamplingParams()):
        print(delta, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Optional

from litevllm.config import EngineConfig, SamplingParams
from litevllm.engine.llm_engine import LLMEngine
from litevllm.engine.sequence import SequenceGroupOutput


class AsyncLLM:
    """Async wrapper around LLMEngine that surfaces per-token text deltas."""

    def __init__(self, engine_config: EngineConfig) -> None:
        self.config = engine_config
        self._engine: Optional[LLMEngine] = None
        self._request_streams: dict[str, asyncio.Queue] = {}
        self._step_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize the engine (blocks while loading model)."""
        self._engine = LLMEngine(self.config)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._engine.initialize)

    async def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> str:
        if request_id is None:
            request_id = str(uuid.uuid4())
        self._request_streams[request_id] = asyncio.Queue()
        self._engine.add_request(prompt, sampling_params, request_id)
        self._ensure_step_loop()
        return request_id

    async def stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream output text deltas (one chunk per decoded token) for a prompt.

        A trailing sentinel marks the end of the stream.
        """
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

        request_id = await self.add_request(prompt, sampling_params, request_id)
        queue = self._request_streams[request_id]

        try:
            while True:
                item = await queue.get()
                if item is None:  # end-of-stream sentinel
                    break
                yield item
        finally:
            self._request_streams.pop(request_id, None)

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """Non-streaming convenience: collect the full output text."""
        chunks: list[str] = []
        async for delta in self.stream(prompt, sampling_params):
            chunks.append(delta)
        return "".join(chunks)

    def _ensure_step_loop(self) -> None:
        if self._running:
            return
        self._running = True
        self._step_task = asyncio.ensure_future(self._step_loop())

    async def _step_loop(self) -> None:
        """Background loop that drives engine steps and fans out deltas."""
        loop = asyncio.get_event_loop()
        try:
            while self._engine.has_unfinished():
                outputs: list[SequenceGroupOutput] = await loop.run_in_executor(
                    None, self._engine.step
                )
                for out in outputs:
                    queue = self._request_streams.get(out.request_id)
                    if queue is None:
                        continue
                    if out.delta_text:
                        await queue.put(out.delta_text)
                    if out.finished:
                        await queue.put(None)  # end-of-stream sentinel
                # Yield to other coroutines so the consumer can pull tokens.
                await asyncio.sleep(0)
        finally:
            self._running = False
            self._step_task = None

    async def abort(self, request_id: str) -> None:
        self._engine.scheduler.abort_request(request_id)
        queue = self._request_streams.pop(request_id, None)
        if queue is not None:
            await queue.put(None)

    @property
    def has_unfinished(self) -> bool:
        return self._engine is not None and self._engine.has_unfinished()
