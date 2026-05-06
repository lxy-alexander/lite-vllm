#!/usr/bin/env python3
"""Shared benchmark helpers for experiment scripts in examples/."""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import threading
import time
from pathlib import Path

from litevllm.config import CacheConfig, EngineConfig, ModelConfig, SamplingParams, SchedulerConfig
from litevllm.engine.llm_engine import LLMEngine


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    idx = (len(ys) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ys[lo]
    w = idx - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _long_prompt(seed: int, repeat: int = 256) -> str:
    random.seed(seed)
    vocab = ["throughput", "latency", "prefill", "decode", "cache", "scheduler", "gpu", "token"]
    return " ".join(random.choice(vocab) for _ in range(repeat))


def make_prompts(num_requests: int, mode: str) -> list[str]:
    base = [
        "Explain continuous batching briefly.",
        "What is TTFT in LLM serving?",
        "Summarize PagedAttention in three points.",
        "How does chunked prefill reduce latency spikes?",
    ]
    if mode == "short":
        return [base[i % len(base)] for i in range(num_requests)]
    if mode == "long":
        return [_long_prompt(i, 400) for i in range(num_requests)]
    prompts: list[str] = []
    for i in range(num_requests):
        prompts.append(base[i % len(base)] if i % 2 == 0 else _long_prompt(i, 300))
    return prompts


class _GPUSampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            t = time.time()
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                util_s, used_s, total_s = [x.strip() for x in out.splitlines()[0].split(",")]
                self.samples.append(
                    {
                        "t": t,
                        "gpu_util": float(util_s),
                        "mem_used_mb": float(used_s),
                        "mem_total_mb": float(total_s),
                    }
                )
            except Exception:
                pass
            time.sleep(self.interval_s)


def run_one_experiment(
    *,
    model: str,
    dtype: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    chunked_prefill: bool,
    chunk_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    block_size: int,
    gpu_memory_utilization: float,
    gpu_sample_interval: float,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EngineConfig(
        model_config=ModelConfig(model=model, dtype=dtype),
        cache_config=CacheConfig(block_size=block_size, gpu_memory_utilization=gpu_memory_utilization),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            chunked_prefill=chunked_prefill,
            chunk_size=chunk_size,
        ),
        seed=0,
    )
    engine = LLMEngine(cfg)
    engine.initialize()

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    req_ids: list[str] = []
    submit_t = time.time()
    first_run_t: dict[str, float] = {}
    first_tok_t: dict[str, float] = {}
    finish_t: dict[str, float] = {}
    prev_out: dict[str, int] = {}
    for p in prompts:
        rid = engine.add_request(p, sampling)
        req_ids.append(rid)
        prev_out[rid] = 0

    gpu_sampler = _GPUSampler(gpu_sample_interval)
    gpu_sampler.start()

    step_rows: list[dict] = []
    tpot_step_values: list[float] = []
    internal_waste_peak_ratio = 0.0
    empty_step_streak = 0
    max_empty_steps = 200
    t0 = time.time()
    try:
        while engine.has_unfinished():
            st = time.time()
            outs = engine.step()
            et = time.time()

            running_ids = {sg.request_id for sg in engine.scheduler.running}
            waiting_ids = {sg.request_id for sg in engine.scheduler.waiting}
            for rid in running_ids:
                first_run_t.setdefault(rid, et)

            delta_tokens = 0
            for out in outs:
                rid = out.request_id
                curr = out.num_output_tokens
                delta = max(0, curr - prev_out[rid])
                prev_out[rid] = curr
                if delta > 0:
                    first_tok_t.setdefault(rid, et)
                    delta_tokens += delta
                if out.finished:
                    finish_t[rid] = et

            if delta_tokens > 0:
                tpot_step_values.append((et - st) / delta_tokens)
                empty_step_streak = 0
            else:
                empty_step_streak += 1
                if empty_step_streak >= max_empty_steps:
                    raise RuntimeError(
                        "Benchmark made no token-level progress for too many scheduler steps. "
                        "Try reducing --num-requests / --max-tokens, lowering "
                        "--gpu-memory-utilization pressure, or shortening prompts."
                    )

            bm = engine.block_manager
            total_blocks = bm.gpu_allocator.num_blocks
            active_blocks = total_blocks - bm.num_free_gpu_blocks
            active_seqs = sum(len(sg.get_unfinished_seqs()) for sg in engine.scheduler.running)
            idle_slot_ratio = (
                (engine.scheduler.config.max_num_seqs - active_seqs) / engine.scheduler.config.max_num_seqs
                if engine.scheduler.config.max_num_seqs > 0
                else 0.0
            )

            waste_slots = 0
            for sg in engine.scheduler.running:
                for seq in sg.get_unfinished_seqs():
                    table = bm.block_tables.get(seq.seq_id)
                    alloc_slots = (len(table.blocks) if table is not None else 0) * bm.block_size
                    waste_slots += max(0, alloc_slots - seq.data.total_len)
            if active_blocks > 0:
                internal_waste_peak_ratio = max(
                    internal_waste_peak_ratio,
                    waste_slots / (active_blocks * bm.block_size),
                )

            step_rows.append(
                {
                    "t": et,
                    "running_seq_groups": len(engine.scheduler.running),
                    "waiting_seq_groups": len(waiting_ids),
                    "active_seqs": active_seqs,
                    "idle_slot_ratio": idle_slot_ratio,
                    "active_blocks": active_blocks,
                    "total_blocks": total_blocks,
                    "block_utilization": (active_blocks / total_blocks) if total_blocks > 0 else 0.0,
                    "delta_output_tokens_step": delta_tokens,
                }
            )
    finally:
        gpu_sampler.stop()

    elapsed = time.time() - t0

    total_output_tokens = sum(prev_out.values())
    total_prompt_tokens = sum(len(engine.tokenizer.encode(p)) for p in prompts)

    ttft = [first_tok_t[rid] - submit_t for rid in req_ids if rid in first_tok_t]
    queue_wait = [first_run_t[rid] - submit_t for rid in req_ids if rid in first_run_t]
    tpot_req = [
        (finish_t[rid] - first_tok_t[rid]) / (prev_out[rid] - 1)
        for rid in req_ids
        if rid in finish_t and rid in first_tok_t and prev_out[rid] > 1
    ]

    throughput = {
        "request_throughput_req_per_s": (len(req_ids) / elapsed) if elapsed > 0 else 0.0,
        "output_token_throughput_tok_per_s": (total_output_tokens / elapsed) if elapsed > 0 else 0.0,
        "total_token_throughput_tok_per_s": ((total_prompt_tokens + total_output_tokens) / elapsed) if elapsed > 0 else 0.0,
        "num_requests": len(req_ids),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
    }
    latency = {
        "ttft_p99_s": _pct(ttft, 0.99),
        "tpot_req_mean_s": _mean(tpot_req),
        "tpot_step_mean_s": _mean(tpot_step_values),
        "tpot_step_std_s": _pct(tpot_step_values, 0.84) - _pct(tpot_step_values, 0.16),
        "tpot_step_p99_s": _pct(tpot_step_values, 0.99),
    }
    batching = {
        "active_sequences_mean": _mean([r["active_seqs"] for r in step_rows]),
        "active_sequences_p95": _pct([r["active_seqs"] for r in step_rows], 0.95),
        "idle_slot_ratio_mean": _mean([r["idle_slot_ratio"] for r in step_rows]),
        "idle_slot_ratio_p95": _pct([r["idle_slot_ratio"] for r in step_rows], 0.95),
        "queue_wait_p99_s": _pct(queue_wait, 0.99),
    }
    kv = {
        "total_blocks": engine.block_manager.gpu_allocator.num_blocks,
        "peak_block_utilization": max([r["block_utilization"] for r in step_rows], default=0.0),
        "avg_block_utilization": _mean([r["block_utilization"] for r in step_rows]),
    }
    fragmentation = {
        "external_fragmentation_rate": 0.0,
        "internal_fragmentation_peak_ratio": internal_waste_peak_ratio,
    }
    gpu = {
        "sm_util_mean": _mean([s["gpu_util"] for s in gpu_sampler.samples]),
        "sm_util_p99": _pct([s["gpu_util"] for s in gpu_sampler.samples], 0.99),
    }

    _write_json(out_dir / "throughput.json", throughput)
    _write_json(out_dir / "latency.json", latency)
    _write_json(out_dir / "continuous_batching.json", batching)
    _write_json(out_dir / "kv_cache.json", kv)
    _write_json(out_dir / "pagedattention_fragmentation.json", fragmentation)
    _write_json(out_dir / "gpu_utilization.json", gpu)
    _write_csv(out_dir / "step_stats.csv", step_rows)
    _write_csv(out_dir / "gpu_samples.csv", gpu_sampler.samples)

    return {
        "throughput": throughput,
        "latency": latency,
        "batching": batching,
        "kv": kv,
        "fragmentation": fragmentation,
        "gpu": gpu,
    }

