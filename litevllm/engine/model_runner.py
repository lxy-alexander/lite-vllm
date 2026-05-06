"""Model runner: orchestrates model forward pass.

Responsibilities:
- Load model weights (safetensors or .bin)
- Profile available KV cache memory
- Pack scheduled sequences into flat input tensors
- Drive a single prefill or decode forward pass
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ..cache.kv_cache import KVCache
from ..config import EngineConfig
from ..layers.sampler import Sampler
from ..models.base import BaseModelForCausalLM
from .sequence import SequenceGroup


def _resolve_model_dir(model_name_or_path: str) -> str:
    """Resolve a HF model name or local path to a directory with weight files."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name_or_path)
    except Exception:
        pass
    raise FileNotFoundError(
        f"Cannot find model weights at {model_name_or_path}. "
        f"Provide a local directory or valid HuggingFace model name."
    )


@torch.inference_mode()
def _load_model_weights(
    model: BaseModelForCausalLM,
    model_name_or_path: str,
    dtype: torch.dtype,
) -> None:
    """Load weights from safetensors or pytorch checkpoints."""
    from glob import glob

    model_dir = _resolve_model_dir(model_name_or_path)

    st_files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    if st_files:
        from safetensors import safe_open

        weights: dict[str, torch.Tensor] = {}
        for f in st_files:
            # Load on CPU then cast; loading directly on CUDA can hit
            # "no kernel image" on wheels missing kernels for the local arch.
            with safe_open(f, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    weights[key] = sf.get_tensor(key).to(dtype)
        model.load_weights(weights)
        print(f"[litevllm] Loaded {len(weights)} tensors from "
              f"{len(st_files)} safetensors file(s)")
        return

    pt_files = sorted(glob(os.path.join(model_dir, "*.bin")))
    if pt_files:
        weights = {}
        for f in pt_files:
            w = torch.load(f, map_location="cpu", weights_only=True)
            for k, v in w.items():
                weights[k] = v.to(dtype)
        model.load_weights(weights)
        print(f"[litevllm] Loaded {len(weights)} tensors from "
              f"{len(pt_files)} bin file(s)")
        return

    raise FileNotFoundError(
        f"No safetensors or pytorch checkpoint found in {model_dir}"
    )


class InputMetadata:
    """Packed input tensors for a single forward step."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_tables: Optional[torch.Tensor],
        context_lens: Optional[torch.Tensor],
        is_prefill: bool,
        seq_ids: list[int],
    ) -> None:
        self.input_ids = input_ids
        self.positions = positions
        self.slot_mapping = slot_mapping
        self.block_tables = block_tables
        self.context_lens = context_lens
        self.is_prefill = is_prefill
        self.seq_ids = seq_ids


class ModelRunner:
    """Loads the model, prepares inputs, and runs forward passes."""

    def __init__(self, engine_config: EngineConfig) -> None:
        self.config = engine_config
        self.model_config = engine_config.model_config
        self.device = engine_config.device

        self.model: Optional[BaseModelForCausalLM] = None
        self.kv_cache: Optional[KVCache] = None
        self.sampler = Sampler()

    def load_model(self) -> None:
        model_cls = BaseModelForCausalLM.get_model_cls(self.model_config.model_type)
        self.model = model_cls(self.model_config)
        self.model = self.model.to(
            dtype=self.model_config.torch_dtype, device=self.device
        )
        _load_model_weights(
            self.model,
            self.model_config.model,
            self.model_config.torch_dtype,
        )
        self.model.eval()

    def init_kv_cache(self, num_gpu_blocks: int) -> None:
        self.kv_cache = KVCache(
            num_layers=self.model_config.num_hidden_layers,
            num_blocks=num_gpu_blocks,
            block_size=self.config.cache_config.block_size,
            num_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            dtype=self.model_config.torch_dtype,
            device=self.device,
        )

    def profile_num_gpu_blocks(self) -> int:
        """Profile available GPU memory and compute how many KV blocks fit."""
        fallback = self.config.cache_config.num_gpu_blocks or 256

        if not torch.cuda.is_available() or self.device == "cpu":
            return fallback

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            dummy_ids = torch.zeros(1, dtype=torch.long, device=self.device)
            dummy_pos = torch.zeros(1, dtype=torch.long, device=self.device)
            with torch.no_grad():
                self.model(dummy_ids, dummy_pos, is_prefill=True)

            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            total_memory = torch.cuda.get_device_properties(
                torch.device(self.device)
            ).total_memory
            available = int(
                total_memory * self.config.cache_config.gpu_memory_utilization
            ) - peak_memory

            num_blocks = KVCache.profile_num_blocks(
                available,
                self.model_config.num_hidden_layers,
                self.config.cache_config.block_size,
                self.model_config.num_key_value_heads,
                self.model_config.head_dim,
                self.model_config.torch_dtype,
            )
            return max(num_blocks, 16)
        except Exception as e:
            print(f"[litevllm] CUDA profiling failed ({e}), using {fallback} blocks")
            return fallback

    def prepare_inputs(
        self,
        seq_groups: list[SequenceGroup],
        block_manager,
    ) -> InputMetadata:
        """Flatten scheduled sequences into packed tensors."""
        input_ids_list: list[int] = []
        positions_list: list[int] = []
        slot_mapping_list: list[int] = []
        block_tables_list: list[list[int]] = []
        context_lens_list: list[int] = []
        seq_ids: list[int] = []
        is_prefill = False

        block_size = self.config.cache_config.block_size

        for sg in seq_groups:
            for seq in sg.get_unfinished_seqs():
                seq_ids.append(seq.seq_id)
                bt = block_manager.get_block_table(seq.seq_id)

                if seq.is_prefill:
                    is_prefill = True
                    start = seq.num_computed_tokens
                    end = seq.data.prompt_len
                    tokens = seq.data.prompt_token_ids[start:end]
                    for i, tok in enumerate(tokens):
                        pos = start + i
                        input_ids_list.append(tok)
                        positions_list.append(pos)
                        block_idx = pos // block_size
                        block_offset = pos % block_size
                        if block_idx < len(bt):
                            slot = bt.blocks[block_idx].block_id * block_size + block_offset
                        else:
                            slot = 0
                        slot_mapping_list.append(slot)
                else:
                    token = seq.data.all_token_ids[-1]
                    pos = seq.data.total_len - 1
                    input_ids_list.append(token)
                    positions_list.append(pos)

                    block_idx = pos // block_size
                    block_offset = pos % block_size
                    if block_idx < len(bt):
                        slot = bt.blocks[block_idx].block_id * block_size + block_offset
                    else:
                        slot = 0
                    slot_mapping_list.append(slot)

                    context_lens_list.append(seq.data.total_len)
                    block_tables_list.append(bt.physical_block_ids)

        device = self.device
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        positions = torch.tensor(positions_list, dtype=torch.long, device=device)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=device)

        if is_prefill:
            block_tables = None
            context_lens = None
        else:
            max_blocks = (
                max(len(bt) for bt in block_tables_list) if block_tables_list else 0
            )
            padded = [bt + [0] * (max_blocks - len(bt)) for bt in block_tables_list]
            block_tables = torch.tensor(padded, dtype=torch.int32, device=device)
            context_lens = torch.tensor(
                context_lens_list, dtype=torch.int32, device=device
            )

        return InputMetadata(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            is_prefill=is_prefill,
            seq_ids=seq_ids,
        )

    @torch.inference_mode()
    def execute_model(self, metadata: InputMetadata) -> torch.Tensor:
        """Run the model forward pass and return logits."""
        kv_caches = None
        if self.kv_cache is not None:
            kv_caches = [
                self.kv_cache.get_kv(i)
                for i in range(self.model_config.num_hidden_layers)
            ]

        return self.model(
            input_ids=metadata.input_ids,
            positions=metadata.positions,
            kv_caches=kv_caches,
            slot_mapping=metadata.slot_mapping,
            block_tables=metadata.block_tables,
            context_lens=metadata.context_lens,
            is_prefill=metadata.is_prefill,
        )
