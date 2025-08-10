#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a DeepSpeed ZeRO checkpoint into a Hugging Face model folder.

- Adds "<think>" and "</think>" to the tokenizer if missing.
- Resizes the model's embeddings to match the updated tokenizer.
- Loads the consolidated state dict from DS shards and saves sharded safetensors.
- Works even if `checkpoint_dir` points to the parent folder (auto-detects 'global_step*').

Usage:
  python scripts/convert_ds_to_hf.py \
    --checkpoint_dir output/run_.../ckpt-100 \
    --original_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir output/run_.../ckpt-100-hf \
    --dtype bf16   # or fp16/fp32 (default: fp16)
    --max_shard_size 2GB
"""
import argparse, re, os, sys
from pathlib import Path
from typing import Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import the canonical helper
try:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
except Exception as e:
    raise RuntimeError(
        "Could not import DeepSpeed's get_fp32_state_dict_from_zero_checkpoint. "
        "Please ensure DeepSpeed is installed in this environment."
    ) from e


def _find_zero_step_dir(root: Path) -> Path:
    """
    Given a DS checkpoint root (e.g., ckpt-100), return the directory that contains
    mp_rank_*_model_states.pt files. Handles both:
      - ckpt-100/global_step100/...
      - ckpt-100/...
    If multiple global_step* dirs exist, pick the one with the largest trailing int.
    """
    root = root.resolve()
    # Case 1: files are directly in root
    if any(
        p.name.startswith("mp_rank_") and p.name.endswith("_model_states.pt")
        for p in root.glob("**/*")
    ):
        # Check if they are at root level; if not, try to find the deepest containing directory
        direct = list(root.glob("mp_rank_*_model_states.pt"))
        if direct:
            return root

    # Case 2: search for global_step*
    candidates = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("global_step"):
            has_files = any(
                p.name.startswith("mp_rank_") and p.name.endswith("_model_states.pt")
                for p in d.glob("mp_rank_*_model_states.pt")
            )
            if has_files:
                m = re.search(r"global_step(\d+)", d.name)
                step = int(m.group(1)) if m else -1
                candidates.append((step, d))
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a 'global_step*' directory with mp_rank_*_model_states.pt inside '{root}'."
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint_dir",
        required=True,
        type=str,
        help="Path to DS checkpoint (ckpt-XXX or global_stepXXX)",
    )
    ap.add_argument(
        "--original_model",
        required=True,
        type=str,
        help="Base HF model name or path (architecture & config)",
    )
    ap.add_argument(
        "--output_dir", required=True, type=str, help="Where to write the HF model"
    )
    ap.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Save weights dtype",
    )
    ap.add_argument(
        "--max_shard_size",
        default="2GB",
        type=str,
        help="Max shard size for safetensors",
    )
    args = ap.parse_args()

    ckpt_root = Path(args.checkpoint_dir)
    step_dir = _find_zero_step_dir(ckpt_root)

    # Load tokenizer from the base model and add reasoning tokens if missing
    tok = AutoTokenizer.from_pretrained(args.original_model)
    # ensure pad_token exists (helps for inference)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    added = 0
    for t in ["<think>", "</think>"]:
        if t not in tok.get_vocab():
            tok.add_tokens([t])
            added += 1

    # Decide dtype
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    save_dtype = dtype_map[args.dtype]

    print(f"[convert] Using DS step dir: {step_dir}")
    print(
        f"[convert] Loading consolidated state dict from DeepSpeed shards ... (this can take a while)"
    )
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(step_dir))
    state_dict = _strip_module_prefix(state_dict)

    # Optionally downcast before constructing the model (reduces peak RAM)
    if save_dtype != torch.float32:
        for k in list(state_dict.keys()):
            t = state_dict[k]
            if torch.is_floating_point(t):
                state_dict[k] = t.to(save_dtype)

    print(f"[convert] Instantiating base model: {args.original_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.original_model,
        torch_dtype=save_dtype,
        low_cpu_mem_usage=False,  # keep explicit to avoid lazy materialization surprises
        device_map="cpu",
    )

    # If we added tokens, make sure embeddings match
    if len(tok) != model.config.vocab_size:
        print(
            f"[convert] Resizing token embeddings: {model.config.vocab_size} → {len(tok)}"
        )
        model.resize_token_embeddings(len(tok))

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"[convert][warn] Missing keys: {len(missing)} (showing up to 10) -> {missing[:10]}"
        )
    if unexpected:
        print(
            f"[convert][warn] Unexpected keys: {len(unexpected)} (showing up to 10) -> {unexpected[:10]}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] Saving HF model to: {out_dir}")
    model.save_pretrained(
        str(out_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tok.save_pretrained(str(out_dir))
    print("[convert] Done.")


if __name__ == "__main__":
    main()
