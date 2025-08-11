#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a DeepSpeed ZeRO checkpoint into a Hugging Face model folder.

- Adds "<think>" and "</think>" to the tokenizer if missing.
- Resizes the model's embeddings to match the updated tokenizer.
- Loads the consolidated state dict from DS shards and saves sharded safetensors.
- Works whether --checkpoint_dir points to ckpt-XXX/ (with a 'latest' file) or directly to global_stepXXX/.

Usage:
  python scripts/convert_ds_to_hf.py \
    --checkpoint_dir output/run_.../ckpt-100 \
    --original_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir output/run_.../ckpt-100-hf \
    --dtype bf16   # or fp16/fp32 (default: fp16)
    --max_shard_size 2GB
"""
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# DeepSpeed helper to reconstruct FP32 params from ZeRO shards
try:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
except Exception as e:
    raise RuntimeError(
        "Could not import DeepSpeed's get_fp32_state_dict_from_zero_checkpoint. "
        "Please ensure DeepSpeed is installed in this environment."
    ) from e


def _find_zero_step_dir(root: Path) -> Path:
    """
    Given a DS checkpoint root (e.g., ckpt-100) or a nested path, return the
    directory that actually contains *_model_states.pt files. Accepts DS 0.13+
    naming like zero_pp_rank_*_mp_rank_*_model_states.pt.

    Preference order:
      1) If 'latest' exists, use it (global_stepX).
      2) If root itself holds *_model_states.pt, return root.
      3) Recursively search for *_model_states.pt and return its parent.
      4) Else, choose the highest global_step* that contains such files.
    """
    root = root.resolve()

    # 1) Prefer 'latest' marker if present
    latest = root / "latest"
    if latest.is_file():
        name = latest.read_text().strip()
        step_name = name if name.startswith("global_step") else f"global_step{name}"
        cand = root / step_name
        if cand.is_dir():
            return cand

    # 2) If files are directly in root
    if any(root.glob("*_model_states.pt")):
        return root

    # 3) Recursive search
    files = list(root.rglob("*_model_states.pt"))
    if files:
        return files[0].parent

    # 4) Fallback: pick highest global_step* that has files
    candidates = []
    for d in root.glob("global_step*"):
        if d.is_dir() and any(d.glob("*_model_states.pt")):
            m = re.search(r"global_step(\d+)", d.name)
            step = int(m.group(1)) if m else -1
            candidates.append((step, d))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    raise FileNotFoundError(
        f"Could not find any '*_model_states.pt' under '{root}'. "
        "Pass the explicit step dir (e.g., '.../ckpt-XXX/global_stepYYY')."
    )


def _resolve_ds_root_and_tag(checkpoint_dir: Path) -> Tuple[Path, Optional[str], Path]:
    """
    Resolve a user-supplied path into:
      - ds_root: the directory passed to DS (should contain 'latest')
      - tag:    the 'global_stepX' to load (or None to use 'latest')
      - step_dir: the fully qualified 'global_stepX' directory we resolved (for logging)

    Rules:
      - If user passed a global_stepX dir: ds_root = parent, tag = global_stepX.
      - Else if ckpt root has 'latest': ds_root = root, tag = name in 'latest'.
      - Else: fall back to discovered step_dir; if it's global_stepX, use parent+tag.
    """
    checkpoint_dir = checkpoint_dir.resolve()

    # Case A: user passed a global_stepX directly
    if checkpoint_dir.name.startswith("global_step") and checkpoint_dir.is_dir():
        return checkpoint_dir.parent, checkpoint_dir.name, checkpoint_dir

    # Case B: user passed ckpt-XXX root (typical)
    latest = checkpoint_dir / "latest"
    if latest.is_file():
        name = latest.read_text().strip()
        tag = name if name.startswith("global_step") else f"global_step{name}"
        step_dir = checkpoint_dir / tag
        if not step_dir.is_dir():
            # Fall back to probing
            step_dir = _find_zero_step_dir(checkpoint_dir)
            # Derive tag if possible
            tag = step_dir.name if step_dir.name.startswith("global_step") else None
        return checkpoint_dir, tag, step_dir

    # Case C: no 'latest' – probe to find a step dir
    step_dir = _find_zero_step_dir(checkpoint_dir)
    if step_dir.name.startswith("global_step"):
        return step_dir.parent, step_dir.name, step_dir
    # If shards are directly in 'checkpoint_dir' (rare), DS may accept it without tag
    return step_dir, None, step_dir


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
    ds_root, tag, step_dir = _resolve_ds_root_and_tag(ckpt_root)

    print(f"[convert] Using DS step dir: {step_dir}")
    print(
        "[convert] Loading consolidated state dict from DeepSpeed shards ... (this can take a while)"
    )

    # Single DS call: pass (root, tag) so it never looks for 'latest' in global_stepX
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(ds_root), tag=tag)
    state_dict = _strip_module_prefix(state_dict)

    # Choose dtype for saving
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    save_dtype = dtype_map[args.dtype]

    # Optional downcast to reduce peak RAM before building the model
    if save_dtype != torch.float32:
        for k in list(state_dict.keys()):
            t = state_dict[k]
            if torch.is_floating_point(t):
                state_dict[k] = t.to(save_dtype)

    # Load tokenizer & add reasoning tokens if missing
    tok = AutoTokenizer.from_pretrained(args.original_model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    added = 0
    for tkn in ["<think>", "</think>"]:
        if tkn not in tok.get_vocab():
            tok.add_tokens([tkn])
            added += 1

    print(f"[convert] Instantiating base model: {args.original_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.original_model,
        torch_dtype=save_dtype,
        low_cpu_mem_usage=False,
        device_map="cpu",
    )

    if added > 0 and len(tok) != model.config.vocab_size:
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
