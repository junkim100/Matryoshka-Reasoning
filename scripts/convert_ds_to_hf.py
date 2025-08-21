#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jun Kim
# Licensed under the MIT License
"""
Convert a DeepSpeed ZeRO checkpoint into a Hugging Face model folder.

- Adds "<think>" and "</think>" to the tokenizer if missing.
- Resizes the model's embeddings to match the updated tokenizer.
- Loads the consolidated state dict from DS shards and saves sharded safetensors.
- Works whether --checkpoint_dir points to ckpt-XXX/ (with a 'latest' file)
  or directly to global_stepXXX/.
- Optionally extracts & saves a gating head (Linear) as gating_head.pt, and
  writes gate metadata and (optionally) budgets into config.json.

Usage:
  python scripts/convert_ds_to_hf.py \
    --checkpoint_dir output/run_.../ckpt-100 \
    --original_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir output/run_.../ckpt-100-hf \
    --dtype bf16 \
    --max_shard_size 2GB \
    --budgets "0,64,-1" \
    --save_gate_as_module False
"""
import fire
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# DeepSpeed helper to reconstruct FP32/FP16/BF16 params from ZeRO shards
try:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
except Exception as e:
    raise RuntimeError(
        "Could not import DeepSpeed's get_fp32_state_dict_from_zero_checkpoint. "
        "Please ensure DeepSpeed is installed in this environment."
    ) from e


def _find_zero_step_dir(root: Path) -> Path:
    """
    Return a directory that actually contains *_model_states.pt files.

    Preference order:
      1) If 'latest' exists in root, use it (global_stepX).
      2) If root itself holds *_model_states.pt, return root.
      3) Recursively search for *_model_states.pt and return its parent.
      4) Else, choose the highest global_step* that contains such files.
    """
    root = root.resolve()

    # 1) 'latest' marker
    latest = root / "latest"
    if latest.is_file():
        name = latest.read_text().strip()
        step_name = name if name.startswith("global_step") else f"global_step{name}"
        cand = root / step_name
        if cand.is_dir():
            return cand

    # 2) files directly in root
    if any(root.glob("*_model_states.pt")):
        return root

    # 3) recursive search
    files = list(root.rglob("*_model_states.pt"))
    if files:
        return files[0].parent

    # 4) fallback to highest global_step*
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
      - step_dir: the 'global_stepX' directory we resolved (for logging)

    Rules:
      - If user passed a global_stepX dir: ds_root = parent, tag = global_stepX.
      - Else if ckpt root has 'latest': ds_root = root, tag = name in 'latest'.
      - Else: fall back to discovered step_dir; if it's global_stepX, use parent+tag.
    """
    checkpoint_dir = checkpoint_dir.resolve()

    # Case A: explicit global_stepX
    if checkpoint_dir.name.startswith("global_step") and checkpoint_dir.is_dir():
        return checkpoint_dir.parent, checkpoint_dir.name, checkpoint_dir

    # Case B: ckpt root with 'latest'
    latest = checkpoint_dir / "latest"
    if latest.is_file():
        name = latest.read_text().strip()
        tag = name if name.startswith("global_step") else f"global_step{name}"
        step_dir = checkpoint_dir / tag
        if not step_dir.is_dir():
            # probe
            step_dir = _find_zero_step_dir(checkpoint_dir)
            tag = step_dir.name if step_dir.name.startswith("global_step") else None
        return checkpoint_dir, tag, step_dir

    # Case C: no 'latest' – probe
    step_dir = _find_zero_step_dir(checkpoint_dir)
    if step_dir.name.startswith("global_step"):
        return step_dir.parent, step_dir.name, step_dir
    # shards directly at this folder
    return step_dir, None, step_dir


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def _parse_budgets(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    # Handle both string and already-parsed inputs
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    arr = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        arr.append(int(x))
    return arr if arr else None


def _find_gating_params(
    sd: Dict[str, torch.Tensor],
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Search the state_dict for a linear 'gating head' (weight/bias) by name heuristics.
    Returns a dict {'weight': tensor, 'bias': tensor, 'prefix': '...'} if found.
    """
    candidates = []
    keys = list(sd.keys())
    # prefer obvious names first
    preferred_prefixes = ["gating_head", "gate_head", "depth_gate", "gate", "gater"]
    for pref in preferred_prefixes:
        w, b = f"{pref}.weight", f"{pref}.bias"
        if w in sd and b in sd:
            candidates.append(dict(weight=sd[w], bias=sd[b], prefix=pref))
    # fallback: any "*gate*.weight/.bias" pair
    if not candidates:
        for k in keys:
            if k.endswith(".weight") and ("gate" in k or "gating" in k):
                prefix = k[: -len(".weight")]
                b = prefix + ".bias"
                if b in sd:
                    candidates.append(dict(weight=sd[k], bias=sd[b], prefix=prefix))
    if not candidates:
        return None

    # pick the first viable by shape: [out, hidden], [out]
    for c in candidates:
        W, b = c["weight"], c["bias"]
        if W.ndim == 2 and b.ndim == 1 and W.shape[0] == b.shape[0]:
            return c
    return None


def _create_think_aware_chat_template(original_template: str) -> str:
    """
    Create a modified chat template that automatically starts assistant responses with <think>.

    This ensures training-inference consistency for depth-controlled Matryoshka models
    where all training data has assistant responses starting with <think> tokens.

    For Llama-3.1 templates, we modify the content rendering to prepend <think> to assistant messages.
    """
    # For Llama-3.1 templates, we need to modify the content rendering
    if "<|start_header_id|>assistant<|end_header_id|>" in original_template:
        print(
            "[convert] Detected Llama-3.1 style template, modifying for <think> injection"
        )

        # Find the pattern where message content is rendered for assistant
        # In Llama-3.1 template: {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}

        # Strategy: Modify the content rendering to add <think> for assistant messages
        modified_template = original_template

        # Look for the pattern where message content is rendered
        content_pattern = "message['content'] | trim"
        if content_pattern in modified_template:
            # Replace with conditional <think> injection
            replacement = "('<think>' + message['content'] if message['role'] == 'assistant' else message['content']) | trim"
            modified_template = modified_template.replace(content_pattern, replacement)
            print(
                "[convert] Modified content rendering to inject <think> for assistant messages"
            )
        else:
            # Fallback: look for other content patterns
            content_pattern_alt = "message['content']"
            if content_pattern_alt in modified_template:
                replacement = "('<think>' + message['content'] if message['role'] == 'assistant' else message['content'])"
                modified_template = modified_template.replace(
                    content_pattern_alt, replacement
                )
                print(
                    "[convert] Modified content rendering (fallback) to inject <think> for assistant messages"
                )

        # Also modify the generation prompt to include <think>
        gen_prompt_pattern = (
            "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        )
        if gen_prompt_pattern in modified_template:
            replacement = (
                "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n<think>' }}"
            )
            modified_template = modified_template.replace(
                gen_prompt_pattern, replacement
            )
            print("[convert] Modified generation prompt to start with <think>")

        return modified_template

    else:
        # For other template formats, use a more generic approach
        print(
            "[convert] Warning: Unknown chat template format, using generic <think> injection"
        )

        # Try to find where assistant content is inserted and prepend <think>
        if "assistant" in original_template.lower():
            # Simple replacement strategy
            return original_template.replace(
                "{{- message['content'] }}",
                "{{- '<think>' + message['content'] if message['role'] == 'assistant' else message['content'] }}",
            )

        return original_template


def convert_deepspeed_to_hf(
    checkpoint_dir: str,
    original_model: str,
    output_dir: str,
    dtype: str = "fp16",
    max_shard_size: str = "2GB",
    budgets: str = "",
    save_gate_as_module: bool = True,
):
    """
    Convert a DeepSpeed ZeRO checkpoint into a Hugging Face model folder.

    Args:
        checkpoint_dir: Path to DS checkpoint (ckpt-XXX or global_stepXXX)
        original_model: Base HF model name or path (architecture & config)
        output_dir: Where to write the HF model
        dtype: Save weights dtype (fp16, bf16, or fp32)
        max_shard_size: Max shard size for safetensors (default: "2GB")
        budgets: Optional comma-separated budgets to store in config, e.g. "0,64,-1"
        save_gate_as_module: Also attach the gating head module to the model before saving
    """
    # Validate dtype
    if dtype not in ["fp16", "bf16", "fp32"]:
        raise ValueError(f"dtype must be one of ['fp16', 'bf16', 'fp32'], got: {dtype}")

    ckpt_root = Path(checkpoint_dir)
    ds_root, tag, step_dir = _resolve_ds_root_and_tag(ckpt_root)

    print(f"[convert] Using DS step dir: {step_dir}")
    print(
        "[convert] Loading consolidated state dict from DeepSpeed shards ... (this can take a while)"
    )
    # Single DS call: avoid 'latest' lookup inside global_stepX
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(ds_root), tag=tag)
    state_dict = _strip_module_prefix(state_dict)

    # dtype map for model save; keep gating head we write to pt as float32 (safer for CPU inference)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    save_dtype = dtype_map[dtype]

    # Optional downcast to reduce peak RAM (only model weights, not the external gate file)
    if save_dtype != torch.float32:
        for k in list(state_dict.keys()):
            t = state_dict[k]
            if torch.is_floating_point(t):
                state_dict[k] = t.to(save_dtype)

    # Try to discover gating head tensors (BEFORE we mutate config/model)
    gate_pack = _find_gating_params(state_dict)  # uses (possibly downcast) tensors
    if gate_pack is not None:
        # We'll also keep an fp32 copy for the separate file
        w32 = gate_pack["weight"].detach().to(torch.float32).cpu()
        b32 = gate_pack["bias"].detach().to(torch.float32).cpu()
        gate_info = dict(
            prefix=gate_pack["prefix"],
            out_features=w32.shape[0],
            in_features=w32.shape[1],
        )
    else:
        gate_info = None

    # Load tokenizer & add reasoning tokens if missing
    tok = AutoTokenizer.from_pretrained(original_model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    added = 0
    for tkn in ["<think>", "</think>"]:
        if tkn not in tok.get_vocab():
            tok.add_tokens([tkn])
            added += 1

    print(f"[convert] Instantiating base model: {original_model}")
    model = AutoModelForCausalLM.from_pretrained(
        original_model,
        torch_dtype=save_dtype,
        low_cpu_mem_usage=False,
        device_map="cpu",
    )

    if added > 0 and len(tok) != model.config.vocab_size:
        print(
            f"[convert] Resizing token embeddings: {model.config.vocab_size} → {len(tok)}"
        )
        model.resize_token_embeddings(len(tok))

    # If user asked to attach the gating module into the model object itself
    if gate_info is not None and save_gate_as_module:
        try:
            hidden = getattr(model.config, "hidden_size", None)
            if hidden is None:
                print(
                    "[convert][warn] model.config.hidden_size missing; cannot attach gating module. Skipping."
                )
            elif hidden != gate_info["in_features"]:
                print(
                    f"[convert][warn] Hidden size mismatch (config={hidden}, gate_in={gate_info['in_features']}). "
                    "Attaching anyway, but downstream loading won't auto-recreate this module."
                )
                model.gating_head = nn.Linear(
                    gate_info["in_features"], gate_info["out_features"], bias=True
                )
                # load weights in save_dtype
                model.gating_head.weight.data.copy_(gate_pack["weight"].to(save_dtype))
                model.gating_head.bias.data.copy_(gate_pack["bias"].to(save_dtype))
            else:
                model.gating_head = nn.Linear(
                    hidden, gate_info["out_features"], bias=True
                )
                model.gating_head.weight.data.copy_(gate_pack["weight"].to(save_dtype))
                model.gating_head.bias.data.copy_(gate_pack["bias"].to(save_dtype))
        except Exception as e:
            print(f"[convert][warn] Failed to attach gating module to model: {e}")

    # Load base LM weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"[convert][warn] Missing keys: {len(missing)} (showing up to 10) -> {missing[:10]}"
        )
    if unexpected:
        # It's fine if gating weights sit in 'unexpected' when we didn't attach a module
        show = [k for k in unexpected if ("gate" in k or "gating" in k)]
        extra = f"  (gate-related: {show[:5]})" if show else ""
        print(
            f"[convert][warn] Unexpected keys: {len(unexpected)} (showing up to 10) -> {unexpected[:10]}{extra}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modify chat template to ensure assistant responses start with <think>
    # This ensures training-inference consistency for depth-controlled generation
    original_template = tok.chat_template
    if original_template and "<think>" not in original_template:
        print(
            "[convert] Modifying chat template to include <think> tokens for assistant responses"
        )

        # Create custom template that starts assistant responses with <think>
        # This matches the training data format where all assistant responses begin with <think>
        custom_template = _create_think_aware_chat_template(original_template)
        tok.chat_template = custom_template

        print(
            f"[convert] Original template preserved as 'original_chat_template' in tokenizer config"
        )
        # Preserve original template for reference
        if hasattr(tok, "init_kwargs"):
            tok.init_kwargs["original_chat_template"] = original_template
        else:
            tok.init_kwargs = {"original_chat_template": original_template}

    # Save HF model + tokenizer
    print(f"[convert] Saving HF model to: {out_dir}")
    model.save_pretrained(
        str(out_dir), safe_serialization=True, max_shard_size=max_shard_size
    )
    tok.save_pretrained(str(out_dir))

    # If we found a gating head, save it separately and annotate config
    if gate_info is not None:
        gate_file = out_dir / "gating_head.pt"
        torch.save({"weight": w32, "bias": b32}, gate_file)
        print(f"[convert] Saved gating head → {gate_file}")

        # Write metadata into config.json (HF allows extra attributes)
        try:
            # Reload config from disk to update JSON (safer than relying on in-memory object post-save)
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(str(out_dir))
            cfg.matryoshka_has_gate = True
            cfg.gating_head_out = int(gate_info["out_features"])
            cfg.gating_head_hidden = int(gate_info["in_features"])
            parsed_budgets = _parse_budgets(budgets)
            if parsed_budgets is not None:
                cfg.matryoshka_budgets = parsed_budgets
            cfg.save_pretrained(str(out_dir))
            print(
                "[convert] Updated config.json with gate metadata"
                + (f" and budgets={budgets}" if budgets else "")
            )
        except Exception as e:
            print(
                f"[convert][warn] Could not update config.json with gate metadata: {e}"
            )
    else:
        print("[convert] No gating head found in DS checkpoint. Skipping gate export.")

    print("[convert] Done.")


if __name__ == "__main__":
    fire.Fire(convert_deepspeed_to_hf)
