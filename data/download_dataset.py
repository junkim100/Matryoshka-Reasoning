#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jun Kim
# Licensed under the MIT License
"""
Unified dataset builder for GAIR/LIMO and openai/gsm8k with optional chat template integration.

This script can create datasets in two formats:
1. Simple format: Basic question/solution pairs
2. Chat template format: Full chat template with <think>reasoning</think>answer

Examples
--------
# Simple format (original behavior)
python data/download_dataset.py \
  --ratio 0.5 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --max_token_length 4096 \
  --shuffle True \
  --create_train_val_split True

# Chat template format (for Matryoshka training)
python data/download_dataset.py \
  --ratio 0.5 \
  --use_chat_template True \
  --system_message "You are a helpful assistant. When solving problems, think step by step inside <think> tags, then provide your final answer." \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --max_token_length 4096
"""

import os
import json
import random
import re
from typing import Optional, Dict, Any, Tuple, List

try:
    import fire
except ImportError:
    fire = None  # Allow importing this module without fire installed
import jsonlines
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ------------- Chat template helpers ------------- #


def _ensure_think_tags(solution: str) -> str:
    """
    Ensure solution begins with <think>...</think> covering the reasoning, with the
    final answer section following. Prefer splitting at '**Final Answer**' when present.
    Otherwise, fall back to the first \\boxed{...} occurrence; if neither exists, wrap all.
    """
    solution = solution.strip()

    # If already has proper think tags, return as-is
    if solution.startswith("<think>") and "</think>" in solution:
        return solution

    # Prefer the canonical '**Final Answer**' header when present
    fa_match = re.search(r"\*\*Final Answer\*\*", solution)
    if fa_match:
        thinking = solution[: fa_match.start()].strip()
        final_section = solution[fa_match.start() :].lstrip()
        if thinking:
            return f"<think>{thinking}</think>\n\n{final_section}"
        else:
            # Synthesize a brief thought if none exists before the header
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
            if boxed_match:
                ans = boxed_match.group(1)
                synth = f"The answer is {ans}."
            else:
                synth = "Let me think step by step."
            return f"<think>{synth}</think>\n\n{final_section}"

    # Fallback: split at first boxed answer
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
    if boxed_match:
        ans = boxed_match.group(1)
        before_boxed = solution[: boxed_match.start()].strip()
        after_boxed = solution[boxed_match.start() :].strip()
        thinking = before_boxed if before_boxed else f"The answer is {ans}."
        # Ensure final section has '**Final Answer**' header
        if "**Final Answer**" in after_boxed:
            final_section = after_boxed
        else:
            final_section = f"**Final Answer**\n{after_boxed}"
        return f"<think>{thinking}</think>\n\n{final_section}"

    # No clear split point: wrap entire solution in think tags
    return f"<think>{solution}</think>"


def apply_chat_template_to_sample(
    question: str, solution: str, answer: str, tokenizer
) -> Dict[str, Any]:
    """
    Apply chat template to create a training sample.

    Returns a sample with:
    - text: Full chat template applied
    - assistant_start: Token position where assistant response begins
    - question, solution, answer: Original components
    """
    # Ensure solution has proper think tags
    formatted_solution = _ensure_think_tags(solution)

    # Create messages for chat template
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When solving problems, think step by step inside <think> tags, then provide your final answer.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": formatted_solution},
    ]

    # Apply chat template
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Find where assistant response starts
    assistant_start = full_text.find(formatted_solution)
    if assistant_start == -1:
        # Fallback: find a reasonable position
        assistant_start = len(full_text) - len(formatted_solution)

    return {
        "text": full_text,
        "assistant_start": assistant_start,
        "question": question,
        "solution": formatted_solution,
        "answer": answer,
    }


# ------------- GSM8K helpers ------------- #

_ANS_LINE_RE = re.compile(r"^\s*####\s*(.+?)\s*$")


def _extract_final_answer_gsm8k(answer_str: str) -> Tuple[str, str]:
    """
    GSM8K 'answer' is free-form reasoning ending with a line like:
      '#### 72'
    Return (reasoning, final_answer).
    If no '####' line, fall back to last number in the string.
    """
    if not isinstance(answer_str, str):
        answer_str = str(answer_str or "")

    lines = answer_str.splitlines()
    reasoning = answer_str
    final_answer = ""

    # Scan bottom-up for '#### ...'
    for i in range(len(lines) - 1, -1, -1):
        m = _ANS_LINE_RE.match(lines[i])
        if m:
            final_answer = m.group(1).strip()
            reasoning = "\n".join(lines[:i])
            break

    if not final_answer:
        # Fallback: last numeric-looking token
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", answer_str)
        if numbers:
            final_answer = numbers[-1]
        else:
            final_answer = answer_str.strip()

    return reasoning.strip(), final_answer.strip()


def _format_gsm8k(ex: Dict[str, Any]) -> Dict[str, str]:
    q = str(ex.get("question", "")).strip()
    raw = str(ex.get("answer", "")).strip()
    reasoning, ans = _extract_final_answer_gsm8k(raw)
    # Enforce think + final answer format
    sol = f"<think>{reasoning}</think>\n\n**Final Answer**\n\\boxed{{{ans}}}"
    return {"question": q, "solution": sol, "answer": ans}


# ------------- LIMO helpers ------------- #


def _strip_think_tags(text: str) -> str:
    return str(text or "").replace("<think>", "").replace("</think>", "")


def _format_limo(ex: Dict[str, Any]) -> Dict[str, str]:
    q = str(ex.get("question", "")).strip()
    sol = str(ex.get("solution", "")).strip()
    ans = str(ex.get("answer", "")).strip()

    # Ensure standardized final answer header and boxed
    if "**Final Answer**" not in sol:
        if "\\boxed" in sol:
            boxed_pos = sol.find("\\boxed")
            sol = (
                sol[:boxed_pos].rstrip()
                + "\n\n**Final Answer**\n"
                + sol[boxed_pos:].lstrip()
            )
        elif ans:
            sol = f"{sol}\n\n**Final Answer**\n\\boxed{{{ans}}}"
    elif "\\boxed" not in sol and ans:
        sol = f"{sol}\n\\boxed{{{ans}}}"

    # Place </think> immediately before '**Final Answer**'
    fa_idx = sol.find("**Final Answer**")
    if fa_idx != -1:
        reasoning = _strip_think_tags(sol[:fa_idx]).strip()
        final_section = sol[fa_idx:].lstrip()
        # Ensure exactly one newline after the '**Final Answer**' header
        final_section = re.sub(
            r"(\*\*Final Answer\*\*)[ \t]*\n*", r"\1\n", final_section, count=1
        )
        sol = f"<think>{reasoning}</think>\n\n{final_section}"
    else:
        # Fallback: split at first boxed
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", sol)
        if boxed_match:
            reasoning = _strip_think_tags(sol[: boxed_match.start()]).strip()
            final_section = f"**Final Answer**\n{sol[boxed_match.start():].lstrip()}"
            sol = f"<think>{reasoning}</think>\n\n{final_section}"
        else:
            # No clear split; wrap all as think and append final if ans exists
            reasoning = _strip_think_tags(sol)
            sol = f"<think>{reasoning}</think>"
            if ans:
                sol += f"\n\n**Final Answer**\n\\boxed{{{ans}}}"

    return {"question": q, "solution": sol, "answer": ans}


# ------------- EleutherAI/arithmetic helpers ------------- #


def _extract_question_from_context(context: str) -> str:
    """Extract the question text from a context like 'Question: ... Answer: '"""
    context = str(context or "").strip()
    # Try robust split using the last occurrence of 'Answer:'
    ans_idx = context.lower().rfind("answer:")
    q_idx = context.lower().find("question:")
    if q_idx != -1 and ans_idx != -1 and q_idx < ans_idx:
        question = context[q_idx + len("Question:") : ans_idx].strip()
        return question if question else context
    # Fallback: strip a trailing 'Answer:'
    if ans_idx != -1:
        maybe_q = context[:ans_idx].strip()
        # Remove leading 'Question:' if present
        if maybe_q.lower().startswith("question:"):
            maybe_q = maybe_q[len("Question:") :].strip()
        return maybe_q if maybe_q else context
    # Final fallback: return context as-is
    return context


def _format_arithmetic(ex: Dict[str, Any]) -> Dict[str, str]:
    q_raw = ex.get("context", "")
    q = _extract_question_from_context(q_raw)
    ans = str(ex.get("completion", "")).strip()
    # Empty think, then standardized final answer section with boxed answer
    sol = f"<think></think>\n\n**Final Answer**\n\\boxed{{{ans}}}"
    return {"question": q, "solution": sol, "answer": ans}


# ------------- Token length utility ------------- #


def _count_tokens(tok: AutoTokenizer, question: str, solution: str) -> int:
    """
    Approximate token length by tokenizing 'question + solution'.
    (Training will add a small chat-template overhead.)
    """
    txt = f"{question}\n\n{solution}"
    return len(tok.encode(txt, add_special_tokens=True))


# ------------- Main mixing pipeline ------------- #


def mix_gsm8k_limo(
    ratio: float = 0.5,
    # splits
    gsm8k_split: str = "train",
    limo_split: str = "train",
    # filtering
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_token_length: int = 4096,
    min_token_length: int = 0,
    # sampling & shuffling
    seed: int = 42,
    shuffle: bool = True,
    strict_ratio: bool = False,  # if True, may downsample LIMO to hit ratio exactly
    # chat template options
    use_chat_template: bool = False,
    system_message: str = "You are a helpful assistant. When solving problems, think step by step inside <think> tags, then provide your final answer.",
    # output
    create_train_val_split: bool = True,
    train_ratio: float = 0.9,
    output_prefix: Optional[str] = None,  # e.g., "data/gsm8k_limo_mix"
):
    """
    Build a mixed dataset of GAIR/LIMO and openai/gsm8k with optional chat template integration.

    Args
    ----
    ratio: desired fraction of LIMO in the final mix (0<ratio<1). Default 0.5.
           Example: ratio=0.5 → LIMO=GSM8K count.
    gsm8k_split: HF split for GSM8K (e.g., 'train' or 'test')
    limo_split: HF split for LIMO (usually 'train')
    tokenizer: model name/path for token counting
    max_token_length, min_token_length: keep examples within [min, max]
    seed: RNG seed for reproducible sampling/shuffling
    shuffle: shuffle before saving
    strict_ratio: if True and GSM8K is too small, will downsample LIMO to match ratio.
                  if False (default), keeps ALL LIMO and accepts ratio slippage.
    use_chat_template: if True, create chat template format with <think> tags
    system_message: system message for chat template (only used if use_chat_template=True)
    create_train_val_split: write train/val files instead of one merged file
    train_ratio: train fraction when splitting
    output_prefix: if None → auto name under 'data/'
    """
    assert 0.0 < ratio < 1.0, "ratio must be in (0,1), e.g. 0.5 for 50/50."

    random.seed(seed)

    print(
        f"[cfg] ratio={ratio} (LIMO share), gsm8k_split={gsm8k_split}, limo_split={limo_split}"
    )
    print(
        f"[cfg] tokenizer={tokenizer}, token range=[{min_token_length},{max_token_length}]"
    )
    print(f"[cfg] seed={seed}, shuffle={shuffle}, strict_ratio={strict_ratio}")
    print(
        f"[cfg] create_train_val_split={create_train_val_split}, train_ratio={train_ratio}"
    )

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Load datasets
    print("[load] GAIR/LIMO ...")
    limo = load_dataset("GAIR/LIMO", split=limo_split)
    print(f"[load] LIMO: {len(limo)} rows")

    print("[load] openai/gsm8k ...")
    gsm = load_dataset("openai/gsm8k", "main", split=gsm8k_split)
    print(f"[load] GSM8K: {len(gsm)} rows")

    # Format + filter LIMO
    print("[prep] Formatting & filtering LIMO ...")
    limo_proc: List[Dict[str, str]] = []
    for ex in tqdm(limo, desc="LIMO"):
        out = _format_limo(ex)
        n_tok = _count_tokens(tok, out["question"], out["solution"])
        if min_token_length <= n_tok <= max_token_length:
            out["token_length"] = n_tok
            limo_proc.append(out)

    # Format + filter GSM8K
    print("[prep] Formatting & filtering GSM8K ...")
    gsm_proc: List[Dict[str, str]] = []
    for ex in tqdm(gsm, desc="GSM8K"):
        out = _format_gsm8k(ex)
        n_tok = _count_tokens(tok, out["question"], out["solution"])
        if min_token_length <= n_tok <= max_token_length:
            out["token_length"] = n_tok
            gsm_proc.append(out)

    print(f"[stats] After filtering: LIMO={len(limo_proc)}, GSM8K={len(gsm_proc)}")

    # Compute target selection sizes
    nL = len(limo_proc)
    nG = len(gsm_proc)
    if nL == 0 or nG == 0:
        raise RuntimeError(
            "After filtering, one side is empty. Adjust token-length bounds or tokenizer."
        )

    # Keep ALL LIMO by default
    target_L = nL
    # To achieve ratio = L / (L + G), we need G = L*(1-r)/r
    target_G = int(round(target_L * (1.0 - ratio) / ratio))

    if target_G > nG:
        msg = (
            f"[warn] Not enough GSM8K to hit ratio={ratio:.2f} with ALL LIMO."
            f" Need {target_G}, have {nG}."
        )
        if strict_ratio:
            # Downsample L to fit available G
            target_G = nG
            # L = G * r/(1-r)
            target_L = int(round(target_G * ratio / (1.0 - ratio)))
            msg += f" strict_ratio=True → downsampling LIMO to {target_L}."
        else:
            # Keep all L; take all G we have (ratio will skew)
            target_G = nG
            msg += " strict_ratio=False → keeping all LIMO; ratio will skew."
        print(msg)
    else:
        print(
            f"[mix] Using ALL LIMO ({target_L}) and GSM8K={target_G} to target ratio={ratio:.2f}."
        )

    # Sample GSM8K and (maybe) sample LIMO for strict ratio case:
    random.shuffle(gsm_proc)
    gsm_sel = gsm_proc[:target_G]

    if strict_ratio and target_L < nL:
        random.shuffle(limo_proc)
        limo_sel = limo_proc[:target_L]
    else:
        limo_sel = limo_proc  # keep all

    # Combine
    mix = limo_sel + gsm_sel
    if shuffle:
        random.shuffle(mix)

    # Apply chat template if requested
    if use_chat_template:
        print(
            f"[chat] Applying chat template with system message: {system_message[:50]}..."
        )

        # Update system message in the apply_chat_template_to_sample function
        def apply_chat_template_with_custom_system(
            question: str, solution: str, answer: str
        ) -> Dict[str, Any]:
            formatted_solution = _ensure_think_tags(solution)

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_solution},
            ]

            full_text = tok.apply_chat_template(messages, tokenize=False)
            assistant_start = full_text.find(formatted_solution)
            if assistant_start == -1:
                assistant_start = len(full_text) - len(formatted_solution)

            return {
                "text": full_text,
                "assistant_start": assistant_start,
                "question": question,
                "solution": formatted_solution,
                "answer": answer,
            }

        # Convert all samples to chat template format
        chat_mix = []
        for ex in tqdm(mix, desc="Applying chat template"):
            chat_sample = apply_chat_template_with_custom_system(
                ex["question"], ex["solution"], ex["answer"]
            )
            chat_mix.append(chat_sample)

        mix = chat_mix
        print(f"[chat] Converted {len(mix)} samples to chat template format")

    # Clean field for saving
    for ex in mix:
        if "token_length" in ex:
            del ex["token_length"]

    # Final stats
    nL_final = sum(
        1 for ex in mix if "\\boxed" in ex.get("solution", "") and ex in limo_sel
    )
    # Better robust counts:
    is_limo_set = set(id(ex) for ex in limo_sel)
    is_gsm_set = set(id(ex) for ex in gsm_sel)
    nL_final = sum(1 for ex in mix if id(ex) in is_limo_set)
    nG_final = sum(1 for ex in mix if id(ex) in is_gsm_set)
    final_ratio = nL_final / max(1, (nL_final + nG_final))
    print(
        f"[done] Mixed dataset size={len(mix)}  (LIMO={nL_final}, GSM8K={nG_final}, ratio≈{final_ratio:.3f})"
    )

    # Output names
    if output_prefix is None:
        base = f"data/gsm8k_limo_r{str(ratio).replace('.','_')}_max{max_token_length}_min{min_token_length}_seed{seed}"
        if use_chat_template:
            base += "_chat"
        if shuffle:
            base += "_shuf"
    else:
        base = output_prefix

    os.makedirs(os.path.dirname(base), exist_ok=True)


def mix_limo_gsm8k_arithmetic_equal(
    # splits/configs
    limo_split: str = "train",
    gsm8k_split: str = "train",
    arithmetic_split: str = "validation",
    arithmetic_config: str = "arithmetic_2da",
    # filtering
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_token_length: int = 4096,
    min_token_length: int = 0,
    # sampling & shuffling
    seed: int = 42,
    shuffle: bool = True,
    # chat template options
    use_chat_template: bool = True,
    system_message: str = "You are a helpful assistant. When solving problems, think step by step inside <think> tags, then provide your final answer.",
    # output
    create_train_val_split: bool = True,
    train_ratio: float = 0.9,
    output_prefix: Optional[str] = None,
):
    """
    Build a mixed dataset from GAIR/LIMO, openai/gsm8k, and EleutherAI/arithmetic
    in a 1:1:1 ratio by taking the minimum available size after filtering
    (default max_token_length=4096).

    Output format matches other builders. If use_chat_template is True, wraps with
    the tokenizer's chat template and positions </think> before '**Final Answer**'.
    """
    assert 0 <= min_token_length <= max_token_length

    random.seed(seed)

    print("[cfg] 1:1:1 mix across LIMO, GSM8K, Arithmetic")
    print(
        f"[cfg] splits: limo={limo_split}, gsm8k={gsm8k_split}, arithmetic={arithmetic_split} ({arithmetic_config})"
    )
    print(
        f"[cfg] tokenizer={tokenizer}, token range=[{min_token_length},{max_token_length}]"
    )
    print(f"[cfg] seed={seed}, shuffle={shuffle}")
    print(
        f"[cfg] create_train_val_split={create_train_val_split}, train_ratio={train_ratio}"
    )

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Load datasets
    print("[load] GAIR/LIMO ...")
    limo = load_dataset("GAIR/LIMO", split=limo_split)
    print(f"[load] LIMO: {len(limo)} rows")

    print("[load] openai/gsm8k ...")
    gsm = load_dataset("openai/gsm8k", "main", split=gsm8k_split)
    print(f"[load] GSM8K: {len(gsm)} rows")

    print("[load] EleutherAI/arithmetic ...")
    arith = load_dataset(
        "EleutherAI/arithmetic", arithmetic_config, split=arithmetic_split
    )
    print(f"[load] arithmetic: {len(arith)} rows")

    # Format + filter
    print("[prep] Formatting & filtering LIMO ...")
    limo_proc: List[Dict[str, str]] = []
    for ex in tqdm(limo, desc="LIMO"):
        out = _format_limo(ex)
        n_tok = _count_tokens(tok, out["question"], out["solution"])
        if min_token_length <= n_tok <= max_token_length:
            out["token_length"] = n_tok
            limo_proc.append(out)

    print("[prep] Formatting & filtering GSM8K ...")
    gsm_proc: List[Dict[str, str]] = []
    for ex in tqdm(gsm, desc="GSM8K"):
        out = _format_gsm8k(ex)
        n_tok = _count_tokens(tok, out["question"], out["solution"])
        if min_token_length <= n_tok <= max_token_length:
            out["token_length"] = n_tok
            gsm_proc.append(out)

    print("[prep] Formatting & filtering arithmetic ...")
    arith_proc: List[Dict[str, str]] = []
    for ex in tqdm(arith, desc="arithmetic"):
        out = _format_arithmetic(ex)
        n_tok = _count_tokens(tok, out["question"], out["solution"])
        if min_token_length <= n_tok <= max_token_length:
            out["token_length"] = n_tok
            arith_proc.append(out)

    nL, nG, nA = len(limo_proc), len(gsm_proc), len(arith_proc)
    print(f"[stats] After filtering: LIMO={nL}, GSM8K={nG}, arithmetic={nA}")

    subset_size = min(nL, nG, nA)
    if subset_size == 0:
        raise RuntimeError(
            "At least one dataset is empty after filtering. Adjust token-length bounds or tokenizer."
        )

    print(f"[mix] Selecting {subset_size} from each to form a 1:1:1 mix")

    random.shuffle(limo_proc)
    random.shuffle(gsm_proc)
    random.shuffle(arith_proc)

    limo_sel = limo_proc[:subset_size]
    gsm_sel = gsm_proc[:subset_size]
    arith_sel = arith_proc[:subset_size]

    mix = limo_sel + gsm_sel + arith_sel
    if shuffle:
        random.shuffle(mix)

    # Optional chat template
    if use_chat_template:
        print(
            f"[chat] Applying chat template with system message: {system_message[:50]}..."
        )

        def apply_chat_template_with_custom_system(
            question: str, solution: str, answer: str
        ) -> Dict[str, Any]:
            formatted_solution = _ensure_think_tags(solution)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_solution},
            ]
            full_text = tok.apply_chat_template(messages, tokenize=False)
            assistant_start = full_text.find(formatted_solution)
            if assistant_start == -1:
                assistant_start = len(full_text) - len(formatted_solution)
            return {
                "text": full_text,
                "assistant_start": assistant_start,
                "question": question,
                "solution": formatted_solution,
                "answer": answer,
            }

        chat_mix = []
        for ex in tqdm(mix, desc="Applying chat template"):
            chat_mix.append(
                apply_chat_template_with_custom_system(ex["question"], ex["solution"], ex["answer"])  # type: ignore
            )
        mix = chat_mix
        print(f"[chat] Converted {len(mix)} samples to chat template format")

    # Clean token_length
    for ex in mix:
        if isinstance(ex, dict) and "token_length" in ex:
            del ex["token_length"]

    print(f"[done] Mixed dataset size={len(mix)}  (each subset={subset_size})")

    # Output naming
    if output_prefix is None:
        base = f"data/three_mix_equal_max{max_token_length}_min{min_token_length}_seed{seed}"
        if use_chat_template:
            base += "_chat"
        if shuffle:
            base += "_shuf"
    else:
        base = output_prefix

    os.makedirs(os.path.dirname(base), exist_ok=True)

    if create_train_val_split:
        split_idx = int(len(mix) * train_ratio)
        train_data = mix[:split_idx]
        val_data = mix[split_idx:]
        train_file = f"{base}_train.jsonl"
        val_file = f"{base}_val.jsonl"
        print(f"[save] Train → {train_file}  ({len(train_data)} rows)")
        with jsonlines.open(train_file, "w") as w:
            for ex in train_data:
                w.write(ex)
        print(f"[save] Val   → {val_file}  ({len(val_data)} rows)")
        with jsonlines.open(val_file, "w") as w:
            for ex in val_data:
                w.write(ex)
    else:
        out_file = f"{base}.jsonl"
        print(f"[save] All → {out_file}  ({len(mix)} rows)")
        with jsonlines.open(out_file, "w") as w:
            for ex in mix:
                w.write(ex)


def build_arithmetic(
    arithmetic_split: str = "train",
    arithmetic_config: str = "arithmetic_2da",
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_token_length: int = 4096,
    min_token_length: int = 0,
    seed: int = 42,
    shuffle: bool = True,
    use_chat_template: bool = False,
    system_message: str = "You are a helpful assistant. When solving problems, think step by step inside <think> tags, then provide your final answer.",
    create_train_val_split: bool = True,
    train_ratio: float = 0.9,
    output_prefix: Optional[str] = None,
):
    """
    Build a preprocessed dataset from EleutherAI/arithmetic in the same format as others.
    - Extract question from 'context', final answer from 'completion'.
    - Produce solution as '<think></think>\n\n**Final Answer**\n\\boxed{answer}'.
    - Optionally convert to chat-template format.
    """
    random.seed(seed)

    print(f"[cfg] arithmetic_split={arithmetic_split}, config={arithmetic_config}")
    print(
        f"[cfg] tokenizer={tokenizer}, token range=[{min_token_length},{max_token_length}]"
    )
    print(f"[cfg] seed={seed}, shuffle={shuffle}")
    print(
        f"[cfg] create_train_val_split={create_train_val_split}, train_ratio={train_ratio}"
    )

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Load dataset
    print("[load] EleutherAI/arithmetic ...")
    arith = load_dataset(
        "EleutherAI/arithmetic", arithmetic_config, split=arithmetic_split
    )
    print(f"[load] arithmetic: {len(arith)} rows")

    # Format + filter
    print("[prep] Formatting & filtering arithmetic ...")
    proc: List[Dict[str, str]] = []
    for ex in tqdm(arith, desc="arithmetic"):
        out = _format_arithmetic(ex)
        n_tok = _count_tokens(tok, out["question"], out["solution"])
        if min_token_length <= n_tok <= max_token_length:
            out["token_length"] = n_tok
            proc.append(out)

    print(f"[stats] After filtering: arithmetic={len(proc)}")

    if shuffle:
        random.shuffle(proc)

    # Apply chat template if requested
    if use_chat_template:
        print(
            f"[chat] Applying chat template with system message: {system_message[:50]}..."
        )

        def apply_chat_template_with_custom_system(
            question: str, solution: str, answer: str
        ) -> Dict[str, Any]:
            formatted_solution = _ensure_think_tags(solution)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": formatted_solution},
            ]
            full_text = tok.apply_chat_template(messages, tokenize=False)
            assistant_start = full_text.find(formatted_solution)
            if assistant_start == -1:
                assistant_start = len(full_text) - len(formatted_solution)
            return {
                "text": full_text,
                "assistant_start": assistant_start,
                "question": question,
                "solution": formatted_solution,
                "answer": answer,
            }

        chat_proc = []
        for ex in tqdm(proc, desc="Applying chat template"):
            chat_proc.append(
                apply_chat_template_with_custom_system(ex["question"], ex["solution"], ex["answer"])  # type: ignore
            )
        proc = chat_proc
        print(f"[chat] Converted {len(proc)} samples to chat template format")

    # Clean field for saving
    for ex in proc:
        if isinstance(ex, dict) and "token_length" in ex:
            del ex["token_length"]

    # Output naming
    if output_prefix is None:
        base = f"data/arithmetic{('_' + arithmetic_config) if arithmetic_config else ''}_max{max_token_length}_min{min_token_length}_seed{seed}"
        if use_chat_template:
            base += "_chat"
        if shuffle:
            base += "_shuf"
    else:
        base = output_prefix

    os.makedirs(os.path.dirname(base), exist_ok=True)

    if create_train_val_split:
        split_idx = int(len(proc) * train_ratio)
        train_data = proc[:split_idx]
        val_data = proc[split_idx:]
        train_file = f"{base}_train.jsonl"
        val_file = f"{base}_val.jsonl"
        print(f"[save] Train → {train_file}  ({len(train_data)} rows)")
        with jsonlines.open(train_file, "w") as w:
            for ex in train_data:
                w.write(ex)
        print(f"[save] Val   → {val_file}  ({len(val_data)} rows)")
        with jsonlines.open(val_file, "w") as w:
            for ex in val_data:
                w.write(ex)
    else:
        out_file = f"{base}.jsonl"
        print(f"[save] All → {out_file}  ({len(proc)} rows)")
        with jsonlines.open(out_file, "w") as w:
            for ex in proc:
                w.write(ex)


if __name__ == "__main__":
    # Expose entry points via Fire only if installed
    if fire is None:
        raise RuntimeError(
            "The 'fire' package is required to run CLI entry points. Install via 'pip install fire'."
        )
    fire.Fire(
        {
            "mix_gsm8k_limo": mix_gsm8k_limo,
            "build_arithmetic": build_arithmetic,
            "mix_three_equal": mix_limo_gsm8k_arithmetic_equal,
        }
    )
