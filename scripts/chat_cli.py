#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Chat CLI for Matryoshka (gate-only) and plain HF models.

Examples
--------
# Matryoshka (gate head lives in the model folder)
python scripts/chat_cli.py --model output/r1_gate_only_run

# Plain HF (no gate) + force a budget of 256 thinking tokens
python scripts/chat_cli.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --force_budget 256

# Override budgets list (used if a gate file is present)
python scripts/chat_cli.py --model output/r1_gate_only_run --budgets "0,4,16,64,256,1024"

Notes
-----
• If a gating head file is found (e.g., gating_head_final.pt), the CLI will default
  to using it (unless --force_budget >= 0 is provided).
• If the saved gating-head "anchor" does not match the engine’s runtime anchor,
  a warning is shown (mismatch can hurt budget selection quality).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Dict, Any

import fire

# Import the inference engine and the budget parser
from matryoshka_infer import MatryoshkaEngine, parse_budgets_arg


def _fmt_probs(budgets, probs):
    if not probs:
        return "n/a"
    parts = []
    for i, p in enumerate(probs):
        b = budgets[i]
        parts.append(f"{b}:{p*100:.1f}%")
    return ", ".join(parts)


def _gate_meta(engine: MatryoshkaEngine) -> Dict[str, Any]:
    """Best-effort extraction of gate metadata for display."""
    meta = {
        "loaded_gate": getattr(engine, "loaded_gate", False),
        "budgets": getattr(engine, "BUDGETS", []),
        "engine_anchor": getattr(engine, "gate_anchor", None),
        "head_anchor": None,
        "head_model": None,
        "head_path": getattr(engine, "gating_head_path", None),
    }
    # Try to read whatever the engine exposes
    gh = getattr(engine, "gate_meta", None)
    if isinstance(gh, dict):
        meta["head_anchor"] = gh.get("anchor", None)
        meta["head_model"] = gh.get("model_name_or_path", None)
    else:
        # Some variants store anchor directly
        ha = getattr(engine, "gating_head_anchor", None)
        if ha is not None:
            meta["head_anchor"] = ha
    return meta


def chat(
    model: str,
    gating_head: Optional[str] = None,
    budgets: str = "",  # empty → use saved budgets from head if present
    max_length: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    force_budget: int = -1,  # -1 → auto (use gate if loaded)
    force_close_bias: float = 12.0,  # logit boost for </think> at budget
    seed: Optional[int] = 42,
    use_bf16: bool = True,
):
    """
    Simple terminal chat loop.

    Args:
        model: HF id or local path to base model (or Matryoshka output dir).
        gating_head: Optional path to a gating head .pt file. If None, the
            engine tries to discover one under `model` (e.g., gating_head_final.pt).
        budgets: Comma/JSON list of budgets. If empty and a head is found,
            the head's budgets are used.
        force_budget: -1 → use gate if available; >=0 → ignore gate and force B.
    """
    print("Loading Matryoshka engine...")
    engine = MatryoshkaEngine(
        model_name_or_path=model,
        gating_head_path=gating_head,
        use_bf16=use_bf16,
        max_length=max_length,
        budgets=budgets,  # engine will parse/merge with head metadata
        seed=seed,
    )

    meta = _gate_meta(engine)
    bud = engine.BUDGETS if engine.BUDGETS else []
    print(f"Model: {model}")
    if meta["loaded_gate"]:
        print("✅ Gating head found and loaded.")
        print(f"   • Budgets: {bud}")
        print(f"   • Runtime anchor: {meta['engine_anchor']}")
        print(f"   • Head anchor   : {meta['head_anchor']}")
        if meta["head_model"]:
            print(f"   • Head trained on: {meta['head_model']}")
        if (
            meta["engine_anchor"]
            and meta["head_anchor"]
            and meta["engine_anchor"] != meta["head_anchor"]
        ):
            print(
                "⚠️  Anchor mismatch between engine and gating head. "
                "Selection quality may degrade. Consider retraining or aligning anchors."
            )
    else:
        print("ℹ️  No gating head detected — running as a plain HF model.")
        if force_budget is None or force_budget < 0:
            print("   (Tip: pass --force_budget N to cap reasoning tokens.)")

    # Human-friendly defaults display
    if force_budget is not None and force_budget >= 0:
        print(f"⚙️  Forced budget: {force_budget}")
    else:
        print("🤖 Budget source: gate (if present), otherwise unlimited.")

    print("\nMatryoshka Chat — type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            user = input("\nUser ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user:
            continue

        t0 = time.time()
        res = engine.generate(
            user_query=user,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            force_budget=(
                force_budget
                if (force_budget is not None and force_budget >= 0)
                else None
            ),
            force_close_bias=force_close_bias,
        )
        dt = time.time() - t0

        # Report gate info (if any)
        bi = res.get("budget_info", {})
        src = bi.get("source", "none")
        if src == "gate":
            probs = bi.get("probs") or []
            print(f"   ✅ Gate anchor: {bi.get('anchor')}")
            print(
                f"   📊 Selected budget: {bi.get('selected_budget')}  (probs: {_fmt_probs(engine.BUDGETS, probs)})"
            )
        elif src == "forced":
            print(f"   ⚙️  Forced budget: {bi.get('selected_budget')}")
        else:
            print("   ℹ️  No budget selection (plain chat).")

        closure = (
            "natural"
            if res.get("closed_naturally", False)
            else ("forced" if bi.get("selected_budget") is not None else "n/a")
        )
        print(
            f"   ⏱️  {res.get('generated_tokens', 0)} tokens in {dt:.2f}s — closure: {closure}"
        )

        # Print the assistant text (separate reasoning & answer if available)
        reasoning = res.get("reasoning", "")
        answer = res.get("answer", "")
        print("\n🤖 Assistant\n")
        if reasoning:
            print(f"<think>{reasoning}</think>\n")
        print(answer)


if __name__ == "__main__":
    # Fire provides a clean CLI: python scripts/chat_cli.py --model ...
    fire.Fire(chat)
