#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matryoshka Inference Utilities

Supports:
  • Plain HuggingFace chat models (no gating).
  • Matryoshka gate-only models (a saved gating_head_*.pt alongside the base LM).
    - Selects a global reasoning budget via the gate head.
    - Forces </think> by biasing its logit once the budget is consumed.
    - If the model naturally emits </think> earlier, no forcing is applied.

No FSM is used. We rely only on a gentle logits processor.

Author: Jun Kim (2025)
License: MIT
"""
from __future__ import annotations

import os
import re
import ast
import json
import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    GenerationConfig,
)


# ------------------------------ helpers ------------------------------ #


def parse_budgets_arg(
    budgets: Any,
    max_length: int,
    default=(0, 4, 16, 64, 256, 1024, 4096),
) -> List[int]:
    """
    Robust parsing for budgets lists.
      • accepts CSV, JSON, tuples, lists, "0,256,-1", etc.
      • negative numbers map to max_length (treated as 'unlimited').
      • returns sorted unique non-negative ints and ensures 0 is present.
    """
    if budgets is None or (isinstance(budgets, str) and budgets.strip() == ""):
        arr = list(default)
    elif isinstance(budgets, (list, tuple)):
        arr = list(budgets)
    else:
        s = str(budgets).strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            try:
                lit = ast.literal_eval(s)
                arr = (
                    list(lit)
                    if isinstance(lit, (list, tuple))
                    else [int(x) for x in re.findall(r"-?\d+", s)]
                )
            except Exception:
                arr = [int(x) for x in re.findall(r"-?\d+", s)]
        else:
            arr = [int(x) for x in re.findall(r"-?\d+", s)]

    out: List[int] = []
    for x in arr:
        if x is None:
            continue
        xi = int(x)
        if xi < 0:
            xi = max_length
        xi = min(max_length, max(0, xi))
        out.append(xi)

    out = sorted(set(out))
    if 0 not in out:
        out = [0] + out
    if len(out) == 0:
        out = list(default)
    return out


def ensure_think_tokens(
    tokenizer: AutoTokenizer, model: AutoModelForCausalLM
) -> Tuple[int, int]:
    """
    Make sure <think> and </think> exist; resize embeddings if we add them.
    Returns (think_id, cthink_id).
    """
    newly_added = []
    for t in ["<think>", "</think>"]:
        if t not in tokenizer.get_vocab():
            newly_added.append(t)
    if newly_added:
        tokenizer.add_tokens(newly_added)
        model.resize_token_embeddings(len(tokenizer))
    return (
        tokenizer.convert_tokens_to_ids("<think>"),
        tokenizer.convert_tokens_to_ids("</think>"),
    )


def find_latest_gate_file(model_path: str) -> Optional[str]:
    """
    Look for a gate head snapshot in the model folder.
    Preference: gating_head_final.pt, else the numerically largest step.
    """
    path = Path(model_path)
    cand_final = path / "gating_head_final.pt"
    if cand_final.exists():
        return str(cand_final)
    cands = sorted(glob.glob(str(path / "gating_head_*.pt")))
    if not cands:
        return None

    # Pick the one with the largest trailing step number.
    def step_num(p: str) -> int:
        m = re.search(r"gating_head_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    cands.sort(key=step_num, reverse=True)
    return cands[0]


def build_chat_prompt_with_think(
    tok: AutoTokenizer,
    user: str,
    system: str = "You are a helpful reasoning assistant. Think step-by-step, then answer.",
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Build a chat prompt where the assistant message is started with '<think>'.
    This ensures generation begins inside the reasoning segment.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "<think>"},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return enc.input_ids, enc.attention_mask


def build_chat_prompt_plain(
    tok: AutoTokenizer,
    user: str,
    system: str = "You are a helpful assistant.",
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Standard chat prompt expecting generation to start after assistant prefix.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return enc.input_ids, enc.attention_mask


def _find_token_index(seq: torch.LongTensor, token_id: int) -> int:
    idx = (seq == token_id).nonzero(as_tuple=False)
    return int(idx[-1].item()) if idx.numel() > 0 else -1


# ------------------------------ Gate loader ----------------------------- #


class GateMLPLoaded(nn.Module):
    """
    A 2-layer MLP instantiated to exactly match a saved state_dict
    (so we don't assume hidden_mult). We read shapes from the state dict.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], dtype: torch.dtype):
        super().__init__()
        # Try the common key pattern from training: net.0 (Linear), net.3 (Linear)
        # Fallback: find two Linear layers in order.
        # Expected keys:
        #   'net.0.weight' (Hid, In), 'net.0.bias' (Hid)
        #   'net.3.weight' (Out, Hid), 'net.3.bias' (Out)
        try:
            w1 = state_dict["net.0.weight"]
            b1 = state_dict["net.0.bias"]
            w2 = state_dict["net.3.weight"]
            b2 = state_dict["net.3.bias"]
        except KeyError:
            # Attempt a more flexible discovery
            # Find first *.0.weight and last *.weight
            weight_keys = [k for k in state_dict.keys() if k.endswith(".weight")]
            bias_keys = [k for k in state_dict.keys() if k.endswith(".bias")]
            weight_keys.sort()
            bias_keys.sort()
            assert (
                len(weight_keys) >= 2 and len(bias_keys) >= 2
            ), "Unrecognized gate state dict format"
            w1 = state_dict[weight_keys[0]]
            b1 = state_dict[bias_keys[0]]
            w2 = state_dict[weight_keys[-1]]
            b2 = state_dict[bias_keys[-1]]

        in_dim = w1.shape[1]
        hid_dim = w1.shape[0]
        out_dim = w2.shape[0]

        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.act = nn.SiLU()

        # Assign weights
        with torch.no_grad():
            self.lin1.weight.copy_(w1.to(dtype))
            self.lin1.bias.copy_(b1.to(dtype))
            self.lin2.weight.copy_(w2.to(dtype))
            self.lin2.bias.copy_(b2.to(dtype))

        self.lin1.to(dtype=dtype)
        self.lin2.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype match
        x = x.to(self.lin1.weight.dtype)
        return self.lin2(self.act(self.lin1(x)))


@dataclass
class LoadedGate:
    module: nn.Module
    budgets: List[int]
    anchor: str  # 'prethink' | 'think' | 'prompt_mean'


def try_load_gate(
    model_or_path: str, model: AutoModelForCausalLM
) -> Optional[LoadedGate]:
    """
    Try to load a saved gating head snapshot from model_or_path (folder or file).
    Returns None if not found.
    """
    gate_file = None
    if isinstance(model_or_path, str) and os.path.isdir(model_or_path):
        gate_file = find_latest_gate_file(model_or_path)
    elif isinstance(model_or_path, str) and os.path.isfile(model_or_path):
        gate_file = model_or_path

    if gate_file is None:
        return None

    blob = torch.load(gate_file, map_location="cpu")
    sd = blob.get("state_dict", None)
    budgets = blob.get("budgets", None)
    anchor = blob.get("anchor", "prethink")
    if sd is None or budgets is None:
        return None

    param_dtype = next(model.parameters()).dtype
    module = GateMLPLoaded(sd, dtype=param_dtype)
    module.eval()
    return LoadedGate(module=module, budgets=budgets, anchor=anchor)


# --------------------------- Logits processor --------------------------- #


class CloseThinkAtBudget(LogitsProcessor):
    """
    When we've generated `budget` tokens after the last <think> token (and we
    haven't seen </think> yet), either add a positive bias to the </think> logit
    or force it with very high probability.

    This processor assumes *one* sample (batch size 1); chat_cli uses it that way.

    Args:
        force_probability: If True, when budget is reached, force </think> by setting
                          its logit to 100.0 and all others to -inf. If False, use
                          the original bias-based approach.
    """

    def __init__(
        self,
        think_id: int,
        cthink_id: int,
        budget: int,
        bias: float = 12.0,
        force_probability: bool = False,
    ):
        self.think_id = int(think_id)
        self.cthink_id = int(cthink_id)
        self.budget = int(max(0, budget))
        self.bias = float(bias)
        self.force_probability = force_probability

    def _count_after_think(self, input_ids: torch.LongTensor) -> Tuple[int, bool]:
        """
        Returns (count_since_think, already_closed).
        """
        seq = input_ids[0]  # [T]
        last_think = (seq == self.think_id).nonzero(as_tuple=False)
        if last_think.numel() == 0:
            return 0, False
        t_idx = int(last_think[-1].item())
        # If </think> already present after this <think>, we are closed
        post = seq[t_idx + 1 :]
        closed = (post == self.cthink_id).any().item()
        # Count how many tokens we have generated after <think> (until now)
        count = int((seq.shape[0] - 1) - t_idx)  # tokens emitted after <think>
        return max(0, count), bool(closed)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Single sequence only (interactive chat); extend if you need batch
        c, closed = self._count_after_think(input_ids)
        if closed:
            return scores
        if c >= self.budget:
            # Encourage closure now
            if 0 <= self.cthink_id < scores.shape[-1]:
                if self.force_probability:
                    # Force </think> with very high probability by setting its logit very high
                    # and suppressing all other tokens
                    scores[:, :] = -float("inf")  # Set all tokens to -inf
                    scores[:, self.cthink_id] = 100.0  # Set </think> to very high value
                else:
                    # Original behavior: add bias
                    scores[:, self.cthink_id] = scores[:, self.cthink_id] + self.bias
        return scores


# ------------------------------ Engine --------------------------------- #


class MatryoshkaEngine:
    """
    Unified wrapper to chat with HF or Matryoshka-gated models.

    If a gating head is found next to the model (gating_head_*.pt), it will be used
    to pick a budget from its saved 'budgets'. Otherwise you can supply --force_budget
    or let the model free-generate without <think> scaffolding.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        use_bf16: bool = True,
        max_length: int = 4096,
        budgets: Optional[str] = None,  # optional override for non-saved gates
        gating_head_path: Optional[str] = None,
        system_prompt_reasoning: str = "You are a helpful reasoning assistant. Think step-by-step, then answer.",
        system_prompt_plain: str = "You are a helpful assistant.",
        seed: Optional[int] = None,
    ):
        self.model_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
        )
        self.max_length = int(max_length)
        self.system_prompt_reasoning = system_prompt_reasoning
        self.system_prompt_plain = system_prompt_plain

        # Load model + tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        # Make sure we have <think> tokens and remember ids
        self.think_id, self.cthink_id = ensure_think_tokens(self.tok, self.model)

        # Gating head (optional)
        self.loaded_gate: Optional[LoadedGate] = None
        if gating_head_path is not None:
            self.loaded_gate = try_load_gate(gating_head_path, self.model)
        if self.loaded_gate is None:
            self.loaded_gate = try_load_gate(model_name_or_path, self.model)

        # Move gating head to same device as main model
        if self.loaded_gate is not None:
            self.loaded_gate.module = self.loaded_gate.module.to(self.device)

        # Budgets
        if self.loaded_gate is not None:
            self.BUDGETS = list(self.loaded_gate.budgets)
        else:
            self.BUDGETS = parse_budgets_arg(budgets, max_length=self.max_length)

        # Seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # ---- gating ---- #

    @torch.no_grad()
    def _pool_gate_feature(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, anchor: str
    ) -> torch.Tensor:
        """
        Run the base LM encoder to get the last hidden; pool according to anchor.
        """
        out = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        last_h = out.hidden_states[-1]  # [1, L, H]
        ids = input_ids[0].to(self.device)

        # Locate <think>
        tpos = (ids == self.think_id).nonzero(as_tuple=False)
        r0 = int(tpos[-1].item()) if tpos.numel() > 0 else 0

        if anchor == "think":
            # one-hot at <think>
            mask = torch.zeros_like(ids, dtype=last_h.dtype, device=self.device)
            if 0 <= r0 < mask.numel():
                mask[r0] = 1.0
        elif anchor == "prethink":
            mask = torch.zeros_like(ids, dtype=last_h.dtype, device=self.device)
            anchor_idx = max(0, r0 - 1)
            mask[anchor_idx] = 1.0
        else:  # prompt_mean
            mask = torch.zeros_like(ids, dtype=last_h.dtype, device=self.device)
            if r0 > 0:
                mask[:r0] = 1.0
            else:
                mask[0] = 1.0

        denom = mask.sum().clamp_min(1e-6)
        feat = (last_h[0] * mask.unsqueeze(-1)).sum(dim=0) / denom  # [H]
        return feat.unsqueeze(0)  # [1, H]

    @torch.no_grad()
    def select_budget(
        self, user_query: str, force_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute gate probabilities on the prompt (with assistant '<think>') and select a budget.
        If no gate is available or force_budget is provided, obey that.
        """
        # If user forces a budget, comply.
        if force_budget is not None and force_budget >= 0:
            b = int(force_budget)
            # Snap to the nearest available budget in BUDGETS
            chosen = min(self.BUDGETS, key=lambda x: abs(x - b)) if self.BUDGETS else b
            return {
                "selected_budget": chosen,
                "probs": None,
                "anchor": "forced",
                "source": "forced",
            }

        # If no gate available, no selection
        if self.loaded_gate is None:
            return {
                "selected_budget": None,
                "probs": None,
                "anchor": None,
                "source": "none",
            }

        # Build prompt with assistant "<think>" to create the prethink anchor
        input_ids, attention_mask = build_chat_prompt_with_think(
            self.tok, user_query, system=self.system_prompt_reasoning
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Pool feature and run gate head
        feat = self._pool_gate_feature(
            input_ids, attention_mask, anchor=self.loaded_gate.anchor
        )  # [1,H]
        logits = self.loaded_gate.module(feat).float()  # [1,D]
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        # Choose argmax budget
        idx = int(torch.argmax(logits, dim=-1).item())
        selected = int(self.BUDGETS[idx])

        return {
            "selected_budget": selected,
            "probs": probs,
            "anchor": self.loaded_gate.anchor,
            "source": "gate",
        }

    # ---- generation ---- #

    @torch.no_grad()
    def generate(
        self,
        user_query: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 1024,
        force_budget: Optional[int] = None,
        force_close_bias: float = 12.0,
        force_close_probability: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a response. If a budget is selected (gate or force), we:
          • prompt with '<think>' at assistant start,
          • bias '</think>' once the budget is consumed (gently with bias, or aggressively with force_close_probability).
        Otherwise, we do plain chat prompting.

        Args:
            force_close_probability: If True, when budget is reached, force </think> with very high probability
                                   by setting its logit to 100.0 and all others to -inf. If False, use the
                                   original bias-based approach.
        """
        budget_info = self.select_budget(user_query, force_budget=force_budget)
        budget = budget_info["selected_budget"]

        if budget is not None:
            # Reasoning-style prompt with <think>
            input_ids, attention_mask = build_chat_prompt_with_think(
                self.tok, user_query, system=self.system_prompt_reasoning
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            processors = LogitsProcessorList(
                [
                    CloseThinkAtBudget(
                        think_id=self.think_id,
                        cthink_id=self.cthink_id,
                        budget=int(budget),
                        bias=float(force_close_bias),
                        force_probability=force_close_probability,
                    )
                ]
            )
        else:
            # Plain chat
            input_ids, attention_mask = build_chat_prompt_plain(
                self.tok, user_query, system=self.system_prompt_plain
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            processors = LogitsProcessorList([])

        gen_cfg = GenerationConfig(
            do_sample=(temperature > 0 and temperature != 1.0) or (top_p < 1.0),
            temperature=max(0.0, float(temperature)),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )

        out_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            logits_processor=processors,
        )
        # Decode only the newly generated portion
        gen_text = self.tok.decode(
            out_ids[0][input_ids.shape[1] :], skip_special_tokens=False
        )

        # Parse reasoning and answer spans for display
        full_text = self.tok.decode(out_ids[0], skip_special_tokens=False)
        think_s = full_text.find("<think>")
        think_e = full_text.find("</think>")
        reasoning = ""
        answer = ""
        if think_s >= 0 and think_e > think_s:
            reasoning = full_text[think_s + len("<think>") : think_e]
            answer = full_text[think_e + len("</think>") :]
        else:
            answer = full_text[input_ids.shape[1] :]

        # Closure heuristic
        closed_naturally = think_e >= 0

        return {
            "text": full_text,
            "reasoning": reasoning.strip(),
            "answer": answer.strip(),
            "budget_info": budget_info,
            "closed_naturally": closed_naturally,
            "generated_tokens": out_ids.shape[1] - input_ids.shape[1],
        }
