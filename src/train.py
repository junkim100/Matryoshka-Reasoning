#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025
# Licensed under the MIT License
"""
Matryoshka Reasoning — Gate-Only Trainer (fixed oracle)
Freeze the LM; train only a gating head that selects a global reasoning budget.
The oracle now measures *answer-only* NLL conditioned on the first B rationale tokens.

Key changes vs. earlier version
-------------------------------
• For each budget b: re-encode with the *over-budget* <think> tokens masked out
  in attention, and compute CE on the *answer span only*. This makes budgets
  really affect the answer loss.
• Soft-oracle is built from those answer CEs plus a light absolute length penalty.
• Keep LM frozen; only the gate MLP is trained. Safe with ZeRO-3.

Quick start
-----------
deepspeed --include localhost:0 src/train.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --train_file data/train.jsonl --valid_file data/val.jsonl \
  --output_dir out/gate_only_deepseek_r1 \
  --budgets "0,4,16,64,256,1024" --max_length 4096
"""

from __future__ import annotations

import os
import re
import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


# ------------------------------ Logging --------------------------------- #


def is_rank0() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def get_logger(level: str = "INFO") -> logging.Logger:
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(format=fmt, level=getattr(logging, level.upper(), logging.INFO))
    return logging.getLogger("train")


# ------------------------------- Data ----------------------------------- #

SYSTEM_MSG = "You are a helpful reasoning assistant. Think step-by-step, then answer."


class JsonlDS(Dataset):
    """Each line: {"question": str, "solution": str, "answer": str}"""

    def __init__(self, path: str):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    obj = json.loads(l)
                except Exception:
                    continue
                if isinstance(obj, dict) and {"question", "solution", "answer"} <= set(
                    obj.keys()
                ):
                    rows.append(obj)
        if not rows:
            raise ValueError(f"Empty or invalid dataset: {path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def parse_budgets_arg(
    budgets: Any,
    max_length: int,
    default=(0, 4, 16, 64, 256, 1024),
) -> List[int]:
    """
    Parse budgets from CSV / JSON / tuple / list.
    • negatives -> max_length
    • clamp to [0, max_length]
    • dedupe + sort
    • ensure 0 is present
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


class ReasoningCollator:
    """
    Tokenizes chat; records indices of <think>, </think>, answer span, and end-of-seq.
    Also builds a gate anchor mask.

    Output:
        input_ids      : [B, L]
        attention_mask : [B, L]
        gate_mask      : [B, L]
        info: {
          "think_idx"  : [B]    index of "<think>" token in input_ids, or answer start fallback
          "cthink_idx" : [B]    index of "</think>", or -1 if missing
          "stop_idx"   : [B]    index of the last EOS/EOT in visible region
          "ans_span"   : [B,2]  [start, end) inclusive-exclusive in input_ids
        }
    """

    def __init__(
        self, tok: AutoTokenizer, max_length: int, gate_anchor: str = "prethink"
    ):
        self.tok = tok
        self.max_length = max_length
        self.gate_anchor = gate_anchor  # "prethink" | "think" | "prompt_mean"

        for t in ["<think>", "</think>"]:
            if t not in self.tok.get_vocab():
                self.tok.add_tokens([t])

        self.eos_id = self.tok.eos_token_id
        self.eot_id = getattr(self.tok, "eot_token_id", None)
        self.t_think = self.tok.convert_tokens_to_ids("<think>")
        self.t_cthink = self.tok.convert_tokens_to_ids("</think>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, attn, gate_masks = [], [], []
        think_idx, cthink_idx, stop_idx, ans_spans = [], [], [], []

        for ex in batch:
            q, sol, ans = ex["question"], ex["solution"], ex["answer"]

            # Ensure rationale & answer blocks exist; be robust to datasets lacking explicit tags.
            if "<think>" not in sol and "\\boxed" in sol:
                sol = "<think>" + sol.replace("\\boxed", "</think>\n\n\\boxed")
            elif "<think>" not in sol:
                sol = f"<think>{sol}</think>\n\n\\boxed{{{ans}}}"

            chat = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": q},
                {"role": "assistant", "content": sol + self.tok.eos_token},
            ]
            txt = self.tok.apply_chat_template(chat, tokenize=False)
            enc = self.tok(
                txt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=False,
                return_tensors="pt",
            )
            ids = enc.input_ids[0]
            mask = enc.attention_mask[0]
            L = int(mask.sum().item())

            # Stop index: last EOS/EOT in visible region
            vis = ids[:L]
            eom = vis == self.eos_id
            if self.eot_id is not None:
                eom = eom | (vis == self.eot_id)
            eos_pos = eom.nonzero(as_tuple=False)
            sid = int(eos_pos[-1].item()) if eos_pos.numel() > 0 else (L - 1)

            # Indices of <think> and </think>
            tpos = (ids == self.t_think).nonzero(as_tuple=True)[0]
            cpos = (ids == self.t_cthink).nonzero(as_tuple=True)[0]
            found_t = len(tpos) > 0
            found_c = len(cpos) > 0
            r0 = int(tpos[0]) if found_t else 0
            c0 = int(cpos[0]) if found_c else -1

            # Locate answer span (robust)
            raw_norm = str(ans).strip().replace("−", "-")
            patterns = [
                self.tok.encode(f"\\boxed{{{raw_norm}}}", add_special_tokens=False),
                self.tok.encode(raw_norm, add_special_tokens=False),
                self.tok.encode(" " + raw_norm, add_special_tokens=False),
            ]

            def find_span(ids_t, pat, start, end):
                if not pat:
                    return -1, -1
                seq = ids_t[start:end].tolist()
                m = len(pat)
                for i in range(len(seq) - m + 1):
                    if seq[i : i + m] == pat:
                        return start + i, start + i + m
                return -1, -1

            a_st, a_en = (-1, -1)
            for pat in patterns:
                a_st, a_en = find_span(ids, pat, r0, L)
                if a_st >= 0:
                    break
            if a_st < 0:
                a_en = sid
                a_st = max(r0, sid - 1)

            if not found_t:
                r0 = a_st  # fallback: anchor at answer start

            # Gate mask by anchor
            gm = torch.zeros_like(ids, dtype=torch.float32)
            if self.gate_anchor == "think":
                if 0 <= r0 < len(gm):
                    gm[r0] = 1.0
            elif self.gate_anchor == "prethink":
                anchor = max(0, r0 - 1)
                gm[anchor] = 1.0
            else:  # prompt_mean
                gm[: max(1, r0)] = 1.0

            input_ids.append(ids)
            attn.append(mask)
            gate_masks.append(gm)
            think_idx.append(r0)
            cthink_idx.append(c0)
            stop_idx.append(sid)
            ans_spans.append([a_st, a_en])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "gate_mask": torch.stack(gate_masks),
            "info": {
                "think_idx": torch.tensor(think_idx, dtype=torch.long),
                "cthink_idx": torch.tensor(cthink_idx, dtype=torch.long),
                "stop_idx": torch.tensor(stop_idx, dtype=torch.long),
                "ans_span": torch.tensor(ans_spans, dtype=torch.long),
            },
        }


# --------------------------- Gate head ---------------------------------- #


@dataclass
class GateConfig:
    hidden_mult: float = 2.0
    dropout: float = 0.0
    temperature: float = 1.0  # divides logits inside forward()


class GateMLP(nn.Module):
    """Two-layer MLP with SiLU."""

    def __init__(self, in_dim: int, out_dim: int, cfg: GateConfig, dtype: torch.dtype):
        super().__init__()
        hid = max(64, int(in_dim * cfg.hidden_mult))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid, bias=True),
            nn.SiLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(hid, out_dim, bias=True),
        )
        self.net.to(dtype=dtype)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(next(self.net[0].parameters()).dtype)
        return self.net(x) / max(1e-6, self.cfg.temperature)


# --------------------- Gated wrapper (LM frozen) ------------------------ #


class GateOnlyWrapper(nn.Module):
    """
    Wrap a HF causal LM but freeze it. We still need:
      • last hidden state (no grads) to pool gate features
      • token-wise NLL computed via lm_head
    Only the gate MLP has trainable parameters.
    """

    def __init__(
        self, base: AutoModelForCausalLM, num_classes: int, gate_cfg: GateConfig
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        hidden = getattr(self.base.config, "hidden_size", None) or getattr(
            self.base.config, "n_embd", None
        )
        if hidden is None:
            raise ValueError("Cannot infer hidden size from model.config")

        # Model/lm_head handles (generic across HF causal LMs)
        self.transformer = getattr(self.base, "model", None)
        if self.transformer is None:
            raise ValueError("Expected base.model to exist (HF causal LM).")
        self.norm = getattr(self.transformer, "norm", None)
        self.lm_head = self.base.get_output_embeddings()

        # Gate head dtype matches base params to avoid dtype mismatch in DS
        param_dtype = next(self.base.parameters()).dtype
        self.gate = GateMLP(
            in_dim=hidden, out_dim=num_classes, cfg=gate_cfg, dtype=param_dtype
        )

    @torch.no_grad()
    def _encode(self, input_ids, attention_mask):
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_h = out.last_hidden_state  # [B,L,H]
        h_for_lm = self.norm(last_h) if self.norm is not None else last_h
        return last_h, h_for_lm

    def forward(self, input_ids=None, attention_mask=None, gate_mask=None):
        # Encode without grads
        last_h, h_for_lm = self._encode(input_ids, attention_mask)

        # Pool gate feature with normalized mask
        if gate_mask is None:
            vis = attention_mask.to(dtype=last_h.dtype)
            denom = vis.sum(dim=1, keepdim=True).clamp_min(1.0)
            feat = (last_h * vis.unsqueeze(-1)).sum(dim=1) / denom
        else:
            gm = gate_mask.to(dtype=last_h.dtype)
            denom = gm.sum(dim=1, keepdim=True).clamp_min(1e-6)
            feat = (last_h * gm.unsqueeze(-1)).sum(dim=1) / denom

        # Gate logits (trainable)
        gate_logits = self.gate(feat)  # [B, D]

        return {"last_h": last_h, "h_for_lm": h_for_lm, "gate_logits": gate_logits}


# ---------------------- Loss/NLL utilities ------------------------------ #


@torch.no_grad()
def per_token_nll_from_hidden(
    h_for_lm: torch.Tensor,  # [B, L, H]
    lm_head: nn.Module,  # Linear(H->V)
    input_ids: torch.Tensor,  # [B, L]
    chunk: int = 256,
) -> torch.Tensor:
    """
    Compute token-level NLL for standard teacher-forcing:
    targets = input_ids[:, 1:] aligned to logits over positions [:, :-1]
    Returns: nll_tok: [B, L-1] in float32
    """
    B, L, H = h_for_lm.shape
    T = L - 1
    nll = torch.empty((B, T), dtype=torch.float32, device=h_for_lm.device)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        logits_chunk = lm_head(h_for_lm[:, s:e, :])  # [B, e-s, V] (bf16)
        targets = input_ids[:, s + 1 : e + 1]  # [B, e-s]
        ce = F.cross_entropy(
            logits_chunk.transpose(1, 2),  # [B, V, e-s]
            targets,
            reduction="none",
        ).to(
            torch.float32
        )  # [B, e-s]
        nll[:, s:e] = ce
        del logits_chunk, targets, ce
    return nll


# -------------------------- DeepSpeed helper ---------------------------- #


def patch_ds_cfg(
    cfg_path: str, lr: float, total_steps: int, micro_bs: Optional[int]
) -> str:
    abs_path = os.path.abspath(cfg_path)
    with open(abs_path, "r") as f:
        cfg = json.load(f)

    cfg.setdefault("optimizer", {})
    cfg["optimizer"].setdefault("type", "AdamW")
    cfg["optimizer"].setdefault("params", {})
    cfg["optimizer"]["params"]["lr"] = lr

    if "scheduler" not in cfg or not cfg["scheduler"].get("type"):
        cfg["scheduler"] = {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": max(1, int(0.05 * total_steps)),
                "total_num_steps": total_steps,
            },
        }

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_acc = int(cfg.get("gradient_accumulation_steps", 1))
    if micro_bs is not None:
        cfg["train_micro_batch_size_per_gpu"] = int(micro_bs)
        cfg["train_batch_size"] = int(micro_bs) * grad_acc * world_size
    else:
        micro = int(cfg.get("train_micro_batch_size_per_gpu", 1))
        cfg["train_batch_size"] = micro * grad_acc * world_size

    patched = abs_path + ".patched"
    if is_rank0():
        tmp = patched + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, patched)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if not os.path.exists(patched):
        raise RuntimeError(f"Patched DeepSpeed config not found: {patched}")
    return patched


# ------------------------------- Train ---------------------------------- #


def train(
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    valid_file: str = "",
    # sequence / budgets
    max_length: int = 4096,
    budgets: str = "0,4,16,64,256,1024",
    ce_chunk_tokens: int = 256,
    # gating
    use_gating: bool = True,  # for API parity; always True here
    gate_anchor: str = "prethink",  # "prethink" | "think" | "prompt_mean"
    gate_temperature: float = 1.0,
    gate_hidden_mult: float = 2.0,
    gate_dropout: float = 0.0,
    gate_entropy_penalty: float = 0.02,  # <-- default tuned
    gate_length_penalty: float = 0.05,  # <-- default tuned
    oracle_temperature: float = 0.25,  # <-- default tuned (sharper targets)
    hard_oracle_weight: float = 0.0,  # optional CE to argmin(b) (off)
    exp_loss_weight: float = 0.0,  # optional E_p[b][loss_b] (off)
    # optimization / ds
    learning_rate: float = 1e-4,  # gate only; a bit higher is fine
    num_train_epochs: int = 2,
    per_device_train_batch_size: Optional[int] = None,
    logging_steps: int = 20,
    save_steps: int = 1000,
    eval_steps: int = 200,
    deepspeed_config: str = "configs/ds_config.json",
    wandb_project: str = "",
    bf16: bool = True,
    max_eval_batches: int = 0,
    log_level: str = "INFO",
    **_unused,
):
    # Pin device per rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    log = get_logger(log_level)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"

    # Budgets
    BUDGETS = parse_budgets_arg(budgets, max_length=max_length)
    D = len(BUDGETS)
    if is_rank0():
        log.info(f"> Global budgets: {BUDGETS} (classes={D}, max_length={max_length})")

    # Distributed
    if (not torch.distributed.is_initialized()) and int(
        os.environ.get("WORLD_SIZE", "1")
    ) > 1:
        deepspeed.init_distributed()

    # Data
    train_ds = JsonlDS(train_file)
    sampler = (
        DistributedSampler(train_ds) if torch.distributed.is_initialized() else None
    )
    collator = ReasoningCollator(tok, max_length=max_length, gate_anchor=gate_anchor)

    dl = DataLoader(
        train_ds,
        batch_size=per_device_train_batch_size or 1,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    val_dl = None
    if valid_file:
        val_ds = JsonlDS(valid_file)
        val_sampler = (
            DistributedSampler(val_ds, shuffle=False)
            if torch.distributed.is_initialized()
            else None
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=per_device_train_batch_size or 1,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

    total_steps = len(dl) * num_train_epochs
    ds_cfg_patched = patch_ds_cfg(
        deepspeed_config, learning_rate, total_steps, per_device_train_batch_size
    )

    # Model
    dtype = torch.bfloat16 if bf16 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        use_cache=False,
    )
    if len(tok) != base.config.vocab_size:
        # only happens if we added new special tokens
        base.resize_token_embeddings(len(tok))

    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()  # OK even if LM frozen (keeps memory lower)

    gate_cfg = GateConfig(
        hidden_mult=gate_hidden_mult, dropout=gate_dropout, temperature=gate_temperature
    )
    model = GateOnlyWrapper(base, num_classes=D, gate_cfg=gate_cfg)

    # Only gate params are trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_rank0():
        n_gate = sum(p.numel() for p in trainable)
        log.info(f"> Trainable parameters (gate only): {n_gate/1e6:.3f} M")

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable,  # only gate params go to optimizer
        config=ds_cfg_patched,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if wandb_project and is_rank0() and wandb is not None:
        base_name = Path(model_name_or_path).name
        run_name = f"{base_name}|GATE-ONLY|{gate_anchor}|{D}b"
        wandb.init(project=wandb_project, name=run_name)
        wandb.config.update(
            dict(
                model=model_name_or_path,
                budgets=BUDGETS,
                max_length=max_length,
                ce_chunk_tokens=ce_chunk_tokens,
                gate_anchor=gate_anchor,
                gate_temperature=gate_temperature,
                gate_hidden_mult=gate_hidden_mult,
                gate_dropout=gate_dropout,
                gate_length_penalty=gate_length_penalty,
                oracle_temperature=oracle_temperature,
                hard_oracle_weight=hard_oracle_weight,
                exp_loss_weight=exp_loss_weight,
                lr=learning_rate,
                epochs=num_train_epochs,
            )
        )

    # ---------------- core step function (fixed oracle) ---------------- #

    def step_once(
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_ids = batch["input_ids"].to(engine.device)
        attention_mask = batch["attention_mask"].to(engine.device)
        gate_mask = batch["gate_mask"].to(engine.device)
        info = batch["info"]
        think_idx = info["think_idx"].to(engine.device)
        cthink_idx = info["cthink_idx"].to(engine.device)
        stop_idx = info["stop_idx"].to(engine.device)
        ans_span = info["ans_span"].to(engine.device)

        # Forward (LM frozen) for gate features & logits
        out = engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            gate_mask=gate_mask,
        )
        gate_logits = out["gate_logits"]  # [B,D]

        B, L = input_ids.size()
        T = L - 1
        D_ = len(BUDGETS)
        assert D_ == gate_logits.size(-1)

        # Build CE masks for answer tokens (over CE time indices)
        ans_masks = []
        for i in range(B):
            a_st, a_en = int(ans_span[i, 0].item()), int(ans_span[i, 1].item())
            m = torch.zeros(T, dtype=torch.bool, device=engine.device)
            if a_st >= 1 and a_en > a_st:
                t0 = max(0, a_st - 1)
                t1 = min(T, a_en - 1)  # exclusive
                m[t0:t1] = True
            ans_masks.append(m)
        ans_masks = torch.stack(ans_masks)  # [B,T]

        loss_db = torch.empty((B, D_), dtype=torch.float32, device=engine.device)

        # For each budget: hide *excess* rationale tokens and re-encode (no grad)
        for j, budget in enumerate(BUDGETS):
            # Copy attention; we'll zero out over-budget <think> tokens
            attn_b = attention_mask.clone()

            for i in range(B):
                t_idx = int(think_idx[i].item())  # position of "<think>"
                c_idx = int(cthink_idx[i].item())  # position of "</think>" (or -1)
                s_idx = int(stop_idx[i].item())  # last visible (eos/eot)
                # Range of rationale tokens = (t_idx+1) .. end_reason_pos (inclusive)
                end_reason_pos = (c_idx if c_idx >= 0 else s_idx) - 1
                if end_reason_pos <= t_idx:
                    continue  # no rationale in the window

                # Keep only the first `budget` rationale tokens (after <think>)
                keep_last_pos = min(t_idx + budget, end_reason_pos)

                hide_start = max(keep_last_pos + 1, 0)
                hide_end = max(end_reason_pos + 1, hide_start)  # slice exclusive
                if hide_end > hide_start:
                    attn_b[i, hide_start:hide_end] = (
                        0  # invisible to all subsequent tokens
                    )

            with torch.no_grad():
                out_b = engine.module.transformer(
                    input_ids=input_ids,
                    attention_mask=attn_b,
                    use_cache=False,
                    return_dict=True,
                )
                h_last = out_b.last_hidden_state
                h_lm = (
                    engine.module.norm(h_last)
                    if engine.module.norm is not None
                    else h_last
                )

                nll_tok_b = per_token_nll_from_hidden(
                    h_for_lm=h_lm,
                    lm_head=engine.module.lm_head,
                    input_ids=input_ids,
                    chunk=ce_chunk_tokens,
                )  # [B,T]

                # Answer-only CE under this budget
                num = (nll_tok_b * ans_masks.float()).sum(dim=1)
                den = ans_masks.float().sum(dim=1).clamp_min(1.0)
                loss_db[:, j] = num / den

            # free budget-local tensors
            del attn_b, out_b, h_last, h_lm, nll_tok_b

        # Soft oracle (length-regularized)
        length_cost = (
            torch.tensor(BUDGETS, dtype=torch.float32, device=engine.device)
            / float(max_length)
        ).unsqueeze(0)
        score = -(loss_db + gate_length_penalty * length_cost) / max(
            1e-6, oracle_temperature
        )  # [B,D]
        q = F.softmax(score, dim=-1)  # [B,D]
        hard = (loss_db + gate_length_penalty * length_cost).argmin(dim=-1)  # [B]

        # Gate distribution
        p = F.softmax(gate_logits.float(), dim=-1)
        logp = torch.log(p.clamp_min(1e-8))

        # Loss: CE with soft targets (KL up to constant)
        gate_loss = -(q * logp).sum(dim=-1).mean()

        # Optional additions
        hard_ce = (
            F.cross_entropy(gate_logits.float(), hard)
            if hard_oracle_weight > 0.0
            else torch.tensor(0.0, device=engine.device)
        )
        ent_pen = torch.tensor(0.0, device=engine.device)
        if gate_entropy_penalty != 0.0:
            ent = (-(p * logp)).sum(dim=-1).mean()
            ent_pen = gate_entropy_penalty * ent
        exp_loss = (
            (p * loss_db).sum(dim=-1).mean()
            if exp_loss_weight > 0.0
            else torch.tensor(0.0, device=engine.device)
        )

        total = gate_loss + hard_oracle_weight * hard_ce + ent_pen + exp_loss

        with torch.no_grad():
            oracle_agree = (p.argmax(dim=-1) == hard).float().mean()
            gate_mean = p.mean(dim=0)
            loss_spread = (loss_db.max(dim=1).values - loss_db.min(dim=1).values).mean()
            avg_ans_ce = loss_db.mean()

        stats = {
            "gate_loss": gate_loss.detach(),
            "hard_ce": hard_ce.detach(),
            "entropy_pen": ent_pen.detach(),
            "exp_loss": exp_loss.detach(),
            "oracle_agreement": oracle_agree.detach(),
            "loss_per_budget_mean": loss_db.mean(dim=0).detach(),  # [D]
            "gate_probs_mean": gate_mean.detach(),  # [D]
            "loss_spread": loss_spread.detach(),
            "avg_ans_ce": avg_ans_ce.detach(),
        }
        return total, stats

    @torch.no_grad()
    def evaluate(val_loader):
        if val_loader is None:
            return None
        engine.eval()
        acc = {
            "gate_loss": 0.0,
            "hard_ce": 0.0,
            "entropy_pen": 0.0,
            "exp_loss": 0.0,
            "oracle_agreement": 0.0,
            "loss_spread": 0.0,
            "avg_ans_ce": 0.0,
        }
        loss_budget_sum = torch.zeros(D, dtype=torch.float32, device=engine.device)
        gate_prob_sum = torch.zeros(D, dtype=torch.float32, device=engine.device)
        n = 0

        for bi, batch in enumerate(val_loader):
            total, st = step_once(batch)
            acc["gate_loss"] += float(st["gate_loss"].item())
            acc["hard_ce"] += float(st["hard_ce"].item())
            acc["entropy_pen"] += float(st["entropy_pen"].item())
            acc["exp_loss"] += float(st["exp_loss"].item())
            acc["oracle_agreement"] += float(st["oracle_agreement"].item())
            acc["loss_spread"] += float(st["loss_spread"].item())
            acc["avg_ans_ce"] += float(st["avg_ans_ce"].item())
            loss_budget_sum += st["loss_per_budget_mean"]
            gate_prob_sum += st["gate_probs_mean"]
            n += 1
            if max_eval_batches and (bi + 1) >= max_eval_batches:
                break

        if torch.distributed.is_initialized():
            # Reduce scalars
            for k in acc:
                t = torch.tensor(acc[k], device=engine.device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                acc[k] = float(t.item())
            n_t = torch.tensor(float(n), device=engine.device)
            torch.distributed.all_reduce(n_t, op=torch.distributed.ReduceOp.SUM)
            n = int(n_t.item())
            # Reduce vectors
            torch.distributed.all_reduce(
                loss_budget_sum, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                gate_prob_sum, op=torch.distributed.ReduceOp.SUM
            )

        if n > 0:
            for k in acc:
                acc[k] /= n
        loss_budget_mean = (loss_budget_sum / max(1, n)).detach().tolist()
        gate_prob_mean = (gate_prob_sum / max(1, n)).detach().tolist()

        engine.train()
        return acc, loss_budget_mean, gate_prob_mean

    # ---------------- loop ---------------- #

    pbar = tqdm(total=total_steps, disable=not (tqdm and is_rank0()), unit="step")
    step = 0
    best = float("inf")
    best_step: Optional[int] = None

    for epoch in range(num_train_epochs):
        if isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)
        for batch in dl:
            step += 1

            total_loss, stats = step_once(batch)
            engine.backward(total_loss)
            engine.step()

            if is_rank0() and step % logging_steps == 0:
                log.info(
                    f"step {step}/{total_steps}  "
                    f"gate:{stats['gate_loss']:.4f}  hardCE:{stats['hard_ce']:.4f}  "
                    f"ent:{stats['entropy_pen']:.4f}  spread:{stats['loss_spread']:.3f}  "
                    f"oracle:{stats['oracle_agreement']:.3f}"
                )
                if wandb and wandb.run:
                    wb = {
                        "step": step,
                        "gate/train_gate_loss": float(stats["gate_loss"].item()),
                        "gate/train_hard_ce": float(stats["hard_ce"].item()),
                        "gate/train_entropy_pen": float(stats["entropy_pen"].item()),
                        "gate/train_exp_loss": float(stats["exp_loss"].item()),
                        "gate/train_oracle_agreement": float(
                            stats["oracle_agreement"].item()
                        ),
                        "gate/train_loss_spread": float(stats["loss_spread"].item()),
                        "gate/train_avg_ce": float(stats["avg_ans_ce"].item()),
                    }
                    lpb = stats["loss_per_budget_mean"].detach().cpu().tolist()
                    gp = stats["gate_probs_mean"].detach().cpu().tolist()
                    for i, v in enumerate(lpb):
                        wb[f"gate/train_loss_budget_{BUDGETS[i]}"] = float(v)
                    for i, v in enumerate(gp):
                        wb[f"gate/train_gprob_{BUDGETS[i]}"] = float(v)
                    wandb.log(wb)

            # save
            if step % save_steps == 0 or step >= total_steps:
                ckpt_dir = out_dir / f"ckpt-{step}"
                if is_rank0():
                    log.info(f"Saved → {ckpt_dir}")
                engine.save_checkpoint(str(ckpt_dir))
                if is_rank0():
                    gate_path = out_dir / f"gating_head_{step}.pt"
                    torch.save(
                        {
                            "state_dict": engine.module.gate.state_dict(),
                            "budgets": BUDGETS,
                            "anchor": gate_anchor,
                        },
                        gate_path,
                    )
                    log.info(f"Wrote {gate_path}")

            # eval
            if val_dl and (step % eval_steps == 0 or step >= total_steps):
                out = evaluate(val_dl)
                if out is not None and is_rank0():
                    metrics, lpb_mean, gp_mean = out
                    log.info(
                        f"VAL gate:{metrics['gate_loss']:.4f} hardCE:{metrics['hard_ce']:.4f} "
                        f"ent:{metrics['entropy_pen']:.4f} spread:{metrics['loss_spread']:.3f} "
                        f"oracle:{metrics['oracle_agreement']:.3f}"
                    )
                    if wandb and wandb.run:
                        wb = {
                            "step": step,
                            "gate/val_gate_loss": metrics["gate_loss"],
                            "gate/val_hard_ce": metrics["hard_ce"],
                            "gate/val_entropy_pen": metrics["entropy_pen"],
                            "gate/val_exp_loss": metrics["exp_loss"],
                            "gate/val_oracle_agreement": metrics["oracle_agreement"],
                            "gate/val_loss_spread": metrics["loss_spread"],
                            "gate/val_avg_ce": metrics["avg_ans_ce"],
                        }
                        for i, v in enumerate(lpb_mean):
                            wb[f"gate/val_loss_budget_{BUDGETS[i]}"] = float(v)
                        for i, v in enumerate(gp_mean):
                            wb[f"gate/val_gprob_{BUDGETS[i]}"] = float(v)
                        wandb.log(wb)

                    if metrics["gate_loss"] < best:
                        best = metrics["gate_loss"]
                        best_step = step
                        log.info(f"New BEST at step {step}: gate_loss={best:.4f}")

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()
    if wandb and wandb.run:
        wandb.finish()

    if is_rank0():
        # Final light save
        gate_path = out_dir / "gating_head_final.pt"
        torch.save(
            {
                "state_dict": engine.module.gate.state_dict(),
                "budgets": BUDGETS,
                "anchor": gate_anchor,
            },
            gate_path,
        )
        log.info(f"Wrote {gate_path}")

        if best_step is not None:
            try:
                best_ckpt = (out_dir / f"ckpt-{best_step}").resolve()
                link_ckpt = out_dir / "best"
                if link_ckpt.exists() or link_ckpt.is_symlink():
                    link_ckpt.unlink()
                link_ckpt.symlink_to(best_ckpt)
                log.info(f"Created symlink: {link_ckpt} → {best_ckpt}")
            except Exception as e:
                log.warning(f"Failed to create best symlink: {e}")


if __name__ == "__main__":
    import fire

    fire.Fire(train)
