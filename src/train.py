#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matryoshka‑Reasoning – DeepSpeed trainer (saves DS shards only; HF conversion is offline)
"""
import json, logging, os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch, torch.nn.functional as F, deepspeed
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


# ---------------- helpers & logging ---------------- #
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


# --------------------- dataset --------------------- #
SYSTEM_MSG = "You are a helpful reasoning assistant. Think step‑by‑step, then answer."


class JsonlDS(Dataset):
    def __init__(self, path: str):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if l:
                    rows.append(json.loads(l))
        if not rows:
            raise ValueError(f"Empty dataset: {path}")
        self.rows: List[Dict[str, Any]] = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


# ---------------- depth utilities ------------------ #
def make_depths(n: int, L: int) -> List[int]:
    """
    Depth schedule: [0, k1, …, -1] with kᵢ ∈ [1, max(1, L-1)].
    Mids spaced roughly in [0.3·L, 0.7·L].
    """
    if n <= 1:
        return [-1]
    mids = n - 2
    if mids <= 0:
        return [0, -1]
    upper = max(1, L - 1)
    lo = max(1, int(0.30 * L))
    hi = max(1, int(0.70 * L))
    if hi <= lo:
        mid_k = max(1, min(upper, max(1, L // 2)))
        ks = [mid_k] * mids
    else:
        step = (hi - lo) / (mids + 1)
        ks = [max(1, min(upper, int(lo + (i + 1) * step))) for i in range(mids)]
    return [0] + ks + [-1]


# --------------- collator -------------------------- #
class MatryoshkaCollator:
    """
    Out:
        input_ids      : [B, L]
        attention_mask : [B, L]
        labels_by_depth: { i: Tensor[B, L] } for i in range(D)  (‑100 masked)
    """

    def __init__(self, tok: AutoTokenizer, num_depths: int, max_len: int):
        self.tok = tok
        self.max_len = max_len
        self.depths = make_depths(num_depths, max_len)

        for t in ["<think>", "</think>"]:
            if t not in self.tok.get_vocab():
                self.tok.add_tokens([t])

        self.eos_id = self.tok.eos_token_id
        self.eot_id = getattr(self.tok, "eot_token_id", None)
        self.think_id = self.tok.convert_tokens_to_ids("<think>")

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        ids_all, attn_all = [], []
        labels_by_depth = {i: [] for i in range(len(self.depths))}

        for ex in batch:
            q, sol, ans = ex["question"], ex["solution"], ex["answer"]

            # normalize spaces inside \boxed{ ... } to reduce false mismatches
            sol = sol.replace("\\boxed{ ", "\\boxed{").replace(" }", "}")

            # ensure <think>…</think> and a boxed answer suffix
            if "<think>" not in sol and "**Final Answer**" in sol:
                sol = "<think>" + sol.replace(
                    "**Final Answer**", "</think>\n**Final Answer**"
                )
            elif "<think>" not in sol and "\\boxed" in sol:
                sol = "<think>" + sol.replace("\\boxed", "</think>\n\\boxed")
            elif "<think>" not in sol:
                sol = f"<think>{sol}</think>\n\\boxed{{{ans}}}"

            chat = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": q},
                {"role": "assistant", "content": sol + self.tok.eos_token},
            ]
            txt = self.tok.apply_chat_template(chat, tokenize=False)
            enc = self.tok(
                txt,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                add_special_tokens=False,
                return_tensors="pt",
            )
            ids, msk = enc.input_ids[0], enc.attention_mask[0]

            content_len = int(msk.sum().item())
            content_ids = ids[:content_len]
            eos_mask = content_ids == self.eos_id
            if self.eot_id is not None:
                eos_mask = eos_mask | (content_ids == self.eot_id)
            eos_pos = eos_mask.nonzero(as_tuple=False)
            has_end = eos_pos.numel() > 0
            sid = int(eos_pos[-1].item()) if has_end else (content_len - 1)

            think_pos = (ids == self.think_id).nonzero(as_tuple=True)
            found_think = len(think_pos[0]) > 0
            r0 = int(think_pos[0][0]) if found_think else 0

            # tolerant boxed/raw matching
            raw_norm = ans.strip().replace("−", "-")
            patterns = [
                self.tok.encode(f"\\boxed{{{raw_norm}}}", add_special_tokens=False),
                self.tok.encode(raw_norm, add_special_tokens=False),
                self.tok.encode(" " + raw_norm, add_special_tokens=False),
                self.tok.encode("(" + raw_norm + ")", add_special_tokens=False),
            ]

            def find_span_range(ids_t, pattern, start, end):
                if not pattern:
                    return -1, -1
                seq = ids_t[start:end].tolist()
                M = len(pattern)
                for i in range(len(seq) - M + 1):
                    if seq[i : i + M] == pattern:
                        return start + i, start + i + M
                return -1, -1

            a_sp, a_ep = (-1, -1)
            for pat in patterns:
                a_sp, a_ep = find_span_range(ids, pat, r0, content_len)
                if a_sp >= 0:
                    break
            if a_sp < 0:
                a_ep = sid
                a_sp = max(r0, sid - 1)

            if not found_think:
                r0 = a_sp

            L = max(1, sid - r0)
            depths = make_depths(len(self.depths), L)

            for i, d in enumerate(depths):
                lab = ids.clone().fill_(-100)
                if d == 0:
                    lab[a_sp:a_ep] = ids[a_sp:a_ep]
                else:
                    end = (sid + 1) if (d == -1 and has_end) else min(r0 + d, sid)
                    if end > r0:
                        lab[r0:end] = ids[r0:end]
                    lab[a_sp:a_ep] = ids[a_sp:a_ep]
                    if d == -1 and has_end:
                        lab[sid] = ids[sid]
                if lab.eq(-100).all():
                    lab[a_sp:a_ep] = ids[a_sp:a_ep]
                labels_by_depth[i].append(lab)

            ids_all.append(ids)
            attn_all.append(msk)

        return {
            "input_ids": torch.stack(ids_all),
            "attention_mask": torch.stack(attn_all),
            "labels_by_depth": {i: torch.stack(v) for i, v in labels_by_depth.items()},
        }


# -------------- DeepSpeed cfg patch helper ---------------- #
def patch_ds_cfg(cfg_path: str, lr: float, total_steps: int, micro_bs: int) -> str:
    with open(cfg_path) as f:
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

    cfg["train_micro_batch_size_per_gpu"] = int(micro_bs)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_acc = int(cfg.get("gradient_accumulation_steps", 1))
    cfg["train_batch_size"] = int(micro_bs) * grad_acc * world_size

    patched = cfg_path + ".patched"
    with open(patched, "w") as f:
        json.dump(cfg, f, indent=2)
    return patched


# -------------------- trainer ---------------------- #
def train(
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    valid_file: str = "",
    num_depths: int = 1,
    max_length: int = 4096,
    depth_weights: str = "",
    learning_rate: float = 1e-5,
    per_device_train_batch_size: int = 1,
    num_train_epochs: int = 3,
    logging_steps: int = 20,
    save_steps: int = 100,
    eval_steps: int = 100,
    deepspeed_config: str = "configs/ds_config.json",
    wandb_project: str = "",
    bf16: bool = True,
    max_eval_batches: int = 0,
    log_level: str = "INFO",
    local_rank: int = -1,
    **_unused,  # swallow any extra CLI flags
):
    # If DeepSpeed/torchrun passed LOCAL_RANK, set the CUDA device (optional)
    try:
        lr = int(os.environ.get("LOCAL_RANK", local_rank if local_rank != -1 else 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(lr)
    except Exception:
        pass

    log = get_logger(log_level)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    tok.pad_token = tok.eos_token
    tok.truncation_side = "left"

    coll = MatryoshkaCollator(tok, num_depths, max_length)
    depths = coll.depths
    D = len(depths)

    if depth_weights:
        w = [float(x) for x in depth_weights.split(",")]
        if len(w) != D:
            raise ValueError(
                f"--depth_weights must have exactly {D} comma‑separated values"
            )
        weights = torch.tensor(w, dtype=torch.float32)
    else:
        weights = torch.tensor(
            [1.0] if D == 1 else [0.1] + [0.9 / (D - 1)] * (D - 1), dtype=torch.float32
        )
    log.info(f"> Depth schedule  : {depths}")
    log.info(f"> Depth weights   : {weights.tolist()}")

    if (not torch.distributed.is_initialized()) and int(
        os.environ.get("WORLD_SIZE", "1")
    ) > 1:
        deepspeed.init_distributed()

    train_ds = JsonlDS(train_file)
    sampler = (
        DistributedSampler(train_ds) if torch.distributed.is_initialized() else None
    )
    dl = DataLoader(
        train_ds,
        batch_size=per_device_train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=coll,
        num_workers=2,
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
            batch_size=per_device_train_batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=coll,
            num_workers=2,
            pin_memory=True,
        )

    total_steps = len(dl) * num_train_epochs
    ds_cfg_patched = patch_ds_cfg(
        deepspeed_config, learning_rate, total_steps, per_device_train_batch_size
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    if len(tok) != model.config.vocab_size:
        model.resize_token_embeddings(len(tok))

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg_patched,
    )

    weights = weights.to(engine.device)

    os.makedirs(output_dir, exist_ok=True)

    if wandb_project and is_rank0() and wandb is not None:
        base = model_name_or_path.split("/")[-1]
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"Depth{D}_{base}_{ts}"
        wandb.init(project=wandb_project, name=run_name)
        wandb.config.update(
            dict(
                model=model_name_or_path,
                num_depths=D,
                max_length=max_length,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
            )
        )

    # ------------------- loss fn ------------------- #
    def batch_loss(batch):
        input_ids = batch["input_ids"].to(engine.device)
        attention_mask = batch["attention_mask"].to(engine.device)

        labels_by_d = batch["labels_by_depth"]
        _D = len(labels_by_d)
        labels = torch.stack([labels_by_d[i] for i in range(_D)], dim=0).to(
            engine.device
        )  # [D,B,L]

        logits = engine(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits  # [B,L,V]

        logits_shift = logits[..., :-1, :]  # [B,L-1,V]
        labels_shift = labels[..., 1:]  # [D,B,L-1]
        mask = labels_shift.ne(-100)  # [D,B,L-1]

        _D, B, Lm1 = labels_shift.shape
        V = logits_shift.size(-1)
        logits_expand = logits_shift.unsqueeze(0).expand(_D, -1, -1, -1)

        loss_flat = F.cross_entropy(
            logits_expand.reshape(-1, V),
            labels_shift.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(_D, B, Lm1)

        masked_loss = loss_flat * mask.float()
        token_counts = mask.sum(dim=(1, 2)).clamp_min(1.0)  # [D]
        loss_per_depth = masked_loss.sum(dim=(1, 2)) / token_counts
        total_loss = (loss_per_depth * weights).sum()
        return total_loss, loss_per_depth, token_counts

    @torch.no_grad()
    def evaluate(val_dl_local):
        if val_dl_local is None:
            return None
        engine.eval()
        tot_loss_tokens = torch.zeros(D, device=engine.device)
        tot_tokens = torch.zeros(D, device=engine.device)
        for bi, batch in enumerate(val_dl_local):
            _, depth_losses, token_counts = batch_loss(batch)
            tot_loss_tokens += depth_losses * token_counts
            tot_tokens += token_counts
            if max_eval_batches and (bi + 1) >= max_eval_batches:
                break
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tot_loss_tokens, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(tot_tokens, op=torch.distributed.ReduceOp.SUM)
        mean_per_depth = tot_loss_tokens / tot_tokens.clamp_min(1.0)
        total = (mean_per_depth * weights).sum()
        engine.train()
        return (
            total.item(),
            mean_per_depth.detach().cpu().tolist(),
            tot_tokens.detach().cpu().tolist(),
        )

    # ---------------- training loop -------------- #
    pbar = tqdm(total=total_steps, disable=not (tqdm and is_rank0()), unit="step")
    step = 0
    best_val_total = float("inf")
    best_step: Optional[int] = None

    for epoch in range(num_train_epochs):
        if isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)
        for batch in dl:
            step += 1
            loss, depth_losses, valid_tok = batch_loss(batch)
            engine.backward(loss)
            engine.step()

            if is_rank0() and step % logging_steps == 0:
                msg = (
                    f"step {step}/{total_steps}  "
                    + " ".join(
                        [f"depth{i}:{l.item():.3f}" for i, l in enumerate(depth_losses)]
                    )
                    + f"  tot:{loss.item():.3f}  "
                    + f"lr:{engine.get_lr()[0]:.2e}"
                )
                get_logger().info(msg)

                if wandb and wandb.run:
                    log_dict = {
                        "step": step,
                        "train/total": loss.item(),
                        "lr": engine.get_lr()[0],
                    }
                    for i, (l, v) in enumerate(zip(depth_losses, valid_tok)):
                        log_dict[f"loss/depth_{i}"] = l.item()
                        log_dict[f"valid_tok/depth_{i}"] = int(v)
                        log_dict[f"weight/depth_{i}"] = float(weights[i].item())
                    wandb.log(log_dict)

            # periodic save — DeepSpeed shards only (fast, no huge all‑gather)
            if step % save_steps == 0 or step >= total_steps:
                ckpt_dir = Path(output_dir) / f"ckpt-{step}"
                if is_rank0():
                    get_logger().info(f"Saved → {ckpt_dir}")
                engine.save_checkpoint(str(ckpt_dir))
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

            # periodic validation (lightweight)
            if val_dl and (step % eval_steps == 0 or step >= total_steps):
                out = evaluate(val_dl)
                if out is not None and is_rank0():
                    val_total, val_d_losses, val_tok = out
                    get_logger().info(
                        "VAL  "
                        + " ".join(
                            [f"depth{i}:{v:.3f}" for i, v in enumerate(val_d_losses)]
                        )
                        + f"  tot:{val_total:.3f}"
                    )
                    if wandb and wandb.run:
                        wb = {"step": step, "val/total": float(val_total)}
                        for i, (l, t) in enumerate(zip(val_d_losses, val_tok)):
                            wb[f"val/loss/depth_{i}"] = float(l)
                            wb[f"val/valid_tok/depth_{i}"] = int(t)
                        wandb.log(wb)

                    if val_total < best_val_total:
                        best_val_total = float(val_total)
                        best_step = step
                        get_logger().info(
                            f"New BEST at step {step} (val_total={val_total:.4f})"
                        )

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()
    if wandb and wandb.run:
        wandb.finish()

    if is_rank0():
        get_logger().info("Training completed.")
        # Best symlink pointing to DS checkpoint (parent folder)
        if best_step is not None:
            try:
                best_ckpt = (Path(output_dir) / f"ckpt-{best_step}").resolve()
                link_ckpt = Path(output_dir) / "best"
                if link_ckpt.exists() or link_ckpt.is_symlink():
                    link_ckpt.unlink()
                link_ckpt.symlink_to(best_ckpt)
                get_logger().info(f"Created symlink: {link_ckpt} → {best_ckpt}")
            except Exception as e:
                get_logger().warning(f"Failed to create best symlink: {e}")

    # cleanup patched config
    ds_cfg_patched = deepspeed_config + ".patched"
    if is_rank0() and os.path.exists(ds_cfg_patched):
        os.remove(ds_cfg_patched)


if __name__ == "__main__":
    import fire

    fire.Fire(train)
