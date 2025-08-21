# 🪆 Matryoshka Reasoning

> **Adaptive reasoning fine-tuning** — train one LLM that can emit *nested* reasoning depths in a single forward pass, choosing short or long explanations on demand.

---

## Why adaptive depth?

|                   | Vanilla CoT                                | Matryoshka-Reasoning                      |
| ----------------- | ------------------------------------------ | ----------------------------------------- |
| **Easy prompt**   | Prints long chain → ✖ slow & costly        | Prints **short answer**                   |
| **Medium prompt** | Prints long chain → ✖ overkill             | Prints **mid-length chain + answer**      |
| **Hard prompt**   | Prints long chain → ✓ thorough             | Prints **full chain + answer**            |

*Result:* lower average latency & token cost without sacrificing performance on hard examples.

---

## Method overview

```text
prompt → token IDs → one forward pass → N depth-specific losses
                               ├─ depth 0 : final answer only (no EOS training)
                               ├─ depth k : first k reasoning tokens + answer
                               └─ depth −1: full reasoning + answer + EOS
````

1. **Depth schedule**

   ```python
   depths = [0, k₁, k₂, …, -1]
   ```

   Each `kᵢ` is a token count inside the reasoning span. Mids are spaced automatically per-sample from the actual rationale length.

2. **Loss**

   ```
   loss_d     = CE(logits, labels_by_depth[d])
   total_loss = Σ w[d] · loss_d
   ```

   Example weights (N=3): `--depth_weights "0.1,0.45,0.45"`.

All depths share a single forward pass; masking yields simultaneous supervision.

---

## Installation

**Prereqs:** CUDA 11.8+, Python 3.10+

```bash
git clone https://github.com/junkim100/Matryoshka-Reasoning.git
cd Matryoshka-Reasoning
conda env create -f environment.yml
conda activate matryoshka-reasoning
export PYTHONPATH=$PWD:$PYTHONPATH
```

---

## Quick start 🚀

### Train (save **DeepSpeed shards only**) — recommended

Fast, robust saves (no massive all-gathers). Convert to HF *after* training.

```bash
# 4 GPUs example
deepspeed --num_gpus 4 src/train_ds_only.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --train_file data/train.jsonl \
  --valid_file data/val.jsonl \
  --output_dir output/exp1 \
  --num_depths 3 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 100 \
  --deepspeed_config configs/ds_config.json \
  --wandb_project matryoshka-reasoning \
  --bf16 True
```

This writes ZeRO shards into `output/exp1/ckpt-XXX/` (and a `best` symlink). No HF files are created during training.

---

## Offline conversion: DeepSpeed → Hugging Face

After training, convert **any** DS checkpoint to a standard HF folder. The converter:

* Adds `<think>` and `</think>` to the tokenizer if missing.
* Resizes embeddings to match the updated vocab.
* Consolidates ZeRO shards and saves sharded `safetensors`.

**Script:** `scripts/convert_ds_to_hf.py`

**Example: convert “best” checkpoint**

```bash
python scripts/convert_ds_to_hf.py \
  --checkpoint_dir output/exp1/best \
  --original_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir output/exp1/best-hf \
  --dtype bf16 \
  --max_shard_size 2GB
```

**Example: convert a specific step**

```bash
python scripts/convert_ds_to_hf.py \
  --checkpoint_dir output/exp1/ckpt-100 \
  --original_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir output/exp1/ckpt-100-hf
```

**Notes**

* `--checkpoint_dir` can be a `ckpt-XXX` folder or its internal `global_step*` folder — the script auto-detects the right directory.
* `--dtype` can be `fp16`, `bf16`, or `fp32` (default: `fp16`).
* The tokenizer is saved with the added tokens and the model is resized accordingly.

---

## Key parameters

| Flag                              | Description                                                                       | Default                         |
| --------------------------------- | --------------------------------------------------------------------------------- | ------------------------------- |
| `--num_depths`                    | 1 → full-depth only; 2 → answer-only + full; 3+ → answer-only + mid depths + full | `3`                             |
| `--depth_weights`                 | CSV per-depth loss weights                                                        | (auto: light 0, heavier others) |
| `--max_length`                    | Max tokens per example                                                            | `1024`                          |
| `--per_device_train_batch_size`   | Micro-batch size                                                                  | `1`                             |
| `--logging_steps`, `--save_steps` | Intervals for logging & checkpointing                                             | `10`, `100` (examples)          |
| `--bf16` / `--fp16`               | Mixed precision                                                                   | `bf16`                          |
| `--deepspeed_config`              | ZeRO-3 config JSON                                                                | `configs/ds_config.json`        |
| `--wandb_project`                 | Weights & Biases project name                                                     | `""`                            |

---

## Repository layout

```
Matryoshka-Reasoning/
├── src/
│   ├── train.py              # DeepSpeed trainer with learned gating head
│   └── matryoshka_infer.py   # Unified inference engine with adaptive depth selection
├── scripts/
│   ├── run.sh                # Example training launcher
│   ├── convert_ds_to_hf.py   # DS→HF converter (adds <think>/</think>, resizes embeddings)
│   ├── chat_cli.py           # Interactive chat interface with reasoning visualization
│   └── evaluate.py           # YAML-configurable evaluation framework
├── eval/
│   ├── config.py             # YAML task configuration loader
│   ├── metrics.py            # Math-aware evaluation metrics
│   └── runner.py             # Per-depth evaluation runner
├── configs/
│   └── ds_config.json        # Sample ZeRO-3 config
├── data/
│   ├── download_dataset.py   # Multi-dataset builder with chat template support
│   └── *.jsonl              # Training data files
├── environment.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Inference & Evaluation

### Interactive Chat

```bash
python scripts/chat_cli.py \
  --model output/exp1/ckpt-100-hf \
  --budgets "0,64,160,384,-1" \
  --device cuda:0
```

Features:
- **Adaptive depth selection** via learned gating head
- **Reasoning visualization** with token counts and timing
- **Budget control** for different reasoning depths
- **Natural vs forced closure detection**

### Evaluation Framework

```bash
# Example evaluation (config files need to be created)
python scripts/evaluate.py \
  --config configs/your_task.yaml \
  --model output/exp1/ckpt-100-hf \
  --output results/your_task.json
```

Features:
- **YAML-configurable tasks** with EleutherAI-like schema
- **Per-depth evaluation** with automatic depth selection
- **Math-aware metrics** with `\boxed{}` extraction and numerical equivalence
- **Token usage statistics** for reasoning vs answer generation

---

## License & data

* **Code:** [MIT](LICENSE)
* **Data & checkpoints:** follow original licenses.

PRs & issues welcome!