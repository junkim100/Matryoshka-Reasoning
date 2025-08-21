#!/usr/bin/env bash
# Simple launcher for Matryoshka (Gate-Only, fixed-oracle) training.
# Usage:
#   bash scripts/run.sh
#   # or overrides:
#   bash scripts/run.sh --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --cuda 0,1 --train data/train.jsonl --valid data/val.jsonl

set -euo pipefail

########################
# Defaults
########################
MODEL=${MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}

# Global budgets (token caps inside <think>); keep 0 for answer-only
BUDGETS=${BUDGETS:-"0,4,16,64,256,1024"}

TRAIN=${TRAIN:-"data/three_mix_equal_max4096_min0_seed42_chat_shuf_train.jsonl"}
VALID=${VALID:-"data/three_mix_equal_max4096_min0_seed42_chat_shuf_val.jsonl"}

MAXLEN=${MAXLEN:-4096}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-4}

# Deepspeed / logging
WANDB=${WANDB:-"matryoshka-reasoning"}    # set "" to disable Weights & Biases
LOGDIR=${LOGDIR:-"logs"}
DSCFG=${DSCFG:-"configs/ds_config.json"}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3,4,5,6,7"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Gate head configuration (matches src/train.py args)
GATE_ANCHOR=${GATE_ANCHOR:-"prethink"}         # prethink | think | prompt_mean
GATE_HIDDEN_MULT=${GATE_HIDDEN_MULT:-2.0}
GATE_DROPOUT=${GATE_DROPOUT:-0.0}
GATE_TEMPERATURE=${GATE_TEMPERATURE:-1.0}

# Oracle & regularizers (tuned defaults for fixed oracle)
ORACLE_TEMPERATURE=${ORACLE_TEMPERATURE:-0.25}
GATE_LENGTH_PENALTY=${GATE_LENGTH_PENALTY:-0.05}
GATE_ENTROPY_PENALTY=${GATE_ENTROPY_PENALTY:-0.02}
HARD_ORACLE_WEIGHT=${HARD_ORACLE_WEIGHT:-0.0}
EXP_LOSS_WEIGHT=${EXP_LOSS_WEIGHT:-0.0}

# NLL chunking (reduces peak memory)
CE_CHUNK_TOKENS=${CE_CHUNK_TOKENS:-256}

# DataLoader per-device batch (trainer arg name!)
PER_DEVICE_BS=${PER_DEVICE_BS:-1}

# BF16 compute flag (trainer arg name!)
BF16=${BF16:-True}

# Output directory (auto-named)
MODEL_SLUG="${MODEL//\//-}"
BUDGET_SLUG="${BUDGETS//,/x}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUT_DEFAULT="output/${MODEL_SLUG}_${GATE_ANCHOR}_${BUDGET_SLUG}_${TIMESTAMP}"
OUT=${OUT:-"$OUT_DEFAULT"}

########################
# Parse CLI overrides
########################
while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda) CUDA_DEVICES=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --train) TRAIN=$2; shift 2;;
    --valid) VALID=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --budgets) BUDGETS=$2; shift 2;;
    --maxlen) MAXLEN=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --lr) LR=$2; shift 2;;
    --ds) DSCFG=$2; shift 2;;
    --wandb) WANDB=$2; shift 2;;
    --loglevel) LOG_LEVEL=$2; shift 2;;
    --gate_anchor) GATE_ANCHOR=$2; shift 2;;
    --gate_hidden_mult) GATE_HIDDEN_MULT=$2; shift 2;;
    --gate_dropout) GATE_DROPOUT=$2; shift 2;;
    --gate_temperature) GATE_TEMPERATURE=$2; shift 2;;
    --oracle_temperature) ORACLE_TEMPERATURE=$2; shift 2;;
    --gate_length_penalty) GATE_LENGTH_PENALTY=$2; shift 2;;
    --gate_entropy_penalty) GATE_ENTROPY_PENALTY=$2; shift 2;;
    --hard_oracle_weight) HARD_ORACLE_WEIGHT=$2; shift 2;;
    --exp_loss_weight) EXP_LOSS_WEIGHT=$2; shift 2;;
    --ce_chunk_tokens) CE_CHUNK_TOKENS=$2; shift 2;;
    --per_device_bs) PER_DEVICE_BS=$2; shift 2;;
    --bf16) BF16=$2; shift 2;;
    --help|-h)
      echo "Launch Matryoshka (Gate-Only, fixed-oracle) training"
      echo "Options:"
      echo "  --cuda DEVICES                 CUDA devices (default: $CUDA_DEVICES)"
      echo "  --model MODEL                  HF model path (default: $MODEL)"
      echo "  --train FILE                   Train JSONL (default: $TRAIN)"
      echo "  --valid FILE                   Valid JSONL (default: $VALID)"
      echo "  --out DIR                      Output dir (default:auto)"
      echo "  --budgets CSV                  Budgets CSV (default: $BUDGETS)"
      echo "  --maxlen N                     Max context tokens (default: $MAXLEN)"
      echo "  --epochs N                     Training epochs (default: $EPOCHS)"
      echo "  --lr LR                        Learning rate (default: $LR)"
      echo "  --ds FILE                      DeepSpeed config (default: $DSCFG)"
      echo "  --wandb NAME                   W&B project (default: $WANDB; empty to disable)"
      echo "  --loglevel LEVEL               Logging level (default: $LOG_LEVEL)"
      echo "  --gate_anchor MODE             prethink|think|prompt_mean (default: $GATE_ANCHOR)"
      echo "  --gate_hidden_mult M           Gate MLP hidden multiplier (default: $GATE_HIDDEN_MULT)"
      echo "  --gate_dropout P               Gate dropout (default: $GATE_DROPOUT)"
      echo "  --gate_temperature T           Gate temperature (default: $GATE_TEMPERATURE)"
      echo "  --oracle_temperature T         Soft-oracle temperature (default: $ORACLE_TEMPERATURE)"
      echo "  --gate_length_penalty W        Absolute length penalty (default: $GATE_LENGTH_PENALTY)"
      echo "  --gate_entropy_penalty W       Entropy penalty (default: $GATE_ENTROPY_PENALTY)"
      echo "  --hard_oracle_weight W         Hard-CE weight (default: $HARD_ORACLE_WEIGHT)"
      echo "  --exp_loss_weight W            Expected-loss weight (default: $EXP_LOSS_WEIGHT)"
      echo "  --ce_chunk_tokens N            CE time-chunk size (default: $CE_CHUNK_TOKENS)"
      echo "  --per_device_bs N              Loader micro-batch size (default: $PER_DEVICE_BS)"
      echo "  --bf16 BOOL                    Use bfloat16 (default: $BF16)"
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

########################
# Environment and paths
########################
mkdir -p "$OUT" "$LOGDIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_PROJECT="${WANDB}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PORT=$((12000 + RANDOM % 1000))
LOGFILE="$LOGDIR/train_$(basename "$OUT").log"

########################
# Command
########################
CMD="deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} \
  --master_port ${PORT} \
  src/train.py \
    --model_name_or_path \"$MODEL\" \
    --train_file \"$TRAIN\" \
    --valid_file \"$VALID\" \
    --output_dir \"$OUT\" \
    --budgets \"$BUDGETS\" \
    --max_length ${MAXLEN} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --deepspeed_config \"$DSCFG\" \
    --wandb_project \"$WANDB\" \
    --bf16 ${BF16} \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --logging_steps 20 \
    --save_steps 1000 \
    --eval_steps 200 \
    --log_level \"$LOG_LEVEL\" \
    --ce_chunk_tokens ${CE_CHUNK_TOKENS} \
    --gate_anchor \"$GATE_ANCHOR\" \
    --gate_temperature ${GATE_TEMPERATURE} \
    --gate_hidden_mult ${GATE_HIDDEN_MULT} \
    --gate_dropout ${GATE_DROPOUT} \
    --gate_length_penalty ${GATE_LENGTH_PENALTY} \
    --oracle_temperature ${ORACLE_TEMPERATURE} \
    --gate_entropy_penalty ${GATE_ENTROPY_PENALTY} \
    --hard_oracle_weight ${HARD_ORACLE_WEIGHT} \
    --exp_loss_weight ${EXP_LOSS_WEIGHT}
"

echo "Launching:"
echo "$CMD"
echo "Logs → $LOGFILE"
# shellcheck disable=SC2086
eval $CMD 2>&1 | tee "$LOGFILE"
