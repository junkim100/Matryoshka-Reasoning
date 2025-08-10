#!/usr/bin/env bash
# Usage: bash scripts/run.sh --gpus 4 --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
set -euo pipefail

########################
# Default parameters
########################
GPUS=${GPUS:-4}
MODEL=${MODEL:-"meta-llama/Meta-Llama-3-8B-Instruct"}
DEPTHS=3
TRAIN=data/train.jsonl
VALID=data/val.jsonl
OUT=output/Depth${DEPTHS}_${MODEL//\//-}_$(date +"%Y%m%d_%H%M")
MAXLEN=4096
EPOCHS=3
WANDB=matryoshka-reasoning
LOGDIR=logs
DSCFG=configs/ds_config.json
CUDA_DEVICES="0,1,2,3,4,5,6,7"
LOG_LEVEL=INFO

########################
# Parse CLI overrides
########################
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS=$2; shift 2;;
        --cuda) CUDA_DEVICES=$2; shift 2;;
        --model) MODEL=$2; shift 2;;
        --train) TRAIN=$2; shift 2;;
        --valid) VALID=$2; shift 2;;
        --out) OUT=$2; shift 2;;
        --depths) DEPTHS=$2; shift 2;;
        --help|-h)
          echo "Launch DeepSpeed Matryoshka training"
          exit 0;;
        *) echo "Unknown arg $1"; exit 1;;
    esac
done

mkdir -p "$OUT" "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
CMD="deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} \
  --master_port=$((12000 + RANDOM % 1000)) \
  --module src.train \
    --model_name_or_path $MODEL \
    --train_file $TRAIN \
    --valid_file $VALID \
    --output_dir $OUT \
    --num_depths $DEPTHS \
    --max_length $MAXLEN \
    --num_train_epochs $EPOCHS \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --deepspeed_config $DSCFG \
    --wandb_project $WANDB \
    --bf16 True \
    --log_level $LOG_LEVEL"

echo "Launching command:"
echo "$CMD"
echo "Logs: $LOGDIR/train_$(basename $OUT).log"
eval $CMD 2>&1 | tee "$LOGDIR/train_$(basename $OUT).log"
