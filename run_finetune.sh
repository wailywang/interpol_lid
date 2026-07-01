#!/usr/bin/env bash
set -euo pipefail

cd /export/home2/wa0009xi/ots-lid

source /export/home2/wa0009xi/miniconda3/etc/profile.d/conda.sh
conda activate ots-lid-torch210

export PATH="$CONDA_PREFIX/bin:/export/home2/wa0009xi/miniconda3/bin:$PATH"
export FFMPEG_BINARY=/export/home2/wa0009xi/miniconda3/bin/ffmpeg
export FFPROBE_BINARY=/export/home2/wa0009xi/miniconda3/bin/ffprobe
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="ambernet_lid_18lang_yue"
CUDA_DEVICES="0,1,2"
DEVICES="3"
MAX_EPOCHS="5"
BATCH_SIZE="32"
VAL_BATCH_SIZE="32"
LR="1e-5"
TRAIN_MANIFEST="manifests/lid_train_3s.json"
VAL_MANIFEST="manifests/lid_val_3s.json"
EVAL_MANIFEST="manifests/lid_eval_3s.json"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_name)
      RUN_NAME="$2"
      shift 2
      ;;
    --cuda_visible_devices)
      CUDA_DEVICES="$2"
      shift 2
      ;;
    --devices)
      DEVICES="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --val_batch_size)
      VAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --train_manifest)
      TRAIN_MANIFEST="$2"
      shift 2
      ;;
    --val_manifest)
      VAL_MANIFEST="$2"
      shift 2
      ;;
    --eval_manifest)
      EVAL_MANIFEST="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

WORK_DIR="experiments/${RUN_NAME}"
LOG_FILE="${WORK_DIR}/train.log"
OUT_CKPT="checkpoints/${RUN_NAME}.nemo"

mkdir -p "$WORK_DIR"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

echo "run_name: $RUN_NAME"
echo "work_dir: $WORK_DIR"
echo "log_file: $LOG_FILE"
echo "out_ckpt: $OUT_CKPT"
echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo "devices: $DEVICES"
echo "batch_size: $BATCH_SIZE"
echo "val_batch_size: $VAL_BATCH_SIZE"
echo "lr: $LR"

"$CONDA_PREFIX/bin/python" scripts/finetune_ambernet_lid.py \
  --train_manifest "$TRAIN_MANIFEST" \
  --val_manifest "$VAL_MANIFEST" \
  --eval_manifest "$EVAL_MANIFEST" \
  --work_dir "$WORK_DIR" \
  --log_file "$LOG_FILE" \
  --out_ckpt "$OUT_CKPT" \
  --max_epochs "$MAX_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --val_batch_size "$VAL_BATCH_SIZE" \
  --lr "$LR" \
  --devices "$DEVICES" \
  "${EXTRA_ARGS[@]}"
