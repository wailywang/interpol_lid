#!/usr/bin/env bash
set -euo pipefail

cd /export/home2/wa0009xi/ots-lid

source /export/home2/wa0009xi/miniconda3/etc/profile.d/conda.sh
conda activate ots-lid-torch210

export PATH="$CONDA_PREFIX/bin:/export/home2/wa0009xi/miniconda3/bin:$PATH"
export FFMPEG_BINARY=/export/home2/wa0009xi/miniconda3/bin/ffmpeg
export FFPROBE_BINARY=/export/home2/wa0009xi/miniconda3/bin/ffprobe
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="ambernet_lid_18lang_yue_encoder_frozen"
CUDA_DEVICES="4,5,6,7"
DEVICES="4"
MAX_EPOCHS="10"
BATCH_SIZE="128"
VAL_BATCH_SIZE="128"
LR="1e-4"
TRAIN_MANIFEST="manifests/lid_train_3s.json"
VAL_MANIFEST="manifests/lid_val_3s.json"
EVAL_MANIFEST="manifests/lid_eval_3s.json"
ENABLE_AUGMENTATION="1"
MUSAN_DIR="data/augmentation/musan"
RIRS_DIR="data/augmentation/RIRS_NOISES"
AUGMENT_MANIFEST_DIR="manifests/augmentation"
ENABLE_WANDB="1"
WANDB_PROJECT="ots-lid"
WANDB_ENTITY=""
WANDB_MODE="online"
WANDB_TAGS="ambernet,encoder-frozen,augmentation"
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
    --enable_augmentation)
      ENABLE_AUGMENTATION="1"
      shift
      ;;
    --disable_augmentation)
      ENABLE_AUGMENTATION="0"
      shift
      ;;
    --musan_dir)
      MUSAN_DIR="$2"
      shift 2
      ;;
    --rirs_dir)
      RIRS_DIR="$2"
      shift 2
      ;;
    --augment_manifest_dir)
      AUGMENT_MANIFEST_DIR="$2"
      shift 2
      ;;
    --wandb)
      ENABLE_WANDB="1"
      shift
      ;;
    --no_wandb)
      ENABLE_WANDB="0"
      shift
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb_mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb_tags)
      WANDB_TAGS="$2"
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

if ! "$CONDA_PREFIX/bin/python" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "error: CUDA is not available in this session; refusing to run encoder-frozen training on CPU" >&2
  exit 1
fi

echo "run_name: $RUN_NAME"
echo "work_dir: $WORK_DIR"
echo "log_file: $LOG_FILE"
echo "out_ckpt: $OUT_CKPT"
echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo "devices: $DEVICES"
echo "batch_size: $BATCH_SIZE"
echo "val_batch_size: $VAL_BATCH_SIZE"
echo "lr: $LR"
echo "mode: encoder frozen"
echo "augmentation: $ENABLE_AUGMENTATION"
echo "musan_dir: $MUSAN_DIR"
echo "rirs_dir: $RIRS_DIR"
echo "wandb: $ENABLE_WANDB"
echo "wandb_project: $WANDB_PROJECT"
echo "wandb_mode: $WANDB_MODE"

AUGMENT_ARGS=()
if [[ "$ENABLE_AUGMENTATION" == "1" ]]; then
  AUGMENT_ARGS=(
    --enable_ambernet_augmentor
    --musan_dir "$MUSAN_DIR"
    --rirs_dir "$RIRS_DIR"
    --augment_manifest_dir "$AUGMENT_MANIFEST_DIR"
  )
fi

WANDB_ARGS=()
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_ARGS=(
    --wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_name "$RUN_NAME"
    --wandb_mode "$WANDB_MODE"
    --wandb_tags "$WANDB_TAGS"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
  fi
fi

"$CONDA_PREFIX/bin/python" scripts/finetune_ambernet_lid.py \
  --freeze_encoder \
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
  --accelerator gpu \
  "${AUGMENT_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
