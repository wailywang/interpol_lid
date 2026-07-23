#!/usr/bin/env bash
# Generate all train/val/eval manifests for the 19-class LID experiment.
#
# Final outputs:
#   train: manifests/full_rotating_50h_19class_en_cv_unknown_oversample_epochs/lid_train_epoch{00..19}_cap50h_3s.json
#   val:   manifests/heldout_19class_en_cv_unknown/lid_val_19class_3s.json
#   eval:  manifests/heldout_19class_en_cv_unknown/lid_eval_19class_3s.json
#
# Usage (run from ots-lid project root):
#   bash scripts/generate_all_manifests.sh
#
# Override data root if needed:
#   DATA_ROOT=/your/path bash scripts/generate_all_manifests.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/dataset/yw500/data}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
TARGET_HOURS="${TARGET_HOURS:-50}"
UNKNOWN_HOURS="${UNKNOWN_HOURS:-50}"
TRAIN_OUT_DIR="${MANIFEST_DIR}/full_rotating_50h_19class_en_cv_unknown_oversample_epochs"
FINAL_OUT_DIR="${MANIFEST_DIR}/heldout_19class_en_cv_unknown"

echo "=========================================="
echo " DATA_ROOT  : $DATA_ROOT"
echo " MANIFEST_DIR: $MANIFEST_DIR"
echo " NUM_EPOCHS : $NUM_EPOCHS"
echo "=========================================="
echo

# ── Step 1: base train/val/eval (VoxLingua 17 langs + CommonVoice Cantonese) ──
echo "[1/4] Generating base manifests (VoxLingua + CV Cantonese)..."
python scripts/prepare_balanced_lid_manifests.py \
  --voxlingua_root        "$DATA_ROOT/voxlingua" \
  --commonvoice_yue_root  "$DATA_ROOT/commonvoice_yue_v25/cv-corpus-25.0-2026-03-09/yue" \
  --out_dir               "$MANIFEST_DIR"
echo "[1/4] Done."
echo

# ── Step 2: mix in CommonVoice English → en_cv_mix val/eval (3s segments) ─────
echo "[2/4] Mixing in CommonVoice English..."
python scripts/prepare_en_cv_mixed_val_eval.py \
  --manifest_dir        "$MANIFEST_DIR" \
  --commonvoice_en_root "$DATA_ROOT/common_voice/cv-corpus-25.0-en/cv-corpus-25.0-2026-03-09/en"
echo "[2/4] Done."
echo

# ── Step 3: add unknown class → final 19-class val/eval ───────────────────────
echo "[3/4] Adding unknown class to val/eval..."
python scripts/prepare_19class_unknown_val_eval_manifests.py \
  --base_val_manifest      "$MANIFEST_DIR/lid_val_en_cv_mix_3s.json" \
  --base_eval_manifest     "$MANIFEST_DIR/lid_eval_en_cv_mix_3s.json" \
  --voxlingua_unknown_root "$DATA_ROOT/voxlingua_unknown" \
  --out_dir                "$FINAL_OUT_DIR"
echo "[3/4] Done."
echo

# ── Step 4: generate 20-epoch rotating train manifests (oversampled) ──────────
echo "[4/4] Generating rotating train manifests (${NUM_EPOCHS} epochs, oversample)..."
python scripts/prepare_full_data_rotating_lid_epochs.py \
  --voxlingua_root          "$DATA_ROOT/voxlingua" \
  --commonvoice_en_root     "$DATA_ROOT/common_voice/cv-corpus-25.0-en/cv-corpus-25.0-2026-03-09/en" \
  --commonvoice_yue_clips   "$DATA_ROOT/commonvoice_yue_v25/cv-corpus-25.0-2026-03-09/yue/clips" \
  --magicdata_roots         "$DATA_ROOT/magicdata/dailyuse,$DATA_ROOT/magicdata/conversational,$DATA_ROOT/magicdata_vehicle_mono/vehicle" \
  --include_commonvoice_en \
  --include_unknown \
  --unknown_label           unknown \
  --voxlingua_unknown_root  "$DATA_ROOT/voxlingua_unknown" \
  --exclude_manifests       "$MANIFEST_DIR/lid_val_en_cv_mix_3s.json,$MANIFEST_DIR/lid_eval_en_cv_mix_3s.json" \
  --out_dir                 "$TRAIN_OUT_DIR" \
  --target_hours            "$TARGET_HOURS" \
  --unknown_voxlingua_hours "$UNKNOWN_HOURS" \
  --num_epochs              "$NUM_EPOCHS" \
  --source_balance          equal \
  --oversample
echo "[4/4] Done."
echo

echo "=========================================="
echo " All manifests generated successfully."
echo ""
echo " train : $TRAIN_OUT_DIR/lid_train_epoch{00..$(( NUM_EPOCHS - 1 ))}_cap${TARGET_HOURS%.*}h_3s.json"
echo " val   : $FINAL_OUT_DIR/lid_val_19class_3s.json"
echo " eval  : $FINAL_OUT_DIR/lid_eval_19class_3s.json"
echo "=========================================="
