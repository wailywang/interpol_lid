#!/usr/bin/env bash
# Generate per-epoch training manifests with balanced sampling for all 19 classes.
#
# Compared to generate_rotating_19class_en_cv_unknown_manifests.sh, this adds
# --oversample so that data-limited classes (es/de/id/km/zh, ~37-42h available)
# are cycled to reach the same target_hours as the high-resource classes (~50h).
# Oversampling factors range from ~1.2x (zh) to ~1.36x (es) — all well below 2x.
#
# English uses Common Voice (1800h+), so it always reaches 50h without cycling.
set -euo pipefail

cd /export/home2/wa0009xi/ots-lid

DATA_ROOT="${DATA_ROOT:-/dataset/yw500/data}"
OUT_DIR="${OUT_DIR:-manifests/full_rotating_50h_19class_en_cv_unknown_oversample_epochs}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
TARGET_HOURS="${TARGET_HOURS:-50}"
UNKNOWN_HOURS="${UNKNOWN_HOURS:-50}"
SOURCE_BALANCE="${SOURCE_BALANCE:-equal}"

echo "Generating balanced-oversample 19-class rotating manifests"
echo "data_root:                     $DATA_ROOT"
echo "out_dir:                       $OUT_DIR"
echo "num_epochs:                    $NUM_EPOCHS"
echo "target_hours_per_class:        $TARGET_HOURS"
echo "unknown_hours_per_epoch:       $UNKNOWN_HOURS"
echo "source_balance:                $SOURCE_BALANCE"
echo "oversample:                    enabled"
echo

python scripts/prepare_full_data_rotating_lid_epochs.py \
  --voxlingua_root "$DATA_ROOT/voxlingua" \
  --commonvoice_en_root "$DATA_ROOT/common_voice/cv-corpus-25.0-en/cv-corpus-25.0-2026-03-09/en" \
  --commonvoice_yue_clips "$DATA_ROOT/commonvoice_yue_v25/cv-corpus-25.0-2026-03-09/yue/clips" \
  --magicdata_roots "$DATA_ROOT/magicdata/dailyuse,$DATA_ROOT/magicdata/conversational,$DATA_ROOT/magicdata_vehicle_mono/vehicle" \
  --include_commonvoice_en \
  --include_unknown \
  --unknown_label unknown \
  --voxlingua_unknown_root "$DATA_ROOT/voxlingua_unknown" \
  --exclude_manifests manifests/lid_val_en_cv_mix_3s.json,manifests/lid_eval_en_cv_mix_3s.json \
  --out_dir "$OUT_DIR" \
  --target_hours "$TARGET_HOURS" \
  --unknown_voxlingua_hours "$UNKNOWN_HOURS" \
  --num_epochs "$NUM_EPOCHS" \
  --source_balance "$SOURCE_BALANCE" \
  --oversample

echo
echo "Done. Manifests written to: $OUT_DIR"
echo "Train manifest template:"
echo "  $OUT_DIR/lid_train_epoch{epoch:02d}_cap${TARGET_HOURS%.*}h_3s.json"
