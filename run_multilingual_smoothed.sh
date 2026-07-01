#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_FILE="${1:-$SCRIPT_DIR/data/multi_65s.mp4}"
OUTPUT_JSON="${2:-$SCRIPT_DIR/results/result_video_multi_smoothed.json}"

mkdir -p "$(dirname "$OUTPUT_JSON")"

python run.py "$INPUT_FILE" \
  --top_k 5 \
  --allowed_languages en,es,fr,ar,zh,ru,pt,de,hi,id,ms,ja,ko,tr,km,th,tl,vi,lb \
  --min_speech_duration_ms 500 \
  --min_silence_duration_ms 200 \
  --lid_window_sec 5.0 \
  --lid_hop_sec 2.5 \
  --merge_gap_sec 0.0 \
  --merge_same_language \
  --smooth_language_islands \
  --max_island_duration_sec 2.0 \
  --island_score_threshold 0.6 \
  --output_json "$OUTPUT_JSON"

#allowed languages are the interpol langs
#I added lb as Luxembourgish because our demo video has Luxembourgish language inside.
#The expect correct output of the demo video should be: en/lb/es/pt/fr/de (English, Luxembourgish,Spanish, Portuguese,French, and German).
#The language's abbreviation follow this: https://www.loc.gov/standards/iso639-2/php/code_list.php