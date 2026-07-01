#!/usr/bin/env python3
"""
Extract WAV files and NeMo JSONL manifests from NSC Singlish parquet files.

Output layout:
  /dataset/yw500/data/NSC/wavs/{split}/audio_XXXXXXXX.wav
  manifests/nsc_singlish_{split}.json
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd


SPLITS = {
    "train": sorted(Path("/dataset/yw500/data/NSC/data").glob("train-*.parquet")),
    "validation": sorted(Path("/dataset/yw500/data/NSC/data").glob("validation-*.parquet")),
    "test": sorted(Path("/dataset/yw500/data/NSC/data").glob("test-*.parquet")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", type=Path, default=Path("/dataset/yw500/data/NSC/wavs"))
    parser.add_argument("--manifest_dir", type=Path, default=Path("manifests"))
    parser.add_argument("--label", default="en_sg")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    return parser.parse_args()


def extract_parquet(args_tuple):
    parquet_path, wav_dir, label = args_tuple
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        audio = row["audio"]
        filename = audio["path"]          # e.g. audio_00000000.wav
        wav_path = wav_dir / filename
        if not wav_path.exists():
            wav_path.write_bytes(audio["bytes"])
        records.append({
            "audio_filepath": str(wav_path),
            "duration": float(row["duration"]),
            "label": label,
        })
    return records


def main():
    args = parse_args()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    for split, parquet_files in SPLITS.items():
        if not parquet_files:
            print(f"no parquet files found for split: {split}")
            continue

        wav_dir = args.wav_root / split
        wav_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{split}] {len(parquet_files)} parquet files → {wav_dir}")

        tasks = [(p, wav_dir, args.label) for p in parquet_files]
        with mp.Pool(args.workers) as pool:
            results = pool.map(extract_parquet, tasks)

        records = [r for batch in results for r in batch]

        manifest_path = args.manifest_dir / f"nsc_singlish_{split}.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_hours = sum(r["duration"] for r in records) / 3600
        print(f"[{split}] {len(records)} utterances, {total_hours:.1f}h → {manifest_path}")

    print("done")


if __name__ == "__main__":
    main()
