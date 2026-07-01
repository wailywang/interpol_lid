#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand LID manifests into non-overlapping fixed-duration segment manifests using offset."
    )
    parser.add_argument(
        "--manifest_dir",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/manifests"),
    )
    parser.add_argument("--train_manifest", default="lid_train.json")
    parser.add_argument("--val_manifest", default="lid_val.json")
    parser.add_argument("--eval_manifest", default="lid_eval.json")
    parser.add_argument("--out_train", default="lid_train_3s.json")
    parser.add_argument("--out_val", default="lid_val_3s.json")
    parser.add_argument("--out_eval", default="lid_eval_3s.json")
    parser.add_argument("--segment_sec", type=float, default=3.0)
    parser.add_argument(
        "--include_short_final",
        action="store_true",
        help="Include a final shorter segment if the leftover audio is at least --min_final_sec.",
    )
    parser.add_argument("--min_final_sec", type=float, default=1.0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print summary without writing segment manifests.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_manifest(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def expand_record(record: dict, segment_sec: float, include_short_final: bool, min_final_sec: float) -> list[dict]:
    base_offset = float(record.get("offset", 0.0))
    duration = float(record["duration"])
    label = record["label"]
    audio_filepath = record["audio_filepath"]

    segments = []
    full_segments = int(duration // segment_sec)
    for idx in range(full_segments):
        segments.append(
            {
                "audio_filepath": audio_filepath,
                "offset": round(base_offset + idx * segment_sec, 3),
                "duration": round(segment_sec, 3),
                "label": label,
            }
        )

    leftover = duration - full_segments * segment_sec
    if include_short_final and leftover >= min_final_sec:
        segments.append(
            {
                "audio_filepath": audio_filepath,
                "offset": round(base_offset + full_segments * segment_sec, 3),
                "duration": round(leftover, 3),
                "label": label,
            }
        )
    return segments


def summarize(records: list[dict]) -> dict[str, tuple[int, float]]:
    counts = defaultdict(int)
    seconds = defaultdict(float)
    for record in records:
        label = record["label"]
        counts[label] += 1
        seconds[label] += float(record["duration"])
    return {label: (counts[label], seconds[label] / 3600.0) for label in sorted(counts)}


def convert_one(input_path: Path, output_path: Path, args: argparse.Namespace) -> list[dict]:
    source = read_manifest(input_path)
    expanded = []
    dropped_seconds = 0.0
    for record in source:
        segments = expand_record(
            record,
            args.segment_sec,
            args.include_short_final,
            args.min_final_sec,
        )
        expanded.extend(segments)
        dropped_seconds += float(record["duration"]) - sum(float(seg["duration"]) for seg in segments)

    print(f"\n{input_path.name} -> {output_path.name}")
    print(f"source records: {len(source)}")
    print(f"segment records: {len(expanded)}")
    print(f"dropped leftover hours: {dropped_seconds / 3600.0:.2f}")
    print(f"{'label':<5} {'segments':>9} {'hours':>9}")
    for label, (count, hrs) in summarize(expanded).items():
        print(f"{label:<5} {count:>9} {hrs:>9.2f}")

    if not args.dry_run:
        write_manifest(output_path, expanded)
    return expanded


def main() -> None:
    args = parse_args()
    jobs = [
        (args.manifest_dir / args.train_manifest, args.manifest_dir / args.out_train),
        (args.manifest_dir / args.val_manifest, args.manifest_dir / args.out_val),
        (args.manifest_dir / args.eval_manifest, args.manifest_dir / args.out_eval),
    ]
    print(f"segment_sec: {args.segment_sec}")
    print(f"include_short_final: {args.include_short_final}")
    all_records = []
    for input_path, output_path in jobs:
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {input_path}")
        all_records.extend(convert_one(input_path, output_path, args))

    print("\ncombined")
    print(f"{'label':<5} {'segments':>9} {'hours':>9}")
    for label, (count, hrs) in summarize(all_records).items():
        print(f"{label:<5} {count:>9} {hrs:>9.2f}")

    if args.dry_run:
        print("\ndry-run: no files written")
    else:
        print(f"\nwrote segment manifests to {args.manifest_dir}")


if __name__ == "__main__":
    main()
