#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_full_data_rotating_lid_epochs import (
    expand_to_segments,
    hours,
    scan_voxlingua_unknown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed 19-class val/eval manifests by appending held-out unknown examples."
    )
    parser.add_argument("--base_val_manifest", type=Path, default=Path("manifests/lid_val_3s.json"))
    parser.add_argument("--base_eval_manifest", type=Path, default=Path("manifests/lid_eval_3s.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("manifests/heldout_19class"))
    parser.add_argument("--unknown_label", default="unknown")
    parser.add_argument("--voxlingua_unknown_root", type=Path, default=Path("data/voxlingua_unknown"))
    parser.add_argument("--val_voxlingua_unknown_hours", type=float, default=1.0)
    parser.add_argument("--eval_voxlingua_unknown_hours", type=float, default=1.0)
    parser.add_argument("--segment_sec", type=float, default=3.0)
    parser.add_argument("--min_final_sec", type=float, default=1.0)
    parser.add_argument("--required_channels", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_hours(items, target_hours: float):
    selected = []
    total = 0.0
    target_seconds = target_hours * 3600.0
    for item in items:
        if total >= target_seconds:
            break
        selected.append(item)
        total += item.duration
    return selected


def split_source_items(items, val_hours: float, eval_hours: float):
    val_items = select_hours(items, val_hours)
    remaining = items[len(val_items) :]
    eval_items = select_hours(remaining, eval_hours)
    return val_items, eval_items


def split_voxlingua_unknown_by_language(items, val_hours: float, eval_hours: float, rng: random.Random):
    by_source = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)
    val_selected = []
    eval_selected = []
    sources = sorted(by_source)
    for source in sources:
        source_items = by_source[source]
        rng.shuffle(source_items)
        val_part, eval_part = split_source_items(
            source_items,
            val_hours / len(sources),
            eval_hours / len(sources),
        )
        val_selected.extend(val_part)
        eval_selected.extend(eval_part)
    return val_selected, eval_selected


def summarize(name: str, records: list[dict[str, Any]]) -> None:
    by_label = defaultdict(float)
    counts = defaultdict(int)
    for record in records:
        label = record["label"]
        by_label[label] += float(record["duration"])
        counts[label] += 1
    print(f"\n{name}")
    print(f"{'label':<10} {'hours':>8} {'rows':>8}")
    for label in sorted(by_label):
        print(f"{label:<10} {by_label[label] / 3600.0:>8.2f} {counts[label]:>8}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    skipped_channels: dict[str, int] = defaultdict(int)
    excluded: set[str] = set()

    vox_items = scan_voxlingua_unknown(
        args.voxlingua_unknown_root,
        args.unknown_label,
        excluded,
        args.required_channels,
        skipped_channels,
    )

    vox_val, vox_eval = split_voxlingua_unknown_by_language(
        vox_items,
        args.val_voxlingua_unknown_hours,
        args.eval_voxlingua_unknown_hours,
        rng,
    )

    val_unknown_records = expand_to_segments(vox_val, args.segment_sec, args.min_final_sec)
    eval_unknown_records = expand_to_segments(vox_eval, args.segment_sec, args.min_final_sec)
    rng.shuffle(val_unknown_records)
    rng.shuffle(eval_unknown_records)

    val_records = read_jsonl(args.base_val_manifest) + val_unknown_records
    eval_records = read_jsonl(args.base_eval_manifest) + eval_unknown_records
    rng.shuffle(val_records)
    rng.shuffle(eval_records)

    print(f"voxlingua_unknown available: {hours(vox_items):.2f}h, files={len(vox_items)}")
    print(f"val unknown audio: vox={hours(vox_val):.2f}h")
    print(f"eval unknown audio: vox={hours(vox_eval):.2f}h")
    if skipped_channels:
        print("skipped_non_matching_channels:")
        for key, count in sorted(skipped_channels.items()):
            print(f"  {key}: {count}")
    summarize("val_19class", val_records)
    summarize("eval_19class", eval_records)

    if args.dry_run:
        print("\ndry-run: no files written")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_path = args.out_dir / "lid_val_19class_3s.json"
    eval_path = args.out_dir / "lid_eval_19class_3s.json"
    write_jsonl(val_path, val_records)
    write_jsonl(eval_path, eval_records)
    print(f"\nwrote {val_path}")
    print(f"wrote {eval_path}")


if __name__ == "__main__":
    main()
