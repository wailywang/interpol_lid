#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val/eval manifests where English is split between the existing "
            "VoxLingua English examples and Common Voice English official train/dev/test."
        )
    )
    parser.add_argument("--manifest_dir", type=Path, default=Path("manifests"))
    parser.add_argument("--base_train_manifest", default="lid_train.json")
    parser.add_argument("--base_val_manifest", default="lid_val.json")
    parser.add_argument("--base_eval_manifest", default="lid_eval.json")
    parser.add_argument(
        "--commonvoice_en_root",
        type=Path,
        default=Path(
            "data/common_voice/cv-corpus-25.0-en/"
            "cv-corpus-25.0-2026-03-09/en"
        ),
    )
    parser.add_argument("--out_train", default="lid_train_en_cv_mix.json")
    parser.add_argument("--out_val", default="lid_val_en_cv_mix.json")
    parser.add_argument("--out_eval", default="lid_eval_en_cv_mix.json")
    parser.add_argument("--out_train_3s", default="lid_train_en_cv_mix_3s.json")
    parser.add_argument("--out_val_3s", default="lid_val_en_cv_mix_3s.json")
    parser.add_argument("--out_eval_3s", default="lid_eval_en_cv_mix_3s.json")
    parser.add_argument("--label", default="en")
    parser.add_argument("--train_voxlingua_hours", type=float, default=18.0)
    parser.add_argument("--train_commonvoice_hours", type=float, default=18.0)
    parser.add_argument("--val_voxlingua_hours", type=float, default=0.5)
    parser.add_argument("--val_commonvoice_hours", type=float, default=0.5)
    parser.add_argument("--eval_voxlingua_hours", type=float, default=0.5)
    parser.add_argument("--eval_commonvoice_hours", type=float, default=0.5)
    parser.add_argument("--segment_sec", type=float, default=3.0)
    parser.add_argument("--min_final_sec", type=float, default=1.0)
    parser.add_argument(
        "--drop_short_final",
        action="store_true",
        help="Drop leftover final segments instead of keeping leftovers >= --min_final_sec.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_commonvoice_durations(root: Path) -> dict[str, float]:
    path = root / "clip_durations.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Common Voice durations: {path}")

    durations = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            duration_ms = float(row["duration[ms]"])
            if duration_ms > 0:
                durations[row["clip"]] = duration_ms / 1000.0
    return durations


def load_commonvoice_split(
    root: Path,
    split: str,
    label: str,
    durations: dict[str, float],
) -> list[dict]:
    clips_dir = root / "clips"
    tsv_path = root / f"{split}.tsv"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing Common Voice clips dir: {clips_dir}")
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing Common Voice TSV: {tsv_path}")

    records = []
    missing_duration = 0
    missing_file = 0
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rel_path = row["path"]
            duration = durations.get(rel_path)
            if duration is None:
                missing_duration += 1
                continue
            audio_path = clips_dir / rel_path
            if not audio_path.is_file():
                missing_file += 1
                continue
            records.append(
                {
                    "audio_filepath": str(audio_path.absolute()),
                    "duration": round(duration, 3),
                    "label": label,
                }
            )

    if missing_duration or missing_file:
        print(
            f"warning: Common Voice {split}: skipped "
            f"{missing_duration} missing durations, {missing_file} missing files"
        )
    return records


def select_until_hours(records: list[dict], target_hours: float) -> list[dict]:
    target_seconds = target_hours * 3600.0
    selected = []
    total = 0.0
    for record in records:
        if total >= target_seconds:
            break
        selected.append(record)
        total += float(record["duration"])
    return selected


def expand_to_3s(
    records: list[dict],
    segment_sec: float,
    min_final_sec: float,
    include_short_final: bool,
) -> list[dict]:
    segments = []
    for record in records:
        duration = float(record["duration"])
        full_segments = int(duration // segment_sec)
        base_offset = float(record.get("offset", 0.0))
        for idx in range(full_segments):
            segments.append(
                {
                    "audio_filepath": record["audio_filepath"],
                    "offset": round(base_offset + idx * segment_sec, 3),
                    "duration": round(segment_sec, 3),
                    "label": record["label"],
                }
            )
        leftover = duration - full_segments * segment_sec
        if include_short_final and leftover >= min_final_sec:
            segments.append(
                {
                    "audio_filepath": record["audio_filepath"],
                    "offset": round(base_offset + full_segments * segment_sec, 3),
                    "duration": round(leftover, 3),
                    "label": record["label"],
                }
            )
    return segments


def hours(records: list[dict]) -> float:
    return sum(float(record["duration"]) for record in records) / 3600.0


def summarize(name: str, records: list[dict]) -> None:
    counts = defaultdict(int)
    seconds = defaultdict(float)
    for record in records:
        label = record["label"]
        counts[label] += 1
        seconds[label] += float(record["duration"])

    print(f"\n{name}")
    print(f"{'label':<5} {'hours':>8} {'records':>8}")
    for label in sorted(counts):
        print(f"{label:<5} {seconds[label] / 3600.0:>8.2f} {counts[label]:>8}")


def build_split(
    base_records: list[dict],
    cv_records: list[dict],
    label: str,
    voxlingua_hours: float,
    commonvoice_hours: float,
    rng: random.Random,
) -> list[dict]:
    non_label = [record for record in base_records if record.get("label") != label]
    base_label = [record for record in base_records if record.get("label") == label]

    rng.shuffle(base_label)
    rng.shuffle(cv_records)
    selected = (
        non_label
        + select_until_hours(base_label, voxlingua_hours)
        + select_until_hours(cv_records, commonvoice_hours)
    )
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    base_train_path = args.manifest_dir / args.base_train_manifest
    base_val_path = args.manifest_dir / args.base_val_manifest
    base_eval_path = args.manifest_dir / args.base_eval_manifest
    for path in (base_train_path, base_val_path, base_eval_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing manifest: {path}")

    train_records = read_jsonl(base_train_path)
    val_records = read_jsonl(base_val_path)
    eval_records = read_jsonl(base_eval_path)
    cv_durations = load_commonvoice_durations(args.commonvoice_en_root)
    cv_train = load_commonvoice_split(args.commonvoice_en_root, "train", args.label, cv_durations)
    cv_dev = load_commonvoice_split(args.commonvoice_en_root, "dev", args.label, cv_durations)
    cv_test = load_commonvoice_split(args.commonvoice_en_root, "test", args.label, cv_durations)

    mixed_train = build_split(
        train_records,
        cv_train,
        args.label,
        args.train_voxlingua_hours,
        args.train_commonvoice_hours,
        rng,
    )
    mixed_val = build_split(
        val_records,
        cv_dev,
        args.label,
        args.val_voxlingua_hours,
        args.val_commonvoice_hours,
        rng,
    )
    mixed_eval = build_split(
        eval_records,
        cv_test,
        args.label,
        args.eval_voxlingua_hours,
        args.eval_commonvoice_hours,
        rng,
    )
    include_short_final = not args.drop_short_final
    mixed_train_3s = expand_to_3s(
        mixed_train, args.segment_sec, args.min_final_sec, include_short_final
    )
    mixed_val_3s = expand_to_3s(
        mixed_val, args.segment_sec, args.min_final_sec, include_short_final
    )
    mixed_eval_3s = expand_to_3s(
        mixed_eval, args.segment_sec, args.min_final_sec, include_short_final
    )

    print(f"seed: {args.seed}")
    print(
        f"{args.label} train target: "
        f"voxlingua={args.train_voxlingua_hours:.2f}h "
        f"commonvoice={args.train_commonvoice_hours:.2f}h"
    )
    print(
        f"{args.label} val target: "
        f"voxlingua={args.val_voxlingua_hours:.2f}h "
        f"commonvoice={args.val_commonvoice_hours:.2f}h"
    )
    print(
        f"{args.label} eval target: "
        f"voxlingua={args.eval_voxlingua_hours:.2f}h "
        f"commonvoice={args.eval_commonvoice_hours:.2f}h"
    )
    print(f"segment_sec: {args.segment_sec}")
    print(f"include_short_final: {include_short_final}")
    print(f"min_final_sec: {args.min_final_sec}")
    summarize("train raw", mixed_train)
    summarize("val raw", mixed_val)
    summarize("eval raw", mixed_eval)
    summarize("train 3s", mixed_train_3s)
    summarize("val 3s", mixed_val_3s)
    summarize("eval 3s", mixed_eval_3s)

    if args.dry_run:
        print("\ndry-run: no files written")
        return

    write_jsonl(args.manifest_dir / args.out_train, mixed_train)
    write_jsonl(args.manifest_dir / args.out_val, mixed_val)
    write_jsonl(args.manifest_dir / args.out_eval, mixed_eval)
    write_jsonl(args.manifest_dir / args.out_train_3s, mixed_train_3s)
    write_jsonl(args.manifest_dir / args.out_val_3s, mixed_val_3s)
    write_jsonl(args.manifest_dir / args.out_eval_3s, mixed_eval_3s)
    print(f"\nwrote manifests to {args.manifest_dir}")


if __name__ == "__main__":
    main()
