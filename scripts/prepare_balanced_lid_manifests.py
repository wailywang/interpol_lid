#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


VOXLINGUA_LABELS = [
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "ja",
    "km",
    "ko",
    "ms",
    "pt",
    "ru",
    "th",
    "tl",
    "tr",
    "vi",
    "zh",
]
COMMONVOICE_LABEL = "yue"
VOXLINGUA_TIME_RE = re.compile(r"---([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\.wav$")


@dataclass(frozen=True)
class AudioItem:
    path: Path
    duration: float
    label: str
    source_split: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare balanced train/val/eval manifests for VoxLingua + Common Voice Cantonese LID."
    )
    parser.add_argument(
        "--voxlingua_root",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/data/voxlingua"),
        help="Root directory containing VoxLingua language subdirectories.",
    )
    parser.add_argument(
        "--commonvoice_yue_root",
        type=Path,
        default=Path(
            "/export/home2/wa0009xi/ots-lid/data/commonvoice_yue_v25/"
            "cv-corpus-25.0-2026-03-09/yue"
        ),
        help="Common Voice Cantonese locale directory containing train/dev/test TSVs and clips/.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/manifests"),
        help="Directory for output manifests and summary.",
    )
    parser.add_argument("--train_hours", type=float, default=36.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--eval_hours", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--commonvoice_source",
        choices=("validated", "official_splits"),
        default="validated",
        help=(
            "Use Common Voice validated.tsv and randomly split it, or preserve the official "
            "train/dev/test TSVs. Use validated for the balanced 36/1/1h yue plan."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned sampling summary without writing manifests.",
    )
    return parser.parse_args()


def voxlingua_duration_from_name(path: Path) -> float:
    match = VOXLINGUA_TIME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse VoxLingua duration from filename: {path}")
    start = float(match.group(1))
    end = float(match.group(2))
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Non-positive duration parsed from filename: {path}")
    return duration


def load_voxlingua_items(root: Path, labels: list[str]) -> dict[str, list[AudioItem]]:
    by_label: dict[str, list[AudioItem]] = {}
    for label in labels:
        lang_dir = root / label
        if not lang_dir.is_dir():
            raise FileNotFoundError(f"Missing VoxLingua directory: {lang_dir}")
        items = [
            AudioItem(path=path.absolute(), duration=voxlingua_duration_from_name(path), label=label)
            for path in lang_dir.glob("*.wav")
        ]
        if not items:
            raise RuntimeError(f"No wav files found for VoxLingua label {label}: {lang_dir}")
        by_label[label] = items
    return by_label


def load_commonvoice_durations(root: Path) -> dict[str, float]:
    duration_path = root / "clip_durations.tsv"
    if not duration_path.is_file():
        raise FileNotFoundError(f"Missing Common Voice duration file: {duration_path}")

    durations: dict[str, float] = {}
    with duration_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip = row["clip"]
            duration_ms = float(row["duration[ms]"])
            if duration_ms > 0:
                durations[clip] = duration_ms / 1000.0
    return durations


def load_commonvoice_tsv(
    root: Path,
    tsv_name: str,
    durations: dict[str, float],
    existing_clips: dict[str, str],
) -> list[AudioItem]:
    tsv_path = root / f"{tsv_name}.tsv"
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing Common Voice split file: {tsv_path}")

    clips_dir = root / "clips"
    items: list[AudioItem] = []
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
            stem = Path(rel_path).stem
            actual_name = existing_clips.get(stem)
            if actual_name is None:
                missing_file += 1
                continue
            audio_path = clips_dir / actual_name
            items.append(
                AudioItem(
                    path=audio_path.absolute(),
                    duration=duration,
                    label=COMMONVOICE_LABEL,
                    source_split=tsv_name,
                )
            )

    if missing_duration or missing_file:
        print(
            f"warning: Common Voice {tsv_name}: skipped "
            f"{missing_duration} rows without duration and {missing_file} rows without audio file"
        )
    return items


def load_commonvoice_yue_items(root: Path, source: str) -> dict[str, list[AudioItem]] | list[AudioItem]:
    durations = load_commonvoice_durations(root)
    clips_dir = root / "clips"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing Common Voice clips directory: {clips_dir}")
    # Map stem → actual filename to handle MP3→WAV conversion (TSV references .mp3, clips may be .wav)
    stem_to_clip = {Path(entry.name).stem: entry.name for entry in os.scandir(clips_dir)}
    existing_clips = stem_to_clip
    if source == "validated":
        return load_commonvoice_tsv(root, "validated", durations, existing_clips)
    return {
        "train": load_commonvoice_tsv(root, "train", durations, existing_clips),
        "val": load_commonvoice_tsv(root, "dev", durations, existing_clips),
        "eval": load_commonvoice_tsv(root, "test", durations, existing_clips),
    }


def select_until_hours(items: list[AudioItem], target_hours: float) -> list[AudioItem]:
    target_seconds = target_hours * 3600.0
    selected: list[AudioItem] = []
    total = 0.0
    for item in items:
        if total >= target_seconds:
            break
        selected.append(item)
        total += item.duration
    return selected


def split_items_randomly(
    items: list[AudioItem],
    rng: random.Random,
    train_hours: float,
    val_hours: float,
    eval_hours: float,
) -> dict[str, list[AudioItem]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    eval_items = select_until_hours(shuffled, eval_hours)
    remaining = shuffled[len(eval_items) :]
    val_items = select_until_hours(remaining, val_hours)
    remaining = remaining[len(val_items) :]
    train_items = select_until_hours(remaining, train_hours)
    return {"train": train_items, "val": val_items, "eval": eval_items}


def split_commonvoice_preserving_official_splits(
    cv_items: dict[str, list[AudioItem]],
    rng: random.Random,
    train_hours: float,
    val_hours: float,
    eval_hours: float,
) -> dict[str, list[AudioItem]]:
    selected: dict[str, list[AudioItem]] = {}
    targets = {"train": train_hours, "val": val_hours, "eval": eval_hours}
    for split_name, target_hours in targets.items():
        items = list(cv_items[split_name])
        rng.shuffle(items)
        selected[split_name] = select_until_hours(items, target_hours)
    return selected


def hours(items: list[AudioItem]) -> float:
    return sum(item.duration for item in items) / 3600.0


def write_manifest(path: Path, items: list[AudioItem]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            record = {
                "audio_filepath": str(item.path),
                "duration": round(item.duration, 3),
                "label": item.label,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    target_total = args.train_hours + args.val_hours + args.eval_hours

    rng = random.Random(args.seed)
    print(f"seed: {args.seed}")
    print(
        f"target per label: train={args.train_hours:.2f}h "
        f"val={args.val_hours:.2f}h eval={args.eval_hours:.2f}h total={target_total:.2f}h"
    )

    vox_items = load_voxlingua_items(args.voxlingua_root, VOXLINGUA_LABELS)
    cv_items = load_commonvoice_yue_items(args.commonvoice_yue_root, args.commonvoice_source)

    split_by_label: dict[str, dict[str, list[AudioItem]]] = {}
    for label in VOXLINGUA_LABELS:
        available = hours(vox_items[label])
        if available < target_total:
            raise RuntimeError(
                f"Label {label} has only {available:.2f}h, below target {target_total:.2f}h"
            )
        split_by_label[label] = split_items_randomly(
            vox_items[label], rng, args.train_hours, args.val_hours, args.eval_hours
        )

    if args.commonvoice_source == "validated":
        cv_available_total = hours(cv_items)
        if cv_available_total < target_total:
            raise RuntimeError(
                f"Common Voice yue validated has only {cv_available_total:.2f}h, "
                f"below target {target_total:.2f}h"
            )
        split_by_label[COMMONVOICE_LABEL] = split_items_randomly(
            cv_items, rng, args.train_hours, args.val_hours, args.eval_hours
        )
    else:
        cv_available = {name: hours(items) for name, items in cv_items.items()}
        if cv_available["train"] < args.train_hours:
            raise RuntimeError(f"Common Voice yue train has only {cv_available['train']:.2f}h")
        if cv_available["val"] < args.val_hours:
            raise RuntimeError(f"Common Voice yue dev has only {cv_available['val']:.2f}h")
        if cv_available["eval"] < args.eval_hours:
            raise RuntimeError(f"Common Voice yue test has only {cv_available['eval']:.2f}h")
        split_by_label[COMMONVOICE_LABEL] = split_commonvoice_preserving_official_splits(
            cv_items, rng, args.train_hours, args.val_hours, args.eval_hours
        )

    combined: dict[str, list[AudioItem]] = defaultdict(list)
    for label in VOXLINGUA_LABELS + [COMMONVOICE_LABEL]:
        for split_name in ("train", "val", "eval"):
            combined[split_name].extend(split_by_label[label][split_name])

    for split_name in ("train", "val", "eval"):
        rng.shuffle(combined[split_name])

    print()
    print(f"{'label':<5} {'available_h':>12} {'train_h':>9} {'val_h':>7} {'eval_h':>8} {'files':>8}")
    for label in VOXLINGUA_LABELS:
        splits = split_by_label[label]
        print(
            f"{label:<5} {hours(vox_items[label]):>12.2f} "
            f"{hours(splits['train']):>9.2f} {hours(splits['val']):>7.2f} "
            f"{hours(splits['eval']):>8.2f} "
            f"{sum(len(splits[s]) for s in ('train', 'val', 'eval')):>8}"
        )
    yue_splits = split_by_label[COMMONVOICE_LABEL]
    yue_available = (
        hours(cv_items)
        if args.commonvoice_source == "validated"
        else sum(hours(items) for items in cv_items.values())
    )
    print(
        f"{COMMONVOICE_LABEL:<5} {yue_available:>12.2f} "
        f"{hours(yue_splits['train']):>9.2f} {hours(yue_splits['val']):>7.2f} "
        f"{hours(yue_splits['eval']):>8.2f} "
        f"{sum(len(yue_splits[s]) for s in ('train', 'val', 'eval')):>8}"
    )

    print()
    print(f"{'split':<6} {'hours':>9} {'files':>8}")
    for split_name in ("train", "val", "eval"):
        print(f"{split_name:<6} {hours(combined[split_name]):>9.2f} {len(combined[split_name]):>8}")

    if args.dry_run:
        print("\ndry-run: no manifest files written")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.out_dir / "lid_train.json", combined["train"])
    write_manifest(args.out_dir / "lid_val.json", combined["val"])
    write_manifest(args.out_dir / "lid_eval.json", combined["eval"])
    print(f"\nwrote manifests to {args.out_dir}")


if __name__ == "__main__":
    main()
