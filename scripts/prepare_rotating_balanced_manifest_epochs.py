#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LABELS = [
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
    "yue",
    "zh",
]


@dataclass
class ManifestItem:
    record: dict[str, Any]
    label: str
    source: str
    duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-epoch balanced LID train manifests. Each epoch caps each "
            "language to a target duration, rotates high-resource languages across "
            "epochs, and source-stratifies multi-source languages such as yue."
        )
    )
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=Path("manifests/lid_train_3s.json"),
        help="Input JSONL train manifest. Existing fields such as offset are preserved.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("manifests/rotating_50h_epochs"),
        help="Output directory for epoch manifests and summaries.",
    )
    parser.add_argument(
        "--target_hours",
        type=float,
        default=50.0,
        help="Maximum target hours per label per epoch.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of per-epoch train manifests to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260513,
        help="Base random seed used for deterministic shuffling.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels to include. Use empty string to include all labels found.",
    )
    parser.add_argument(
        "--source_balance",
        choices=("equal", "proportional"),
        default="equal",
        help=(
            "How to allocate a label's per-epoch target across its sources. "
            "equal is recommended for yue source/domain balance."
        ),
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.0,
        help="Skip rows with duration below this threshold in seconds.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print summaries without writing manifests.",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help=(
            "Oversample data-limited classes to reach target_hours by cycling their segments. "
            "Without this flag, classes with less data than target_hours contribute all "
            "their segments but fall short of the per-epoch target."
        ),
    )
    return parser.parse_args()


def infer_source(audio_filepath: str, label: str) -> str:
    parts = Path(audio_filepath).parts
    lowered = [part.lower() for part in parts]
    path_lower = audio_filepath.lower()

    if "commonvoice_can_wav16k" in lowered or "commonvoice_can" in lowered:
        return "commonvoice"
    if "magicdata" in lowered:
        idx = lowered.index("magicdata")
        if idx + 1 < len(parts):
            return f"magicdata_{parts[idx + 1].lower()}"
        return "magicdata"
    if "voxlingua" in lowered:
        return "voxlingua"
    if "voxlingua_unknown" in lowered:
        return "voxlingua_unknown"

    marker = f"/{label}/"
    if marker in path_lower:
        before_label = path_lower.split(marker, 1)[0].rstrip("/")
        return Path(before_label).name or "unknown"
    return Path(audio_filepath).parent.name or "unknown"


def load_manifest(path: Path, labels: set[str] | None, min_duration: float) -> dict[str, list[ManifestItem]]:
    by_label: dict[str, list[ManifestItem]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = str(record.get("label", ""))
            if labels is not None and label not in labels:
                continue
            try:
                duration = float(record["duration"])
            except KeyError as exc:
                raise KeyError(f"{path}:{line_no} is missing duration") from exc
            if duration < min_duration:
                continue
            audio_filepath = str(record.get("audio_filepath", ""))
            source = infer_source(audio_filepath, label)
            by_label[label].append(
                ManifestItem(record=record, label=label, source=source, duration=duration)
            )
    return dict(by_label)


def hours(items: list[ManifestItem]) -> float:
    return sum(item.duration for item in items) / 3600.0


def seconds(items: list[ManifestItem]) -> float:
    return sum(item.duration for item in items)


def select_rotating(
    items: list[ManifestItem],
    target_seconds: float,
    cursor_key: tuple[str, str],
    cursors: dict[tuple[str, str], int],
    rng: random.Random,
    oversample: bool = False,
) -> list[ManifestItem]:
    if not items or target_seconds <= 0:
        return []

    available_seconds = seconds(items)
    if available_seconds <= target_seconds and not oversample:
        return list(items)

    selected: list[ManifestItem] = []
    total = 0.0
    idx = cursors[cursor_key]
    seen = 0
    while total < target_seconds:
        if not oversample and seen >= len(items):
            break
        item = items[idx]
        selected.append(item)
        total += item.duration
        idx = (idx + 1) % len(items)
        seen += 1

    cursors[cursor_key] = idx
    rng.shuffle(selected)
    return selected


def allocate_source_targets(
    source_items: dict[str, list[ManifestItem]],
    target_seconds: float,
    mode: str,
    oversample: bool = False,
) -> dict[str, float]:
    sources = sorted(source_items)
    available = {source: seconds(source_items[source]) for source in sources}
    total_available = sum(available.values())
    if total_available <= target_seconds:
        if not oversample:
            return available
        if mode == "proportional" and total_available > 0:
            return {source: target_seconds * available[source] / total_available for source in sources}
        per_source = target_seconds / len(sources)
        return {source: per_source for source in sources}

    if mode == "proportional":
        return {
            source: target_seconds * (available[source] / total_available)
            for source in sources
        }

    remaining_sources = set(sources)
    targets = {source: 0.0 for source in sources}
    remaining_target = target_seconds
    while remaining_sources and remaining_target > 0:
        quota = remaining_target / len(remaining_sources)
        exhausted: list[str] = []
        for source in sorted(remaining_sources):
            if available[source] <= quota:
                targets[source] = available[source]
                remaining_target -= available[source]
                exhausted.append(source)
        if not exhausted:
            for source in remaining_sources:
                targets[source] = quota
            break
        for source in exhausted:
            remaining_sources.remove(source)
    return targets


def write_manifest(path: Path, items: list[ManifestItem]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    labels = None if args.labels.strip() == "" else {label.strip() for label in args.labels.split(",") if label.strip()}
    target_seconds = args.target_hours * 3600.0

    by_label = load_manifest(args.input_manifest, labels, args.min_duration)
    missing = sorted(labels - set(by_label)) if labels is not None else []
    if missing:
        raise RuntimeError(f"Input manifest has no rows for labels: {', '.join(missing)}")

    rng = random.Random(args.seed)
    by_label_source: dict[str, dict[str, list[ManifestItem]]] = {}
    cursors: dict[tuple[str, str], int] = {}
    for label, items in sorted(by_label.items()):
        source_map: dict[str, list[ManifestItem]] = defaultdict(list)
        for item in items:
            source_map[item.source].append(item)
        by_label_source[label] = dict(source_map)
        for source, source_list in by_label_source[label].items():
            rng.shuffle(source_list)
            cursors[(label, source)] = 0

    print(f"input_manifest: {args.input_manifest}")
    print(f"out_dir: {args.out_dir}")
    print(f"seed: {args.seed}")
    print(f"target_hours_per_label_per_epoch: {args.target_hours:.2f}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"source_balance: {args.source_balance}")
    print(f"oversample: {args.oversample}")
    print()
    print(f"{'label':<5} {'source':<28} {'available_h':>12} {'files':>8} {'oversample_x':>13}")
    for label in sorted(by_label_source):
        label_avail = sum(hours(items) for items in by_label_source[label].values())
        for source, items in sorted(by_label_source[label].items()):
            factor = f"{args.target_hours / label_avail:.3f}x" if label_avail < args.target_hours else "1.000x"
            print(f"{label:<5} {source:<28} {hours(items):>12.2f} {len(items):>8} {factor:>13}")

    if args.dry_run:
        print("\ndry-run: no files written")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for epoch_idx in range(1, args.num_epochs + 1):
        epoch_rng = random.Random(args.seed + epoch_idx)
        epoch_items: list[ManifestItem] = []
        for label in sorted(by_label_source):
            source_map = by_label_source[label]
            source_targets = allocate_source_targets(source_map, target_seconds, args.source_balance, oversample=args.oversample)
            label_items: list[ManifestItem] = []
            for source, source_target in sorted(source_targets.items()):
                selected = select_rotating(
                    source_map[source],
                    source_target,
                    (label, source),
                    cursors,
                    epoch_rng,
                    oversample=args.oversample,
                )
                label_items.extend(selected)
                summary_rows.append(
                    {
                        "epoch": epoch_idx,
                        "label": label,
                        "source": source,
                        "hours": f"{hours(selected):.4f}",
                        "files": len(selected),
                        "available_hours": f"{hours(source_map[source]):.4f}",
                    }
                )
            epoch_rng.shuffle(label_items)
            epoch_items.extend(label_items)

        epoch_rng.shuffle(epoch_items)
        out_path = args.out_dir / f"lid_train_epoch{epoch_idx:02d}_cap{int(args.target_hours)}h.json"
        write_manifest(out_path, epoch_items)
        print(f"wrote {out_path}: {hours(epoch_items):.2f}h, {len(epoch_items)} rows")

    summary_path = args.out_dir / "epoch_source_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "label", "source", "hours", "files", "available_hours"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
