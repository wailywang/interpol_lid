#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABELS = [
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

VOXLINGUA_TIME_RE = re.compile(r"---([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\.wav$")


@dataclass
class AudioItem:
    path: Path
    label: str
    source: str
    duration: float
    channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan full LID audio data and generate source-stratified, per-epoch "
            "balanced 3s train manifests."
        )
    )
    parser.add_argument("--voxlingua_root", type=Path, default=Path("data/voxlingua"))
    parser.add_argument(
        "--include_commonvoice_en",
        action="store_true",
        help="Add Common Voice English official train split as an en source.",
    )
    parser.add_argument(
        "--commonvoice_en_root",
        type=Path,
        default=Path("data/common_voice/cv-corpus-25.0-en/cv-corpus-25.0-2026-03-09/en"),
    )
    parser.add_argument(
        "--commonvoice_yue_clips",
        type=Path,
        default=Path("data/commonvoice_yue_v25/cv-corpus-25.0-2026-03-09/yue/clips"),
    )
    parser.add_argument(
        "--magicdata_roots",
        default="data/magicdata/dailyuse,data/magicdata/conversational,data/magicdata_vehicle_mono/vehicle",
        help="Comma-separated MagicData roots to include as yue sources.",
    )
    parser.add_argument("--include_unknown", action="store_true", help="Add a 19th unknown class.")
    parser.add_argument("--unknown_label", default="unknown")
    parser.add_argument("--voxlingua_unknown_root", type=Path, default=Path("data/voxlingua_unknown"))
    parser.add_argument("--unknown_voxlingua_hours", type=float, default=50.0)
    parser.add_argument(
        "--exclude_manifests",
        default="manifests/lid_val_3s.json,manifests/lid_eval_3s.json",
        help="Comma-separated manifests whose audio_filepath values should be excluded.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("manifests/full_rotating_50h_epochs"))
    parser.add_argument("--target_hours", type=float, default=50.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--segment_sec", type=float, default=3.0)
    parser.add_argument("--min_final_sec", type=float, default=1.0)
    parser.add_argument(
        "--required_channels",
        type=int,
        default=1,
        help="Only include audio with this channel count. Use 0 to disable channel filtering.",
    )
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument(
        "--source_balance",
        choices=("equal", "proportional"),
        default="equal",
        help="Use equal for domain-balanced yue sampling.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--oversample",
        action="store_true",
        help=(
            "Oversample data-limited classes to reach target_hours by cycling their audio. "
            "Without this flag, classes with less audio than target_hours contribute all "
            "their data but fall short of the per-epoch target (current behaviour)."
        ),
    )
    return parser.parse_args()


def wav_info(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / float(f.getframerate()), f.getnchannels()


def voxlingua_info(path: Path) -> tuple[float, int]:
    match = VOXLINGUA_TIME_RE.search(path.name)
    if match:
        # VoxLingua clips in this pipeline are treated as mono; avoid a costly
        # header scan over every high-resource language file.
        return float(match.group(2)) - float(match.group(1)), 1
    return wav_info(path)


def load_excluded_audio(manifest_paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                excluded.add(str(Path(record["audio_filepath"]).resolve()))
    return excluded


def channel_allowed(channels: int, required_channels: int) -> bool:
    return required_channels <= 0 or channels == required_channels


def scan_voxlingua(
    root: Path,
    excluded: set[str],
    required_channels: int,
    skipped_channels: dict[str, int],
) -> dict[str, list[AudioItem]]:
    by_label: dict[str, list[AudioItem]] = defaultdict(list)
    for label in LABELS:
        if label == "yue":
            continue
        lang_dir = root / label
        if not lang_dir.is_dir():
            raise FileNotFoundError(f"Missing VoxLingua directory: {lang_dir}")
        for path in lang_dir.glob("*.wav"):
            resolved = str(path.resolve())
            if resolved in excluded:
                continue
            duration, channels = voxlingua_info(path)
            if not channel_allowed(channels, required_channels):
                skipped_channels[f"voxlingua:{channels}ch"] += 1
                continue
            if duration > 0:
                by_label[label].append(AudioItem(path.resolve(), label, "voxlingua", duration, channels))
    return by_label


def load_commonvoice_durations(root: Path) -> dict[str, float]:
    duration_path = root / "clip_durations.tsv"
    if not duration_path.is_file():
        raise FileNotFoundError(f"Missing Common Voice duration file: {duration_path}")

    durations: dict[str, float] = {}
    with duration_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            duration_ms = float(row["duration[ms]"])
            if duration_ms > 0:
                durations[row["clip"]] = duration_ms / 1000.0
    return durations


def scan_commonvoice_en_train(
    root: Path,
    excluded: set[str],
) -> list[AudioItem]:
    clips_dir = root / "clips"
    tsv_path = root / "train.tsv"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing Common Voice English clips directory: {clips_dir}")
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing Common Voice English train TSV: {tsv_path}")

    durations = load_commonvoice_durations(root)
    items: list[AudioItem] = []
    missing_duration = 0
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rel_path = row["path"]
            duration = durations.get(rel_path)
            if duration is None:
                missing_duration += 1
                continue
            path = (clips_dir / rel_path).resolve()
            if str(path) in excluded:
                continue
            items.append(AudioItem(path, "en", "commonvoice_en", duration, 1))

    if missing_duration:
        print(f"warning: Common Voice en train skipped {missing_duration} rows without duration")
    return items


def scan_yue(
    commonvoice_clips: Path,
    magic_roots: list[Path],
    excluded: set[str],
    required_channels: int,
    skipped_channels: dict[str, int],
) -> list[AudioItem]:
    items: list[AudioItem] = []
    if commonvoice_clips.is_dir():
        for path in commonvoice_clips.glob("*.wav"):
            resolved = str(path.resolve())
            if resolved not in excluded:
                duration, channels = wav_info(path)
                if not channel_allowed(channels, required_channels):
                    skipped_channels[f"commonvoice:{channels}ch"] += 1
                    continue
                items.append(AudioItem(path.resolve(), "yue", "commonvoice", duration, channels))
    for root in magic_roots:
        if not root.is_dir():
            continue
        source = f"magicdata_{root.name}"
        for path in root.rglob("*.wav"):
            resolved = str(path.resolve())
            if resolved not in excluded:
                duration, channels = wav_info(path)
                if not channel_allowed(channels, required_channels):
                    skipped_channels[f"{source}:{channels}ch"] += 1
                    continue
                if duration > 0:
                    items.append(AudioItem(path.resolve(), "yue", source, duration, channels))
    return items


def scan_voxlingua_unknown(
    root: Path,
    label: str,
    excluded: set[str],
    required_channels: int,
    skipped_channels: dict[str, int],
) -> list[AudioItem]:
    items: list[AudioItem] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Missing VoxLingua unknown root: {root}")
    for lang_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        source = f"voxlingua_unknown_{lang_dir.name}"
        for path in lang_dir.glob("*.wav"):
            resolved = str(path.resolve())
            if resolved in excluded:
                continue
            duration, channels = voxlingua_info(path)
            if not channel_allowed(channels, required_channels):
                skipped_channels[f"{source}:{channels}ch"] += 1
                continue
            if duration > 0:
                items.append(AudioItem(path.resolve(), label, source, duration, channels))
    return items



def hours(items: list[AudioItem] | list[dict[str, Any]]) -> float:
    return sum(float(item.duration if isinstance(item, AudioItem) else item["duration"]) for item in items) / 3600.0


def seconds(items: list[AudioItem]) -> float:
    return sum(item.duration for item in items)


def allocate_targets(
    source_items: dict[str, list[AudioItem]],
    target_seconds: float,
    mode: str,
    oversample: bool = False,
) -> dict[str, float]:
    sources = sorted(source_items)
    available = {source: seconds(source_items[source]) for source in sources}
    total = sum(available.values())
    if total <= target_seconds:
        if not oversample:
            return available
        # Oversample: allocate target_seconds across sources so select_rotating can cycle.
        if mode == "proportional" and total > 0:
            return {source: target_seconds * available[source] / total for source in sources}
        # Equal: give each source an equal slice of the target.
        per_source = target_seconds / len(sources)
        return {source: per_source for source in sources}
    if mode == "proportional":
        return {source: target_seconds * available[source] / total for source in sources}

    targets = {source: 0.0 for source in sources}
    remaining_sources = set(sources)
    remaining_target = target_seconds
    while remaining_sources and remaining_target > 0:
        quota = remaining_target / len(remaining_sources)
        exhausted = [source for source in remaining_sources if available[source] <= quota]
        if not exhausted:
            for source in remaining_sources:
                targets[source] = quota
            break
        for source in exhausted:
            targets[source] = available[source]
            remaining_target -= available[source]
            remaining_sources.remove(source)
    return targets


def select_rotating(
    items: list[AudioItem],
    target_seconds: float,
    key: tuple[str, str],
    cursors: dict[tuple[str, str], int],
    oversample: bool = False,
) -> list[AudioItem]:
    if not items:
        return []
    avail = seconds(items)
    if avail <= target_seconds and not oversample:
        return list(items)
    selected: list[AudioItem] = []
    total = 0.0
    idx = cursors[key]
    seen = 0
    while total < target_seconds:
        if not oversample and seen >= len(items):
            break
        item = items[idx]
        selected.append(item)
        total += item.duration
        idx = (idx + 1) % len(items)
        seen += 1
    cursors[key] = idx
    return selected


def select_unknown_rotating(
    source_map: dict[str, list[AudioItem]],
    voxlingua_seconds: float,
    label: str,
    cursors: dict[tuple[str, str], int],
    oversample: bool = False,
) -> list[AudioItem]:
    selected: list[AudioItem] = []
    vox_sources = {
        source: items for source, items in source_map.items() if source.startswith("voxlingua_unknown_")
    }
    for source, target in allocate_targets(vox_sources, voxlingua_seconds, "equal", oversample=oversample).items():
        selected.extend(select_rotating(vox_sources[source], target, (label, source), cursors, oversample=oversample))
    return selected


def expand_to_segments(items: list[AudioItem], segment_sec: float, min_final_sec: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        full_segments = int(item.duration // segment_sec)
        for idx in range(full_segments):
            records.append(
                {
                    "audio_filepath": str(item.path),
                    "offset": round(idx * segment_sec, 3),
                    "duration": round(segment_sec, 3),
                    "label": item.label,
                }
            )
        leftover = item.duration - full_segments * segment_sec
        if leftover >= min_final_sec:
            records.append(
                {
                    "audio_filepath": str(item.path),
                    "offset": round(full_segments * segment_sec, 3),
                    "duration": round(leftover, 3),
                    "label": item.label,
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    exclude_paths = [Path(p) for p in args.exclude_manifests.split(",") if p.strip()]
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {args.out_dir}", flush=True)
    print(f"loading excluded manifests: {args.exclude_manifests}", flush=True)
    excluded = load_excluded_audio(exclude_paths)
    magic_roots = [Path(p) for p in args.magicdata_roots.split(",") if p.strip()]

    skipped_channels: dict[str, int] = defaultdict(int)
    labels = list(LABELS)
    if args.include_unknown and args.unknown_label not in labels:
        labels.append(args.unknown_label)

    print(f"scanning VoxLingua known labels from: {args.voxlingua_root}", flush=True)
    by_label = scan_voxlingua(args.voxlingua_root, excluded, args.required_channels, skipped_channels)
    if args.include_commonvoice_en:
        print(f"scanning Common Voice English train from: {args.commonvoice_en_root}", flush=True)
        by_label["en"].extend(scan_commonvoice_en_train(args.commonvoice_en_root, excluded))
    print("scanning yue sources", flush=True)
    by_label["yue"] = scan_yue(
        args.commonvoice_yue_clips,
        magic_roots,
        excluded,
        args.required_channels,
        skipped_channels,
    )
    if args.include_unknown:
        print(f"scanning unknown sources from: {args.voxlingua_unknown_root}", flush=True)
        by_label[args.unknown_label] = scan_voxlingua_unknown(
            args.voxlingua_unknown_root,
            args.unknown_label,
            excluded,
            args.required_channels,
            skipped_channels,
        )

    by_label_source: dict[str, dict[str, list[AudioItem]]] = {}
    cursors: dict[tuple[str, str], int] = {}
    for label in labels:
        source_map: dict[str, list[AudioItem]] = defaultdict(list)
        for item in by_label.get(label, []):
            source_map[item.source].append(item)
        if not source_map:
            raise RuntimeError(f"No audio found for label {label}")
        for source_items in source_map.values():
            rng.shuffle(source_items)
        by_label_source[label] = dict(source_map)
        for source in source_map:
            cursors[(label, source)] = 0

    print(f"excluded_audio_files: {len(excluded)}")
    print(f"target_hours_per_label_per_epoch: {args.target_hours:.2f}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"source_balance: {args.source_balance}")
    print(f"oversample: {args.oversample}")
    print(f"required_channels: {args.required_channels if args.required_channels > 0 else 'any'}")
    if args.include_unknown:
        print(f"unknown_label: {args.unknown_label}")
        print(f"unknown_voxlingua_hours_per_epoch: {args.unknown_voxlingua_hours:.2f}")
    if skipped_channels:
        print("skipped_non_matching_channels:")
        for key, count in sorted(skipped_channels.items()):
            print(f"  {key}: {count}")
    print()
    print(f"{'label':<5} {'source':<28} {'available_h':>12} {'files':>8} {'oversample_x':>13}")
    for label in labels:
        label_avail = sum(hours(items) for items in by_label_source[label].values())
        target_h = args.target_hours if label != (args.unknown_label if args.include_unknown else "") else args.unknown_voxlingua_hours
        for source, items in sorted(by_label_source[label].items()):
            factor = f"{target_h / label_avail:.3f}x" if label_avail < target_h else "1.000x"
            print(f"{label:<5} {source:<28} {hours(items):>12.2f} {len(items):>8} {factor:>13}")

    if args.dry_run:
        print("\ndry-run: no files written")
        return

    target_seconds = args.target_hours * 3600.0
    summary_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.num_epochs + 1):
        epoch_audio: list[AudioItem] = []
        for label in labels:
            source_map = by_label_source[label]
            if args.include_unknown and label == args.unknown_label:
                selected_by_source: dict[str, list[AudioItem]] = defaultdict(list)
                selected = select_unknown_rotating(
                    source_map,
                    args.unknown_voxlingua_hours * 3600.0,
                    label,
                    cursors,
                    oversample=args.oversample,
                )
                for item in selected:
                    selected_by_source[item.source].append(item)
                for source, source_selected in sorted(selected_by_source.items()):
                    summary_rows.append(
                        {
                            "epoch": epoch,
                            "label": label,
                            "source": source,
                            "audio_hours": f"{hours(source_selected):.4f}",
                            "audio_files": len(source_selected),
                            "available_hours": f"{hours(source_map[source]):.4f}",
                        }
                    )
                epoch_audio.extend(selected)
                continue

            targets = allocate_targets(source_map, target_seconds, args.source_balance, oversample=args.oversample)
            for source, source_target in sorted(targets.items()):
                selected = select_rotating(source_map[source], source_target, (label, source), cursors, oversample=args.oversample)
                epoch_audio.extend(selected)
                summary_rows.append(
                    {
                        "epoch": epoch,
                        "label": label,
                        "source": source,
                        "audio_hours": f"{hours(selected):.4f}",
                        "audio_files": len(selected),
                        "available_hours": f"{hours(source_map[source]):.4f}",
                    }
                )
        epoch_rng = random.Random(args.seed + epoch)
        epoch_rng.shuffle(epoch_audio)
        records = expand_to_segments(epoch_audio, args.segment_sec, args.min_final_sec)
        epoch_rng.shuffle(records)
        out_path = args.out_dir / f"lid_train_epoch{epoch:02d}_cap{int(args.target_hours)}h_3s.json"
        write_jsonl(out_path, records)
        print(f"wrote {out_path}: audio={hours(epoch_audio):.2f}h segments={hours(records):.2f}h rows={len(records)}")

    summary_path = args.out_dir / "epoch_source_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "label", "source", "audio_hours", "audio_files", "available_hours"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
