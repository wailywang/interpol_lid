#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


_INFO_CACHE: dict[str, tuple[int, int, int]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean NeMo LID JSONL manifests by removing too-short, out-of-range, "
            "and near-silent audio segments."
        )
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument(
        "--min_rms_db",
        type=float,
        default=-60.0,
        help="Drop segments with RMS dBFS below this threshold.",
    )
    parser.add_argument(
        "--edge_tolerance_sec",
        type=float,
        default=0.02,
        help="Allowed manifest/header mismatch before treating offset+duration as out of range.",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument(
        "--skip_rms",
        action="store_true",
        help="Only validate duration and offset/header boundaries.",
    )
    parser.add_argument("--summary_name", default="clean_summary.csv")
    return parser.parse_args()


def audio_info(path: str) -> tuple[int, int, int]:
    cached = _INFO_CACHE.get(path)
    if cached is not None:
        return cached
    info = sf.info(path)
    value = (int(info.frames), int(info.samplerate), int(info.channels))
    _INFO_CACHE[path] = value
    return value


def rms_db_for_segment(path: str, offset: float, duration: float, frames: int, sample_rate: int) -> float:
    start = max(0, int(round(offset * sample_rate)))
    stop = min(frames, int(round((offset + duration) * sample_rate)))
    if stop <= start:
        return -math.inf
    data, _ = sf.read(path, start=start, stop=stop, always_2d=True, dtype="float32")
    if data.size == 0:
        return -math.inf
    mean_square = float(np.mean(np.square(data)))
    if mean_square <= 0.0:
        return -math.inf
    return 10.0 * math.log10(mean_square)


def check_record(
    record: dict[str, Any],
    min_duration: float,
    min_rms_db: float,
    edge_tolerance_sec: float,
    skip_rms: bool,
) -> tuple[bool, str, float | None]:
    path = str(record["audio_filepath"])
    offset = float(record.get("offset", 0.0) or 0.0)
    duration = float(record["duration"])
    if duration < min_duration:
        return False, "short_duration", None
    try:
        frames, sample_rate, _channels = audio_info(path)
    except Exception:
        return False, "audio_info_error", None
    audio_duration = frames / float(sample_rate)
    if offset < -edge_tolerance_sec or offset + duration > audio_duration + edge_tolerance_sec:
        return False, "offset_out_of_range", None
    if skip_rms:
        return True, "kept", None
    try:
        rms_db = rms_db_for_segment(path, offset, duration, frames, sample_rate)
    except Exception:
        return False, "rms_read_error", None
    if not math.isfinite(rms_db) or rms_db < min_rms_db:
        return False, "low_rms", rms_db
    return True, "kept", rms_db


def process_chunk(
    lines: list[str],
    min_duration: float,
    min_rms_db: float,
    edge_tolerance_sec: float,
    skip_rms: bool,
) -> tuple[list[str], dict[str, int], dict[str, int], dict[str, int], float | None]:
    kept_lines: list[str] = []
    reason_counts: dict[str, int] = {}
    label_removed: dict[str, int] = {}
    label_kept: dict[str, int] = {}
    min_seen_rms: float | None = None
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            reason_counts["json_error"] = reason_counts.get("json_error", 0) + 1
            continue
        label = str(record.get("label", ""))
        keep, reason, rms_db = check_record(
            record,
            min_duration=min_duration,
            min_rms_db=min_rms_db,
            edge_tolerance_sec=edge_tolerance_sec,
            skip_rms=skip_rms,
        )
        if rms_db is not None and math.isfinite(rms_db):
            min_seen_rms = rms_db if min_seen_rms is None else min(min_seen_rms, rms_db)
        if keep:
            kept_lines.append(json.dumps(record, ensure_ascii=False) + "\n")
            label_kept[label] = label_kept.get(label, 0) + 1
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            label_removed[label] = label_removed.get(label, 0) + 1
    return kept_lines, reason_counts, label_removed, label_kept, min_seen_rms


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def chunked_lines(path: Path, chunk_size: int):
    with path.open(encoding="utf-8") as f:
        chunk: list[str] = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def clean_manifest(manifest: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_path = out_dir / manifest.name
    total = 0
    kept = 0
    reason_counts: dict[str, int] = {}
    label_removed: dict[str, int] = {}
    label_kept: dict[str, int] = {}
    min_seen_rms: float | None = None

    def submit(pool: ProcessPoolExecutor, chunk: list[str]):
        return pool.submit(
            process_chunk,
            chunk,
            args.min_duration,
            args.min_rms_db,
            args.edge_tolerance_sec,
            args.skip_rms,
        )

    with out_path.open("w", encoding="utf-8") as out_f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        chunk_iter = iter(chunked_lines(manifest, args.chunk_size))
        max_pending = max(1, args.workers * 3)
        submitted = 0
        completed = 0

        for _ in range(max_pending):
            try:
                pending.add(submit(pool, next(chunk_iter)))
                submitted += 1
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed += 1
                try:
                    pending.add(submit(pool, next(chunk_iter)))
                    submitted += 1
                except StopIteration:
                    pass
                kept_lines, chunk_reasons, chunk_label_removed, chunk_label_kept, chunk_min_rms = future.result()
                for kept_line in kept_lines:
                    out_f.write(kept_line)
                kept += len(kept_lines)
                total += sum(chunk_reasons.values()) + len(kept_lines)
                merge_counts(reason_counts, chunk_reasons)
                merge_counts(label_removed, chunk_label_removed)
                merge_counts(label_kept, chunk_label_kept)
                if chunk_min_rms is not None:
                    min_seen_rms = chunk_min_rms if min_seen_rms is None else min(min_seen_rms, chunk_min_rms)
                if completed % 100 == 0:
                    print(
                        f"{manifest.name}: processed_chunks={completed}/{submitted} total={total} kept={kept}",
                        flush=True,
                    )

    removed = total - kept
    print(f"wrote {out_path}: total={total} kept={kept} removed={removed}", flush=True)
    return {
        "manifest": str(manifest),
        "out_manifest": str(out_path),
        "total": total,
        "kept": kept,
        "removed": removed,
        "min_seen_rms_db": "" if min_seen_rms is None else f"{min_seen_rms:.2f}",
        "reasons": reason_counts,
        "label_removed": label_removed,
        "label_kept": label_kept,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    reason_keys = sorted({key for row in rows for key in row["reasons"]})
    label_removed_keys = sorted({key for row in rows for key in row["label_removed"]})
    fieldnames = [
        "manifest",
        "out_manifest",
        "total",
        "kept",
        "removed",
        "min_seen_rms_db",
        *[f"reason_{key}" for key in reason_keys],
        *[f"removed_label_{key}" for key in label_removed_keys],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "manifest": row["manifest"],
                "out_manifest": row["out_manifest"],
                "total": row["total"],
                "kept": row["kept"],
                "removed": row["removed"],
                "min_seen_rms_db": row["min_seen_rms_db"],
            }
            for key in reason_keys:
                flat[f"reason_{key}"] = row["reasons"].get(key, 0)
            for key in label_removed_keys:
                flat[f"removed_label_{key}"] = row["label_removed"].get(key, 0)
            writer.writerow(flat)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [clean_manifest(manifest, args.out_dir, args) for manifest in args.manifests]
    summary_path = args.out_dir / args.summary_name
    write_summary(summary_path, rows)
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
