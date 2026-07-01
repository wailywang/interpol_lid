#!/usr/bin/env python3
"""Build balanced en / en_sg manifests for Singlish Accent Head training.

Outputs (JSONL, one JSON object per line):
  manifests/singlish_accent/train.json
  manifests/singlish_accent/val.json
  manifests/singlish_accent/eval.json

en entries are extracted from the existing mixed LID manifests.
en_sg entries come from the NSC Singlish manifests.
"""
import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lid_train", type=Path,
                        default=Path("manifests/lid_train_en_cv_mix_3s.json"))
    parser.add_argument("--lid_val", type=Path,
                        default=Path("manifests/lid_val_en_cv_mix_3s.json"))
    parser.add_argument("--lid_eval", type=Path,
                        default=Path("manifests/lid_eval_en_cv_mix_3s.json"))
    parser.add_argument("--nsc_train", type=Path,
                        default=Path("manifests/nsc_singlish_train.json"))
    parser.add_argument("--nsc_val", type=Path,
                        default=Path("manifests/nsc_singlish_validation.json"))
    parser.add_argument("--nsc_eval", type=Path,
                        default=Path("manifests/nsc_singlish_test.json"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("manifests/singlish_accent"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_label(path: Path, label: str) -> list[dict]:
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("label") == label:
                entries.append(obj)
    return entries


def load_all(path: Path) -> list[dict]:
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def write_manifest(entries: list[dict], path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for obj in entries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    dur = sum(e.get("duration", 0) for e in entries)
    counts = {}
    for e in entries:
        lbl = e.get("label", "?")
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"{description}: {len(entries)} samples, {dur/3600:.1f}h — " +
          ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # ── train ──────────────────────────────────────────────────────────────
    en_train = load_label(args.lid_train, "en")
    ensg_train = load_all(args.nsc_train)
    rng.shuffle(en_train)
    rng.shuffle(ensg_train)
    train = en_train + ensg_train
    rng.shuffle(train)
    write_manifest(train, args.out_dir / "train.json", "train")

    # ── val ────────────────────────────────────────────────────────────────
    en_val = load_label(args.lid_val, "en")
    ensg_val = load_all(args.nsc_val)
    val = en_val + ensg_val
    rng.shuffle(val)
    write_manifest(val, args.out_dir / "val.json", "val")

    # ── eval ───────────────────────────────────────────────────────────────
    en_eval = load_label(args.lid_eval, "en")
    ensg_eval = load_all(args.nsc_eval)
    evl = en_eval + ensg_eval
    write_manifest(evl, args.out_dir / "eval.json", "eval")

    print("Done. Manifests written to", args.out_dir)


if __name__ == "__main__":
    main()
