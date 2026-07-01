#!/usr/bin/env python3
"""
Run CommonAccent accent classification on MP4/WAV files.

Segments audio into 3s chunks and prints a per-segment accent timeline.
Model is downloaded from HuggingFace on first run.

Usage:
    python scripts/infer_commonaccent.py --files data/english.wav data/singlish.mp4
    python scripts/infer_commonaccent.py --model Jzuluaga/accent-id-commonaccent_ecapa \
        --savedir checkpoints/commonaccent_ecapa --files audio.wav
"""
import argparse
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SEGMENT_SEC = 3.0
SAMPLE_RATE = 16000

_COMMONACCENT_TO_CODE: dict[str, str] = {
    "african":        "en_af",
    "australia":      "en_au",
    "bermuda":        "en_bm",
    "canada":         "en_ca",
    "england":        "en_gb",
    "hongkong":       "en_hk",
    "indian":         "en_in",
    "ireland":        "en_ie",
    "malaysia":       "en_my",
    "newzealand":     "en_nz",
    "philippines":    "en_ph",
    "scotland":       "en_sc",
    "singapore":      "en_sg",
    "southatlandtic": "en_sa",
    "us":             "en_us",
    "wales":          "en_wl",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Jzuluaga/accent-id-commonaccent_ecapa",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--savedir",
        type=Path,
        default=Path("checkpoints/commonaccent_ecapa"),
        help="Directory to cache the downloaded model.",
    )
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--segment_sec", type=float, default=SEGMENT_SEC)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1, "expected mono wav"
        assert f.getframerate() == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz"
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _get_label_list(model) -> list[str]:
    """Return the model's label list sorted by index."""
    lab2ind: dict[str, int] = model.hparams.label_encoder.lab2ind
    return [lab for lab, _ in sorted(lab2ind.items(), key=lambda x: x[1])]


def infer_file(model, samples: np.ndarray, segment_sec: float, top_k: int) -> list[dict]:
    seg_len = int(segment_sec * SAMPLE_RATE)
    raw_labels = _get_label_list(model)  # e.g. ["african", "australia", ...]
    results = []

    for start in range(0, len(samples) - seg_len // 2, seg_len):
        chunk = samples[start: start + seg_len]
        if len(chunk) < SAMPLE_RATE:
            continue
        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))

        wavs = torch.tensor(chunk).unsqueeze(0)  # [1, T]
        wav_lens = torch.tensor([1.0])

        out_prob, _, _, text_lab = model.classify_batch(wavs, wav_lens)

        # out_prob may be logits or log-softmax; softmax normalises either way
        probs = torch.softmax(out_prob[0].cpu(), dim=-1).numpy()
        top_indices = np.argsort(probs)[::-1][:top_k]

        raw_label = text_lab[0].strip().lower()
        accent_code = _COMMONACCENT_TO_CODE.get(raw_label, f"en_{raw_label}")
        prob = float(probs[top_indices[0]])

        results.append({
            "start": round(start / SAMPLE_RATE, 1),
            "end": round((start + seg_len) / SAMPLE_RATE, 1),
            "pred": accent_code,
            "prob": prob,
            "top_k": [
                (_COMMONACCENT_TO_CODE.get(raw_labels[i], f"en_{raw_labels[i]}"), float(probs[i]))
                for i in top_indices
            ],
        })

    return results


def print_results(results: list[dict], top_k: int) -> None:
    alt_header = "  ".join(f"{'#'+str(i+1):<8} {'prob':>6}" for i in range(1, top_k))
    print(f"  {'Start':>6}  {'End':>6}  {'Pred':<8} {'Conf':>6}  {alt_header}")
    print(f"  {'-' * (36 + 18 * (top_k - 1))}")
    for r in results:
        alts = "  ".join(f"{code:<8} {p:>6.3f}" for code, p in r["top_k"][1:])
        print(f"  {r['start']:>6.1f}  {r['end']:>6.1f}  {r['pred']:<8} {r['prob']:>6.3f}  {alts}")


def summarize(results: list[dict]) -> None:
    counts = Counter(r["pred"] for r in results)
    total = len(results)
    print(f"\n  Accent distribution ({total} segments):")
    for code, count in counts.most_common():
        bar = "█" * int(count / total * 30)
        print(f"  {code:<8} {count:>4} ({100 * count / total:5.1f}%)  {bar}")


def main():
    args = parse_args()

    from speechbrain.inference.classifiers import EncoderClassifier

    print(f"loading model: {args.model}")
    model = EncoderClassifier.from_hparams(
        source=str(args.model),
        savedir=str(args.savedir),
        run_opts={"device": args.device},
    )
    print(f"model loaded  (device={args.device})\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        for src in args.files:
            print(f"{'=' * 64}")
            print(f"file: {src.name}")

            wav_path = Path(tmpdir) / (src.stem + ".wav")
            if src.suffix.lower() != ".wav":
                print("  converting to wav...")
                to_wav(src, wav_path)
            else:
                shutil.copy(src, wav_path)

            samples = load_wav(wav_path)
            results = infer_file(model, samples, args.segment_sec, args.top_k)
            print()
            print_results(results, args.top_k)
            summarize(results)
            print()


if __name__ == "__main__":
    main()
