#!/usr/bin/env python3
"""
LID inference on MP4/WAV demo files using a finetuned AmberNet .nemo checkpoint.

Segments audio into 3s chunks, runs the 19-class LID model, and prints a
per-segment timeline with the top prediction and confidence.

Usage:
    python scripts/infer_lid.py \
        --model checkpoints/ambernet_lid_19class_oversample_20epoch.nemo \
        --files /dataset/yw500/data/demodata/english.mp4 \
                /dataset/yw500/data/demodata/singlish.mp4 \
                /dataset/yw500/data/demodata/singlish_speakers.mp4
"""
import argparse
import subprocess
import tempfile
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SEGMENT_SEC = 3.0
SAMPLE_RATE = 16000
LABELS = ["de", "en", "es", "fr", "hi", "id", "ja", "km", "ko",
          "ms", "pt", "ru", "th", "tl", "tr", "unknown", "vi", "yue", "zh"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("checkpoints/ambernet_lid_19class_oversample_20epoch.nemo"))
    parser.add_argument("--files", nargs="+", type=Path, default=[
        Path("/dataset/yw500/data/demodata/english.mp4"),
        Path("/dataset/yw500/data/demodata/singlish.mp4"),
        Path("/dataset/yw500/data/demodata/singlish_speakers.mp4"),
    ])
    parser.add_argument("--segment_sec", type=float, default=SEGMENT_SEC)
    parser.add_argument("--top_k", type=int, default=2, help="show top-k predictions per segment")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", str(dst)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == SAMPLE_RATE
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def infer_file(model, audio_path: Path, segment_sec: float, device: str, top_k: int) -> list[dict]:
    samples = load_wav(audio_path)
    seg_len = int(segment_sec * SAMPLE_RATE)
    results = []

    for start in range(0, len(samples) - seg_len // 2, seg_len):
        chunk = samples[start: start + seg_len]
        if len(chunk) < SAMPLE_RATE:
            continue
        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))

        signal = torch.tensor(chunk).unsqueeze(0).to(device)
        length = torch.tensor([len(chunk)]).to(device)

        with torch.no_grad():
            out = model.forward(input_signal=signal, input_signal_length=length)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        results.append({
            "start": round(start / SAMPLE_RATE, 1),
            "end": round((start + seg_len) / SAMPLE_RATE, 1),
            "pred": LABELS[top_indices[0]],
            "pred_prob": float(probs[top_indices[0]]),
            "top_k": [(LABELS[i], float(probs[i])) for i in top_indices],
        })
    return results


def print_results(results: list[dict], top_k: int) -> None:
    alt_header = "  ".join(f"{'#'+str(i+1)+'_lang':<8} {'prob':>6}" for i in range(1, top_k))
    print(f"  {'Start':>6}  {'End':>6}  {'Pred':<10} {'Conf':>6}  {alt_header}")
    print(f"  {'-' * (36 + 18 * (top_k - 1))}")
    for r in results:
        alts = "  ".join(f"{lang:<8} {prob:>6.3f}" for lang, prob in r["top_k"][1:])
        print(f"  {r['start']:>6.1f}  {r['end']:>6.1f}  {r['pred']:<10} {r['pred_prob']:>6.3f}  {alts}")


def summarize(results: list[dict]) -> None:
    counts = Counter(r["pred"] for r in results)
    total = len(results)
    print(f"\n  Language distribution ({total} segments):")
    for lang, count in counts.most_common():
        bar = "█" * int(count / total * 30)
        print(f"  {lang:<10} {count:>4} ({100*count/total:5.1f}%)  {bar}")


def main():
    args = parse_args()

    print(f"loading model: {args.model}")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    model = EncDecSpeakerLabelModel.restore_from(str(args.model))
    model.eval().to(args.device)
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

            results = infer_file(model, wav_path, args.segment_sec, args.device, args.top_k)
            print()
            print_results(results, args.top_k)
            summarize(results)
            print()


if __name__ == "__main__":
    main()
