#!/usr/bin/env python3
"""
Quick inference script: run the Singlish accent head on MP4/WAV demo files.

For each file, segments audio into 3s chunks, runs the accent model,
and prints a timeline of en vs en_sg predictions.

Usage:
    python scripts/infer_singlish_accent.py \
        --model checkpoints/singlish_accent_head.nemo \
        --files data/demodata/singlish.mp4 data/demodata/singlish_speakers.mp4
"""
import argparse
import subprocess
import tempfile
import os
from pathlib import Path

import numpy as np
import torch


SEGMENT_SEC = 3.0
SAMPLE_RATE = 16000
LABELS = ["en", "en_sg"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/singlish_accent_head.nemo"),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            Path("/dataset/yw500/data/demodata/singlish.mp4"),
            Path("/dataset/yw500/data/demodata/singlish_speakers.mp4"),
        ],
    )
    parser.add_argument("--segment_sec", type=float, default=SEGMENT_SEC)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser.parse_args()


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "wav", str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_wav(path: Path) -> np.ndarray:
    import wave, struct
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == SAMPLE_RATE
        raw = f.readframes(f.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def infer_file(model, audio_path: Path, segment_sec: float, device: str) -> list[dict]:
    samples = load_wav(audio_path)
    seg_len = int(segment_sec * SAMPLE_RATE)
    results = []

    for start in range(0, len(samples) - seg_len // 2, seg_len):
        chunk = samples[start : start + seg_len]
        if len(chunk) < SAMPLE_RATE:  # skip < 1s
            continue

        # Pad short final segment
        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))

        signal = torch.tensor(chunk).unsqueeze(0).to(device)
        length = torch.tensor([len(chunk)]).to(device)

        with torch.no_grad():
            logits = model.forward(
                input_signal=signal,
                input_signal_length=length,
            )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        results.append({
            "start": round(start / SAMPLE_RATE, 1),
            "end": round((start + len(chunk)) / SAMPLE_RATE, 1),
            "pred": LABELS[pred_idx],
            "en_prob": float(probs[LABELS.index("en")]),
            "en_sg_prob": float(probs[LABELS.index("en_sg")]),
        })

    return results


def summarize(results: list[dict]) -> None:
    counts = {"en": 0, "en_sg": 0}
    for r in results:
        counts[r["pred"]] += 1
    total = len(results)
    print(f"\n  Summary: {counts['en_sg']} / {total} segments predicted en_sg "
          f"({100*counts['en_sg']/total:.1f}%)")
    avg_en_sg = np.mean([r["en_sg_prob"] for r in results])
    print(f"  Mean en_sg confidence: {avg_en_sg:.3f}")


def main():
    args = parse_args()

    print(f"loading model: {args.model}")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    model = EncDecSpeakerLabelModel.restore_from(str(args.model))
    model.eval()
    model = model.to(args.device)
    print("model loaded\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        for src in args.files:
            print(f"{'='*60}")
            print(f"file: {src.name}")

            wav_path = Path(tmpdir) / (src.stem + ".wav")
            if src.suffix.lower() != ".wav":
                print("  converting to wav...")
                to_wav(src, wav_path)
            else:
                import shutil
                shutil.copy(src, wav_path)

            results = infer_file(model, wav_path, args.segment_sec, args.device)

            print(f"\n  {'Start':>6}  {'End':>6}  {'Pred':<8}  {'en':>6}  {'en_sg':>6}")
            print(f"  {'-'*44}")
            for r in results:
                marker = " ◄" if r["pred"] == "en_sg" else ""
                print(f"  {r['start']:>6.1f}  {r['end']:>6.1f}  {r['pred']:<8}  "
                      f"{r['en_prob']:>6.3f}  {r['en_sg_prob']:>6.3f}{marker}")

            summarize(results)
            print()


if __name__ == "__main__":
    main()
