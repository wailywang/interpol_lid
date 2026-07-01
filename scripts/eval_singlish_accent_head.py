#!/usr/bin/env python3
"""Evaluate the binary en/en_sg Singlish Accent Head.

Runs two evaluations:
  1. Binary accuracy on the accent eval manifest (en + en_sg entries).
  2. Threshold sweep on val manifest to find optimal P(en_sg) cutoff.
"""
import argparse
import json
from pathlib import Path

import torch


DEFAULT_LABELS = ["en", "en_sg"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path("checkpoints/singlish_accent_head.nemo"))
    parser.add_argument("--eval_manifest", type=Path,
                        default=Path("manifests/singlish_accent/eval.json"))
    parser.add_argument("--val_manifest", type=Path,
                        default=Path("manifests/singlish_accent/val.json"))
    parser.add_argument("--work_dir", type=Path,
                        default=Path("experiments/singlish_accent_head"))
    parser.add_argument("--log_file", type=Path, default=None)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", default="1")
    parser.add_argument("--accelerator", default="auto",
                        choices=("auto", "gpu", "cpu"))
    parser.add_argument("--precision", default="32")
    parser.add_argument(
        "--threshold_sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sweep P(en_sg) threshold on val manifest to find best F1.",
    )
    return parser.parse_args()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def setup_logging(args):
    import sys
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.work_dir / "eval_singlish.log")
    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print(f"log: {log_path}")
    return log_fh


def load_model(ckpt: Path):
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    model = EncDecSpeakerLabelModel.restore_from(restore_path=str(ckpt))
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def get_labels(model) -> list[str]:
    for attr in ("cfg.labels", "cfg.train_ds.labels",
                 "decoder.labels", "_cfg.labels"):
        parts = attr.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            return sorted(list(obj))
    raise ValueError("Cannot determine model labels")


def run_ambernet(model, wav: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    sig = wav.unsqueeze(0).to(device)
    length = torch.tensor([wav.numel()], device=device)
    with torch.inference_mode():
        out = model(input_signal=sig, input_signal_length=length)
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item[0].detach().cpu()
    if isinstance(out, torch.Tensor):
        return out[0].detach().cpu()
    raise RuntimeError("Cannot extract logits from model output")


def collect_probs(model, manifest: Path, labels: list[str], sr: int = 16000):
    import soundfile as sf

    label_to_idx = {l: i for i, l in enumerate(labels)}
    all_p_ensg = []
    all_true = []

    with manifest.open() as f:
        entries = [json.loads(l) for l in f if l.strip()]

    for entry in entries:
        path = entry["audio_filepath"]
        true_label = entry.get("label", "")
        try:
            wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue

        wav_t = torch.from_numpy(wav)
        if file_sr != sr:
            wav_t = torch.nn.functional.interpolate(
                wav_t.unsqueeze(0).unsqueeze(0), scale_factor=sr / file_sr, mode="linear",
                align_corners=False,
            ).squeeze()
        if wav_t.numel() < sr:
            wav_t = torch.nn.functional.pad(wav_t, (0, sr - wav_t.numel()))

        logits = run_ambernet(model, wav_t)
        probs = torch.softmax(logits, dim=-1)
        p_ensg = float(probs[label_to_idx["en_sg"]]) if "en_sg" in label_to_idx else 0.0

        all_p_ensg.append(p_ensg)
        all_true.append(true_label)

    return all_p_ensg, all_true


def binary_metrics(p_ensg: list[float], true_labels: list[str], threshold: float):
    tp = fp = tn = fn = 0
    for p, t in zip(p_ensg, true_labels):
        pred = "en_sg" if p > threshold else "en"
        if pred == "en_sg" and t == "en_sg":
            tp += 1
        elif pred == "en_sg" and t != "en_sg":
            fp += 1
        elif pred == "en" and t != "en_sg":
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(true_labels) if true_labels else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def threshold_sweep(p_ensg: list[float], true_labels: list[str]):
    best = {"f1": -1.0, "threshold": 0.5}
    thresholds = [i / 20 for i in range(2, 19)]  # 0.10 … 0.90
    print("\nThreshold sweep:")
    print(f"{'θ':>6}  {'acc':>6}  {'prec':>6}  {'rec':>6}  {'f1':>6}")
    for θ in thresholds:
        m = binary_metrics(p_ensg, true_labels, θ)
        marker = " ←" if m["f1"] > best["f1"] else ""
        print(f"{θ:6.2f}  {m['accuracy']:6.4f}  {m['precision']:6.4f}  "
              f"{m['recall']:6.4f}  {m['f1']:6.4f}{marker}")
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "threshold": θ, **m}
    print(f"\nBest threshold: {best['threshold']:.2f}  F1={best['f1']:.4f}")
    return best


def main():
    args = parse_args()
    log_fh = setup_logging(args)
    labels = sorted([l.strip() for l in args.labels.split(",") if l.strip()])
    print(f"labels: {labels}")
    print(f"ckpt:   {args.ckpt}")

    model = load_model(args.ckpt)
    model_labels = get_labels(model)
    print(f"model labels: {model_labels}")

    # ── eval manifest: binary accuracy at default threshold 0.5 ───────────
    print(f"\n[eval] {args.eval_manifest}")
    p_ensg_eval, true_eval = collect_probs(model, args.eval_manifest, model_labels)
    m = binary_metrics(p_ensg_eval, true_eval, threshold=0.5)
    print(f"Eval (θ=0.5):  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    by_lang = {}
    for p, t in zip(p_ensg_eval, true_eval):
        pred = "en_sg" if p > 0.5 else "en"
        by_lang.setdefault(t, {"correct": 0, "total": 0})
        by_lang[t]["total"] += 1
        if pred == t:
            by_lang[t]["correct"] += 1
    print("Per-class accuracy:")
    for lang, counts in sorted(by_lang.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {lang:8s}: {counts['correct']:5d}/{counts['total']:5d}  ({acc:.4f})")

    best_threshold = 0.5
    if args.threshold_sweep and args.val_manifest.exists():
        print(f"\n[val] {args.val_manifest}")
        p_ensg_val, true_val = collect_probs(model, args.val_manifest, model_labels)
        best = threshold_sweep(p_ensg_val, true_val)
        best_threshold = best["threshold"]
        print(f"\n[eval re-run at best θ={best_threshold}]")
        m2 = binary_metrics(p_ensg_eval, true_eval, threshold=best_threshold)
        print(f"Eval (θ={best_threshold:.2f}): acc={m2['accuracy']:.4f}  "
              f"prec={m2['precision']:.4f}  rec={m2['recall']:.4f}  f1={m2['f1']:.4f}")

    # save metrics
    out = {
        "eval_manifest": str(args.eval_manifest),
        "metrics_at_0.5": m,
        "best_threshold": best_threshold,
    }
    metrics_path = args.work_dir / "singlish_eval_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    print(f"\nMetrics saved: {metrics_path}")
    print(f"\nTo use this model in inference, set:\n"
          f"  export SINGLISH_HEAD_CKPT={args.ckpt}\n"
          f"  export SINGLISH_EN_SG_THRESHOLD={best_threshold:.2f}")

    log_fh.close()


if __name__ == "__main__":
    main()
