#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

import torch


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


class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self) -> int:
        for stream in self.streams:
            if hasattr(stream, "fileno"):
                return stream.fileno()
        raise OSError("No underlying stream exposes fileno()")

    def __getattr__(self, name: str):
        return getattr(self.streams[0], name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a finetuned AmberNet LID .nemo checkpoint.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/checkpoints/ambernet_lid_18lang_yue.nemo"),
    )
    parser.add_argument(
        "--eval_manifest",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/manifests/lid_eval_3s.json"),
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/experiments/eval_ambernet_lid_18lang_yue"),
    )
    parser.add_argument("--log_file", type=Path, default=None)
    parser.add_argument("--metrics_file", type=Path, default=None)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", default="1")
    parser.add_argument("--accelerator", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--precision", default="32")
    parser.add_argument(
        "--per_label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute per-language accuracy and confusion matrix after trainer.test.",
    )
    parser.add_argument("--top_confusions", type=int, default=30)
    return parser.parse_args()


def setup_text_logging(args: argparse.Namespace):
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.work_dir / "eval.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)
    print(f"text log file: {log_path}")
    return log_fh


def parse_labels(labels: str) -> list[str]:
    parsed = [label.strip() for label in labels.split(",") if label.strip()]
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"Duplicate labels in --labels: {parsed}")
    if not parsed:
        raise ValueError("--labels cannot be empty")
    canonical = sorted(parsed)
    if canonical != parsed:
        print(f"Canonical label order: {','.join(canonical)}")
    return canonical


def build_trainer(args: argparse.Namespace):
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl

    kwargs = {
        "default_root_dir": str(args.work_dir),
        "logger": False,
        "enable_checkpointing": False,
    }
    sig = inspect.signature(pl.Trainer)
    if "precision" in sig.parameters:
        kwargs["precision"] = args.precision

    if args.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.accelerator

    if "accelerator" in sig.parameters:
        kwargs["accelerator"] = accelerator
        kwargs["devices"] = args.devices if accelerator == "gpu" else 1
    else:
        kwargs["gpus"] = args.devices if accelerator == "gpu" else 0

    return pl.Trainer(**kwargs)


def extract_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(output, "logits"):
        return output.logits
    raise ValueError(f"Unable to extract logits from model output type {type(output)}")


def compute_per_label_metrics(model, labels: list[str], work_dir: Path, top_k: int) -> dict:
    if not hasattr(model, "_test_dl") or model._test_dl is None:
        raise RuntimeError("model._test_dl is not initialized; call model.setup_test_data first")

    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    label_count = len(labels)
    confusion = torch.zeros(label_count, label_count, dtype=torch.long)

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(model._test_dl, start=1):
            audio_signal, audio_signal_len, targets, _ = batch
            audio_signal = audio_signal.to(device)
            audio_signal_len = audio_signal_len.to(device)
            targets = targets.to(device)
            logits = extract_logits(model(input_signal=audio_signal, input_signal_length=audio_signal_len))
            preds = torch.argmax(logits, dim=-1)
            for gold, pred in zip(targets.detach().cpu().tolist(), preds.detach().cpu().tolist()):
                if 0 <= gold < label_count and 0 <= pred < label_count:
                    confusion[gold, pred] += 1
            if batch_idx % 100 == 0:
                print(f"per-label eval batches: {batch_idx}")

    per_label = []
    correct_total = int(confusion.diag().sum().item())
    count_total = int(confusion.sum().item())
    for idx, label in enumerate(labels):
        support = int(confusion[idx].sum().item())
        correct = int(confusion[idx, idx].item())
        acc = correct / support if support else 0.0
        per_label.append(
            {
                "label": label,
                "support": support,
                "correct": correct,
                "accuracy": acc,
            }
        )

    confusions = []
    for gold_idx, gold_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            if gold_idx == pred_idx:
                continue
            count = int(confusion[gold_idx, pred_idx].item())
            if count:
                gold_support = int(confusion[gold_idx].sum().item())
                confusions.append(
                    {
                        "gold": gold_label,
                        "pred": pred_label,
                        "count": count,
                        "gold_support": gold_support,
                        "rate_within_gold": count / gold_support if gold_support else 0.0,
                    }
                )
    confusions.sort(key=lambda item: item["count"], reverse=True)

    summary = {
        "overall_accuracy": correct_total / count_total if count_total else 0.0,
        "total": count_total,
        "correct": correct_total,
        "macro_accuracy": sum(item["accuracy"] for item in per_label) / len(per_label),
        "per_label": per_label,
        "top_confusions": confusions[:top_k],
    }

    per_label_path = work_dir / "eval_per_language.json"
    with per_label_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    confusion_path = work_dir / "confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *[int(v) for v in confusion[idx].tolist()]])

    print("\nPer-language accuracy")
    print(f"{'label':<5} {'support':>8} {'correct':>8} {'acc':>8}")
    for item in per_label:
        print(f"{item['label']:<5} {item['support']:>8} {item['correct']:>8} {item['accuracy']:>8.4f}")

    print(f"\noverall_accuracy: {summary['overall_accuracy']:.4f}")
    print(f"macro_accuracy: {summary['macro_accuracy']:.4f}")
    print("\nTop confusions")
    print(f"{'gold':<5} {'pred':<5} {'count':>8} {'rate':>8}")
    for item in confusions[:top_k]:
        print(f"{item['gold']:<5} {item['pred']:<5} {item['count']:>8} {item['rate_within_gold']:>8.4f}")

    print(f"\nwrote per-language metrics: {per_label_path}")
    print(f"wrote confusion matrix: {confusion_path}")
    return summary


def main() -> None:
    args = parse_args()
    log_fh = setup_text_logging(args)

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    if not args.eval_manifest.is_file():
        raise FileNotFoundError(f"Missing eval manifest: {args.eval_manifest}")

    labels = parse_labels(args.labels)
    print(f"checkpoint: {args.ckpt}")
    print(f"eval_manifest: {args.eval_manifest}")
    print(f"labels: {','.join(labels)}")

    from omegaconf import open_dict
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    trainer = build_trainer(args)
    model = EncDecSpeakerLabelModel.restore_from(restore_path=str(args.ckpt), trainer=trainer)

    cfg = model.cfg
    with open_dict(cfg):
        cfg.test_ds.manifest_filepath = str(args.eval_manifest)
        cfg.test_ds.sample_rate = args.sample_rate
        cfg.test_ds.labels = labels
        cfg.test_ds.batch_size = args.batch_size
        cfg.test_ds.shuffle = False
        cfg.test_ds.num_workers = args.num_workers
        cfg.test_ds.pin_memory = torch.cuda.is_available()

    model.setup_test_data(cfg.test_ds)
    results = trainer.test(model)
    print(f"eval results: {results}")

    per_label_results = None
    if args.per_label:
        per_label_results = compute_per_label_metrics(
            model=model,
            labels=labels,
            work_dir=args.work_dir,
            top_k=args.top_confusions,
        )

    metrics_path = args.metrics_file or (args.work_dir / "eval_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"trainer_test": results, "per_label": per_label_results}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote metrics: {metrics_path}")

    log_fh.close()


if __name__ == "__main__":
    main()
