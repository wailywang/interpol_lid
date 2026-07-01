#!/usr/bin/env python3
import argparse
import inspect
import json
import sys
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full finetune AmberNet LID on balanced manifests.")
    parser.add_argument(
        "--base_ckpt",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/checkpoints/ambernet.nemo"),
    )
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/manifests/lid_train.json"),
    )
    parser.add_argument(
        "--train_manifest_template",
        default=None,
        help=(
            "Optional per-epoch train manifest template. Supports Python format fields "
            "{epoch} for 1-based epoch and {epoch0} for 0-based epoch, e.g. "
            "manifests/full_rotating_50h_epochs/lid_train_epoch{epoch:02d}_cap50h_3s.json. "
            "When set, Lightning reloads the train dataloader every epoch."
        ),
    )
    parser.add_argument(
        "--val_manifest",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/manifests/lid_val.json"),
    )
    parser.add_argument(
        "--eval_manifest",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/manifests/lid_eval.json"),
    )
    parser.add_argument(
        "--out_ckpt",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/checkpoints/ambernet_lid_18lang_yue.nemo"),
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/experiments/ambernet_lid_18lang_yue"),
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="File to tee stdout/stderr into. Defaults to <work_dir>/train.log.",
    )
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--drop_last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the final incomplete training batch to avoid BatchNorm failures with per-rank batch size 1.",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=2e-6)
    parser.add_argument(
        "--loss_weight",
        choices=("none", "auto"),
        default="none",
        help="Use unweighted CE for balanced manifests, or NeMo auto class weights.",
    )
    parser.add_argument("--devices", default="1", help="Lightning devices value, e.g. 1 or 0,1.")
    parser.add_argument("--accelerator", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--precision", default="32", help="Lightning precision, e.g. 32, 16-mixed.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true", help="Log training and validation metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", default="ots-lid", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", default=None, help="Optional Weights & Biases entity/team.")
    parser.add_argument("--wandb_name", default=None, help="Optional Weights & Biases run name. Defaults to work_dir name.")
    parser.add_argument("--wandb_tags", default="", help="Comma-separated Weights & Biases tags.")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases mode.",
    )
    parser.add_argument(
        "--keep_augmentor",
        action="store_true",
        help="Keep augmentor config from the base checkpoint. Default disables it because paths are checkpoint-local.",
    )
    parser.add_argument(
        "--enable_ambernet_augmentor",
        action="store_true",
        help="Enable AmberNet-style online augmentation: MUSAN noise, RIRS impulse, and speed perturbation.",
    )
    parser.add_argument(
        "--musan_dir",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/data/augmentation/musan"),
        help="Directory containing MUSAN wav files for noise augmentation.",
    )
    parser.add_argument(
        "--rirs_dir",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/data/augmentation/RIRS_NOISES"),
        help="Directory containing RIRS_NOISES wav files for impulse/RIR augmentation.",
    )
    parser.add_argument(
        "--augment_manifest_dir",
        type=Path,
        default=Path("/export/home2/wa0009xi/ots-lid/manifests/augmentation"),
        help="Directory where generated augmentation manifests are stored.",
    )
    parser.add_argument(
        "--rebuild_augment_manifests",
        action="store_true",
        help="Re-scan MUSAN/RIRS wav files even if generated augmentation manifests already exist.",
    )
    parser.add_argument(
        "--include_real_rirs",
        action="store_true",
        help="Also include mono real_rirs_isotropic_noises files in the RIR manifest. Default uses mono simulated RIRs only.",
    )
    parser.add_argument("--noise_prob", type=float, default=0.8)
    parser.add_argument("--noise_min_snr_db", type=float, default=0.0)
    parser.add_argument("--noise_max_snr_db", type=float, default=15.0)
    parser.add_argument("--rir_prob", type=float, default=0.5)
    parser.add_argument("--speed_prob", type=float, default=0.5)
    parser.add_argument("--min_speed_rate", type=float, default=0.95)
    parser.add_argument("--max_speed_rate", type=float, default=1.05)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help=(
            "Stop training if val_loss does not improve for this many epochs. "
            "0 disables early stopping (default)."
        ),
    )
    parser.add_argument(
        "--resume_ckpt",
        type=Path,
        default=None,
        help="Lightning checkpoint to resume training from (e.g. last.ckpt).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default=None,
        help="Override directory for Lightning checkpoints. Defaults to <work_dir>/lightning_checkpoints.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip running eval manifest after training.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze the AmberNet encoder and train only the decoder/classification head.",
    )
    return parser.parse_args()


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


def setup_text_logging(args: argparse.Namespace):
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.work_dir / "train.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)
    print(f"text log file: {log_path}")
    return log_fh


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


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


def resolve_epoch_manifest(template: str, epoch0: int) -> Path:
    epoch = epoch0 + 1
    return Path(template.format(epoch=epoch, epoch0=epoch0))


def build_trainer(args: argparse.Namespace):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger

    args.work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else args.work_dir / "lightning_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="ambernet-lid-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [checkpoint_callback]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )

    logger = True
    if args.wandb:
        wandb_mode = "disabled" if args.wandb_mode == "disabled" else args.wandb_mode
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or args.work_dir.name,
            save_dir=str(args.work_dir),
            tags=tags,
            log_model=False,
            mode=wandb_mode,
        )

    kwargs = {
        "max_epochs": args.max_epochs,
        "default_root_dir": str(args.work_dir),
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": args.log_every_n_steps,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": args.gradient_clip_val,
    }

    sig = inspect.signature(pl.Trainer)
    if args.train_manifest_template and "reload_dataloaders_every_n_epochs" in sig.parameters:
        kwargs["reload_dataloaders_every_n_epochs"] = 1
    if "precision" in sig.parameters:
        kwargs["precision"] = args.precision
    if "val_check_interval" in sig.parameters:
        kwargs["val_check_interval"] = args.val_check_interval

    if "accelerator" in sig.parameters:
        if args.accelerator == "auto":
            kwargs["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
        else:
            kwargs["accelerator"] = args.accelerator
        if kwargs["accelerator"] == "gpu":
            kwargs["devices"] = args.devices
        else:
            kwargs["devices"] = 1
    elif "gpus" in sig.parameters:
        if args.accelerator == "cpu" or not torch.cuda.is_available():
            kwargs["gpus"] = 0
        else:
            kwargs["gpus"] = args.devices

    return pl.Trainer(**kwargs)


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate for {path}: {info.samplerate}")
    return float(info.frames) / float(info.samplerate)


def write_audio_manifest(
    paths: list[Path],
    manifest_path: Path,
    description: str,
    required_channels: int | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total_duration = 0.0
    with manifest_path.open("w", encoding="utf-8") as f:
        for path in paths:
            try:
                import soundfile as sf

                info = sf.info(str(path))
                if required_channels is not None and info.channels != required_channels:
                    continue
                if info.samplerate <= 0:
                    raise ValueError(f"Invalid sample rate: {info.samplerate}")
                duration = float(info.frames) / float(info.samplerate)
            except Exception as exc:
                print(f"skip {description} file with unreadable duration: {path} ({exc})")
                continue
            if duration <= 0.0:
                print(f"skip {description} file with non-positive duration: {path}")
                continue
            f.write(
                json.dumps(
                    {
                        "audio_filepath": str(path),
                        "duration": duration,
                        "text": "",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            count += 1
            total_duration += duration
    if count == 0:
        raise RuntimeError(f"No usable {description} wav files were written to {manifest_path}")
    print(f"wrote {description} manifest: {manifest_path} ({count} files, {total_duration / 3600.0:.2f} h)")


def manifest_has_entries(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as f:
        return any(line.strip() for line in f)


def prepare_augmentation_manifests(args: argparse.Namespace) -> tuple[Path, Path]:
    require_dir(args.musan_dir, "MUSAN augmentation directory")
    require_dir(args.rirs_dir, "RIRS augmentation directory")

    noise_manifest = args.augment_manifest_dir / "musan_noise.json"
    rir_manifest = args.augment_manifest_dir / "rirs_impulse.json"

    if (
        not args.rebuild_augment_manifests
        and manifest_has_entries(noise_manifest)
        and manifest_has_entries(rir_manifest)
    ):
        print(f"using existing MUSAN noise manifest: {noise_manifest}")
        print(f"using existing RIRS impulse manifest: {rir_manifest}")
        return noise_manifest, rir_manifest

    musan_wavs = sorted(args.musan_dir.rglob("*.wav"))
    rir_wavs = sorted((args.rirs_dir / "simulated_rirs").rglob("*.wav"))
    if args.include_real_rirs:
        real_rirs = [
            path
            for path in sorted((args.rirs_dir / "real_rirs_isotropic_noises").rglob("*.wav"))
            if "_rir_" in path.name
        ]
        rir_wavs.extend(real_rirs)

    if not musan_wavs:
        raise RuntimeError(f"No MUSAN wav files found under {args.musan_dir}")
    if not rir_wavs:
        raise RuntimeError(f"No RIR wav files found under {args.rirs_dir}")

    write_audio_manifest(musan_wavs, noise_manifest, "MUSAN noise")
    write_audio_manifest(rir_wavs, rir_manifest, "RIRS impulse", required_channels=1)
    return noise_manifest, rir_manifest


def build_ambernet_augmentor_config(args: argparse.Namespace):
    noise_manifest, rir_manifest = prepare_augmentation_manifests(args)
    return {
        "noise": {
            "manifest_path": str(noise_manifest),
            "prob": args.noise_prob,
            "min_snr_db": args.noise_min_snr_db,
            "max_snr_db": args.noise_max_snr_db,
        },
        "impulse": {
            "manifest_path": str(rir_manifest),
            "prob": args.rir_prob,
        },
        "speed": {
            "prob": args.speed_prob,
            "sr": args.sample_rate,
            "resample_type": "kaiser_fast",
            "min_speed_rate": args.min_speed_rate,
            "max_speed_rate": args.max_speed_rate,
        },
    }


def update_cfg_for_finetune(cfg, args: argparse.Namespace, labels: list[str]):
    with open_dict(cfg):
        cfg.train_ds.manifest_filepath = str(args.train_manifest)
        cfg.train_ds.sample_rate = args.sample_rate
        cfg.train_ds.labels = labels
        cfg.train_ds.batch_size = args.batch_size
        cfg.train_ds.shuffle = True
        cfg.train_ds.num_workers = args.num_workers
        cfg.train_ds.pin_memory = torch.cuda.is_available()
        cfg.train_ds.drop_last = args.drop_last
        cfg.train_ds.cal_labels_occurrence = args.loss_weight == "auto"
        if args.enable_ambernet_augmentor:
            cfg.train_ds.augmentor = build_ambernet_augmentor_config(args)
        elif not args.keep_augmentor and "augmentor" in cfg.train_ds:
            cfg.train_ds.augmentor = None

        cfg.validation_ds.manifest_filepath = str(args.val_manifest)
        cfg.validation_ds.sample_rate = args.sample_rate
        cfg.validation_ds.labels = labels
        cfg.validation_ds.batch_size = args.val_batch_size
        cfg.validation_ds.shuffle = False
        cfg.validation_ds.num_workers = args.num_workers
        cfg.validation_ds.pin_memory = torch.cuda.is_available()
        cfg.validation_ds.cal_labels_occurrence = args.loss_weight == "auto"

        if getattr(cfg, "test_ds", None) is not None:
            cfg.test_ds.manifest_filepath = str(args.eval_manifest)
            cfg.test_ds.sample_rate = args.sample_rate
            cfg.test_ds.labels = labels
            cfg.test_ds.batch_size = args.val_batch_size
            cfg.test_ds.shuffle = False
            cfg.test_ds.num_workers = args.num_workers
            cfg.test_ds.pin_memory = torch.cuda.is_available()

        cfg.decoder.num_classes = len(labels)
        cfg.labels = labels

        cfg.loss.weight = "auto" if args.loss_weight == "auto" else None
        cfg.optim.name = "adam"
        cfg.optim.lr = args.lr
        cfg.optim.weight_decay = args.weight_decay
        if getattr(cfg.optim, "sched", None) is not None:
            cfg.optim.sched.min_lr = args.min_lr

    return cfg


def install_rotating_train_manifest_loader(model, cfg, template: str) -> None:
    original_train_dataloader = model.train_dataloader
    state = {"manifest": None}

    def train_dataloader_with_epoch_manifest(self):
        trainer = getattr(self, "trainer", None)
        epoch0 = int(getattr(trainer, "current_epoch", 0) or 0)
        manifest = resolve_epoch_manifest(template, epoch0)
        require_file(manifest, f"epoch {epoch0 + 1} train manifest")
        manifest_str = str(manifest)
        if state["manifest"] != manifest_str:
            with open_dict(cfg):
                cfg.train_ds.manifest_filepath = manifest_str
            print(f"Using train manifest for epoch {epoch0 + 1}: {manifest_str}")
            self.setup_training_data(cfg.train_ds)
            state["manifest"] = manifest_str
        return original_train_dataloader()

    model.train_dataloader = types.MethodType(train_dataloader_with_epoch_manifest, model)


def load_encoder_weights_from_base(model, base_ckpt: Path) -> None:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    print(f"Restoring base model for weight transfer: {base_ckpt}")
    base_model = EncDecSpeakerLabelModel.restore_from(restore_path=str(base_ckpt), map_location="cpu")
    base_state = base_model.state_dict()
    new_state = model.state_dict()

    compatible = {}
    skipped = []
    for key, value in base_state.items():
        if key not in new_state:
            skipped.append((key, "missing-in-new-model"))
            continue
        if tuple(value.shape) != tuple(new_state[key].shape):
            skipped.append((key, f"{tuple(value.shape)} -> {tuple(new_state[key].shape)}"))
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Transferred tensors: {len(compatible)}")
    print(f"Skipped tensors: {len(skipped)}")
    for key, reason in skipped:
        print(f"  skip {key}: {reason}")
    if missing:
        print(f"Missing after transfer: {len(missing)}")
        for key in missing:
            print(f"  missing {key}")
    if unexpected:
        print(f"Unexpected after transfer: {len(unexpected)}")
        for key in unexpected:
            print(f"  unexpected {key}")

    del base_model


def configure_trainable_parameters(model, freeze_encoder: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

    if freeze_encoder:
        if not hasattr(model, "encoder"):
            raise AttributeError("Model does not expose an encoder module to freeze")
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    print(f"trainable parameters: {trainable:,}")
    print(f"frozen parameters: {frozen:,}")


def main() -> None:
    args = parse_args()
    log_fh = setup_text_logging(args)
    require_file(args.base_ckpt, "base AmberNet checkpoint")
    if args.train_manifest_template:
        for epoch0 in range(args.max_epochs):
            require_file(
                resolve_epoch_manifest(args.train_manifest_template, epoch0),
                f"epoch {epoch0 + 1} train manifest",
            )
        args.train_manifest = resolve_epoch_manifest(args.train_manifest_template, 0)
    require_file(args.train_manifest, "train manifest")
    require_file(args.val_manifest, "val manifest")
    if not args.skip_eval:
        require_file(args.eval_manifest, "eval manifest")
    labels = parse_labels(args.labels)

    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    print("Finetune labels:")
    print(",".join(labels))
    print(f"num_classes: {len(labels)}")
    print(f"loss weight: {args.loss_weight}")
    if args.train_manifest_template:
        print(f"train manifest template: {args.train_manifest_template}")
        print("train dataloader reload: every epoch")
    print(f"ambernet augmentor enabled: {args.enable_ambernet_augmentor}")
    print(f"wandb enabled: {args.wandb}")
    if args.wandb:
        print(f"wandb project: {args.wandb_project}")
        print(f"wandb run name: {args.wandb_name or args.work_dir.name}")
        print(f"wandb mode: {args.wandb_mode}")
    if args.freeze_encoder:
        print("mode: encoder-frozen finetune; encoder frozen, decoder/classification head trainable")
    else:
        print("mode: full-model finetune; all transferred parameters remain trainable")

    trainer = build_trainer(args)

    print(f"Reading base config from: {args.base_ckpt}")
    base_cfg = EncDecSpeakerLabelModel.restore_from(
        restore_path=str(args.base_ckpt),
        map_location="cpu",
        return_config=True,
    )
    cfg = update_cfg_for_finetune(OmegaConf.create(base_cfg), args, labels)

    print(f"Instantiating {len(labels)}-class AmberNet model from modified config")
    model = EncDecSpeakerLabelModel(cfg=cfg, trainer=trainer)
    load_encoder_weights_from_base(model, args.base_ckpt)
    configure_trainable_parameters(model, args.freeze_encoder)

    if hasattr(model, "setup_training_data"):
        model.setup_training_data(cfg.train_ds)
    if args.train_manifest_template:
        install_rotating_train_manifest_loader(model, cfg, args.train_manifest_template)
    if hasattr(model, "setup_validation_data"):
        model.setup_validation_data(cfg.validation_ds)
    if hasattr(model, "setup_optimization"):
        model.setup_optimization(cfg.optim)

    print("Starting training")
    trainer.fit(model, ckpt_path=str(args.resume_ckpt) if args.resume_ckpt else None)

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(args.out_ckpt))
    print(f"Saved finetuned model: {args.out_ckpt}")

    if not args.skip_eval and hasattr(model, "setup_test_data"):
        print("Running eval manifest")
        model.setup_test_data(cfg.test_ds)
        trainer.test(model)

    log_fh.close()


if __name__ == "__main__":
    main()
