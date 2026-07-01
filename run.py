import argparse
from pathlib import Path

import api_endpoint


def main():
    parser = argparse.ArgumentParser(description="Run Silero VAD + AmberNet LID on a single file")
    parser.add_argument("file_path", type=str, help="Path to audio or video file")
    parser.add_argument("--top_k", type=int, default=3, help="Number of predictions to return per segment")
    parser.add_argument("--vad_threshold", type=float, default=0.5, help="Silero VAD threshold")
    parser.add_argument(
        "--min_speech_duration_ms",
        type=int,
        default=250,
        help="Minimum speech segment duration in milliseconds",
    )
    parser.add_argument(
        "--min_silence_duration_ms",
        type=int,
        default=150,
        help="Minimum silence duration used by VAD in milliseconds",
    )
    parser.add_argument(
        "--merge_gap_sec",
        type=float,
        default=0.5,
        help="Merge adjacent same-language segments if gap is smaller than this",
    )
    parser.add_argument(
        "--merge_same_language",
        action="store_true",
        help="Merge consecutive same-language segments regardless of the silence gap",
    )
    parser.add_argument(
        "--smooth_language_islands",
        action="store_true",
        help="Replace short low-confidence language islands between same-language neighbors",
    )
    parser.add_argument(
        "--max_island_duration_sec",
        type=float,
        default=2.0,
        help="Maximum duration for a short language island to be smoothed",
    )
    parser.add_argument(
        "--island_score_threshold",
        type=float,
        default=0.6,
        help="Maximum score for a language island to be considered low confidence",
    )
    parser.add_argument(
        "--lid_window_sec",
        type=float,
        default=0.0,
        help="Split long VAD speech segments into fixed-size LID windows; 0 disables windowing",
    )
    parser.add_argument(
        "--lid_hop_sec",
        type=float,
        default=0.0,
        help="Hop size for LID windows; 0 uses lid_window_sec",
    )
    parser.add_argument(
        "--allowed_languages",
        type=str,
        default="",
        help="Comma-separated allowed language labels, e.g. en,es,fr,zh; empty means all labels",
    )
    parser.add_argument(
        "--unknown_threshold",
        type=float,
        default=0.0,
        help=(
            "If > 0, fall back to the second-best language when 'unknown' wins with a score "
            "below this threshold; 0 disables the fallback"
        ),
    )
    parser.add_argument("--classifier_id", type=str, default="lid_ambernet_v1", help="Classifier ID")
    parser.add_argument(
        "--model_version",
        type=str,
        default="nemo-ambernet",
        help="Model version",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a NeMo .nemo checkpoint; defaults to NEMO_LID_CKPT or checkpoints/ambernet.nemo",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Path to save output JSON file")
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Drop final language segments whose score is below this threshold",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if ckpt_path.suffix != ".nemo":
            raise ValueError(
                f"--ckpt expects a NeMo .nemo checkpoint for inference, got: {ckpt_path}"
            )
        api_endpoint.MODEL_CKPT_PATH = str(ckpt_path)
        api_endpoint._get_lid_model.cache_clear()

    result = api_endpoint.detect_local_file(
        file_path=str(file_path),
        top_k=args.top_k,
        vad_threshold=args.vad_threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        merge_gap_sec=args.merge_gap_sec,
        merge_same_language=args.merge_same_language,
        smooth_language_islands=args.smooth_language_islands,
        max_island_duration_sec=args.max_island_duration_sec,
        island_score_threshold=args.island_score_threshold,
        lid_window_sec=args.lid_window_sec,
        lid_hop_sec=args.lid_hop_sec,
        allowed_languages=args.allowed_languages,
        unknown_threshold=args.unknown_threshold,
        min_score=args.min_score,
        classifier_id=args.classifier_id,
        model_version=args.model_version,
        output_json=args.output_json,
    )

    print(f"Processed file: {result.audio_file_id}")
    print(
        f"Detected {len(result.labels)} speech segments"
        + (f" (score >= {args.min_score})" if args.min_score > 0.0 else "")
    )
    print("Segments:")
    for segment in result.labels:
        print(
            f"- {segment.start_time:.2f}s-{segment.end_time:.2f}s "
            f"language={segment.language_code} "
            f"score={segment.scores:.4f}"
        )


if __name__ == "__main__":
    main()
