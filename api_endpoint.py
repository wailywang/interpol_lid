import json
import os
import shutil
import sys
import tempfile
import uuid
from array import array
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Query, UploadFile
from pydantic import BaseModel
from pydub import AudioSegment

CLASSIFIER_ID = "lid_ambernet_v1"
MODEL_VERSION = "nemo-ambernet"
MODEL_NAME = "langid_ambernet"
MODEL_CKPT_PATH = os.environ.get(
    "NEMO_LID_CKPT",
    os.path.join(os.path.dirname(__file__), "checkpoints", "ambernet.nemo"),
)
TARGET_SR = 16000

VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 150
MERGE_GAP_SEC = 0.5
MERGE_SAME_LANGUAGE = False
SMOOTH_LANGUAGE_ISLANDS = False
MAX_ISLAND_DURATION_SEC = 2.0
ISLAND_SCORE_THRESHOLD = 0.6
LID_WINDOW_SEC = 0.0
LID_HOP_SEC = 0.0
ALLOWED_LANGUAGES = ""
UNKNOWN_LABEL = "unknown"
UNKNOWN_THRESHOLD = 0.0
MIN_SCORE = 0.0

# Singlish Accent Head — optional second-stage classifier.
# Set SINGLISH_HEAD_CKPT to a .nemo path to enable; leave empty to disable.
SINGLISH_HEAD_CKPT = os.environ.get("SINGLISH_HEAD_CKPT", "")
# P(en_sg) must exceed this threshold to override LID's "en" prediction.
SINGLISH_EN_SG_THRESHOLD = float(os.environ.get("SINGLISH_EN_SG_THRESHOLD", "0.5"))


def _configure_ffmpeg_tools() -> None:
    candidate_dirs = []

    env_ffmpeg = os.environ.get("FFMPEG_BINARY")
    env_ffprobe = os.environ.get("FFPROBE_BINARY")
    for binary in (env_ffmpeg, env_ffprobe):
        if binary:
            candidate_dirs.append(str(Path(binary).expanduser().resolve().parent))

    python_path = Path(sys.executable).resolve()
    for parent in python_path.parents:
        candidate_dirs.append(str(parent / "bin"))

    candidate_dirs.append("/export/home2/wa0009xi/miniconda3/bin")

    existing_dirs = []
    for directory in candidate_dirs:
        if os.path.isdir(directory) and directory not in existing_dirs:
            existing_dirs.append(directory)

    if existing_dirs:
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join(existing_dirs + path_parts)

    ffmpeg = env_ffmpeg or shutil.which("ffmpeg")
    ffprobe = env_ffprobe or shutil.which("ffprobe")
    if ffmpeg:
        AudioSegment.converter = ffmpeg
        AudioSegment.ffmpeg = ffmpeg
    if ffprobe:
        AudioSegment.ffprobe = ffprobe


_configure_ffmpeg_tools()


class LanguagePrediction(BaseModel):
    language_code: str
    scores: float
    rank: int


class LanguageSegment(BaseModel):
    start_time: float
    end_time: float
    duration: float
    language_code: str
    scores: float
    predictions: List[LanguagePrediction]


class PeriodLanguageDistribution(BaseModel):
    start_time: float
    end_time: float
    duration: float
    language_code: str
    scores: float
    distribution: List[LanguagePrediction]


class Top1DurationDistribution(BaseModel):
    language_code: str
    duration: float
    ratio: float
    percentage: float
    segment_count: int


class ConfidenceWeightedDistribution(BaseModel):
    language_code: str
    score_seconds: float
    ratio: float
    percentage: float


class LanguageStatistics(BaseModel):
    final_language_code: str | None
    total_speech_duration: float
    total_periods: int
    period_distributions: List[PeriodLanguageDistribution]
    top1_duration_distribution: List[Top1DurationDistribution]
    confidence_weighted_distribution: List[ConfidenceWeightedDistribution]


class ResultOut(BaseModel):
    audio_file_id: str
    classifier_id: str
    model_version: str
    run_id: str
    event_type: str
    labels: List[LanguageSegment]
    language_statistics: LanguageStatistics
    created_at: str


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() != 2:
        raise ValueError("waveform should be in shape (channels, time)")
    return wav.mean(dim=0) if wav.size(0) > 1 else wav[0]


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def _array_to_float_tensor(samples: array, sample_width: int) -> torch.Tensor:
    if not samples:
        return torch.empty(0, dtype=torch.float32)

    wav = torch.tensor(samples, dtype=torch.float32)
    scale = float(1 << (8 * sample_width - 1))
    return torch.clamp(wav / scale, min=-1.0, max=1.0)


def _load_audiosegment_to_waveform(audio: AudioSegment, target_sr: int) -> torch.Tensor:
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    return _array_to_float_tensor(audio.get_array_of_samples(), audio.sample_width)


def _read_soundfile_to_waveform(file_path: str, target_sr: int) -> torch.Tensor:
    data, sr = sf.read(file_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    wav = _to_mono(waveform)
    return _resample_if_needed(wav, sr, target_sr)


def _save_waveform_to_wav(file_path: str, wav: torch.Tensor, sample_rate: int) -> None:
    samples = wav.detach().cpu().to(torch.float32).clamp(-1.0, 1.0).numpy()
    sf.write(file_path, samples, sample_rate, subtype="PCM_16")


def _read_uploadfile_to_waveform(file: UploadFile, target_sr: int = TARGET_SR) -> torch.Tensor:
    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        audio = AudioSegment.from_file(tmp_path)
        return _load_audiosegment_to_waveform(audio, target_sr)
    except Exception as exc:
        raise ValueError(f"Cannot read file {file.filename}: {exc}") from exc
    finally:
        os.remove(tmp_path)


def _read_local_file_to_waveform(file_path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    try:
        audio = AudioSegment.from_file(file_path)
        return _load_audiosegment_to_waveform(audio, target_sr)
    except Exception as pydub_exc:
        try:
            return _read_soundfile_to_waveform(file_path, target_sr)
        except Exception as sf_exc:
            raise ValueError(
                f"Cannot read local audio file {file_path}: pydub failed with {pydub_exc}; "
                f"soundfile failed with {sf_exc}"
            ) from sf_exc


@lru_cache(maxsize=1)
def _get_vad_bundle():
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise ImportError(
            "silero-vad is required for VAD. Install it with `pip install silero-vad`."
        ) from exc

    model = load_silero_vad()
    return model, get_speech_timestamps


@lru_cache(maxsize=1)
def _get_lid_model():
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
    except ImportError as exc:
        raise ImportError(
            "nemo-toolkit[asr] is required for AmberNet LID. "
            "Install it with `pip install nemo-toolkit[asr]`."
        ) from exc

    if MODEL_CKPT_PATH and os.path.exists(MODEL_CKPT_PATH):
        model = EncDecSpeakerLabelModel.restore_from(restore_path=MODEL_CKPT_PATH)
    else:
        model = EncDecSpeakerLabelModel.from_pretrained(model_name=MODEL_NAME)

    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


@lru_cache(maxsize=1)
def _get_singlish_head():
    """Load the binary en/en_sg accent head. Returns None if not configured."""
    if not SINGLISH_HEAD_CKPT or not os.path.exists(SINGLISH_HEAD_CKPT):
        return None
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
    except ImportError:
        return None
    model = EncDecSpeakerLabelModel.restore_from(restore_path=SINGLISH_HEAD_CKPT)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def _get_model_labels(model: Any) -> List[str]:
    labels = None

    if hasattr(model, "cfg") and getattr(model.cfg, "labels", None):
        labels = list(model.cfg.labels)
    elif hasattr(model, "cfg") and getattr(model.cfg, "train_ds", None):
        train_ds = model.cfg.train_ds
        if getattr(train_ds, "labels", None):
            labels = list(train_ds.labels)
    elif hasattr(model, "decoder") and getattr(model.decoder, "labels", None):
        labels = list(model.decoder.labels)
    elif hasattr(model, "_cfg") and getattr(model._cfg, "labels", None):
        labels = list(model._cfg.labels)
    elif hasattr(model, "_cfg") and getattr(model._cfg, "train_ds", None):
        train_ds = model._cfg.train_ds
        if getattr(train_ds, "labels", None):
            labels = list(train_ds.labels)

    if not labels:
        raise ValueError("Unable to resolve AmberNet label list from the loaded NeMo model")

    # AmberNet checkpoints can carry train/val/test label lists in different
    # orders. The decoder logits follow NeMo's canonical sorted label order.
    return sorted(labels)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(output, "logits"):
        return output.logits
    raise ValueError("Unable to extract logits from AmberNet output")


def _run_ambernet_logits(model: Any, segment: torch.Tensor, sample_rate: int) -> torch.Tensor:
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    input_signal = segment.unsqueeze(0).to(device)
    input_signal_length = torch.tensor([segment.numel()], device=device)

    with torch.inference_mode():
        try:
            output = model(input_signal=input_signal, input_signal_length=input_signal_length)
            logits = _extract_logits(output)
            return logits[0].detach().cpu()
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            _save_waveform_to_wav(tmp_wav.name, segment, sample_rate)
            tmp_path = tmp_wav.name

        try:
            if hasattr(model, "transcribe"):
                try:
                    output = model.transcribe(paths2audio_files=[tmp_path], batch_size=1, logprobs=True)
                except TypeError:
                    output = model.transcribe(audio=[tmp_path], batch_size=1, logprobs=True)
                logits = _extract_logits(output)
                return logits[0].detach().cpu()

            if hasattr(model, "infer_file"):
                output = model.infer_file(tmp_path)
                logits = _extract_logits(output)
                return logits[0].detach().cpu()
        finally:
            os.remove(tmp_path)

    raise RuntimeError("AmberNet inference failed for the provided segment")


def _parse_allowed_languages(allowed_languages: str | None) -> List[str]:
    if not allowed_languages:
        return []
    return [item.strip() for item in allowed_languages.split(",") if item.strip()]


def _predict_segment_languages(
    segment: torch.Tensor,
    top_k: int,
    allowed_languages: str | None = None,
    unknown_threshold: float = UNKNOWN_THRESHOLD,
) -> List[LanguagePrediction]:
    if segment.numel() == 0:
        raise ValueError("Cannot run LID on an empty segment")

    if segment.numel() < TARGET_SR:
        segment = torch.nn.functional.pad(segment, (0, TARGET_SR - segment.numel()))

    model = _get_lid_model()
    labels = _get_model_labels(model)
    logits = _run_ambernet_logits(model, segment, TARGET_SR)
    allowed_labels = _parse_allowed_languages(allowed_languages)
    if allowed_labels:
        label_to_index = {label: index for index, label in enumerate(labels)}
        unknown_labels = [label for label in allowed_labels if label not in label_to_index]
        if unknown_labels:
            raise ValueError(
                "Allowed language labels are not supported by this checkpoint: "
                + ", ".join(unknown_labels)
            )
        allowed_indices = torch.tensor([label_to_index[label] for label in allowed_labels], dtype=torch.long)
        labels = allowed_labels
        logits = logits[allowed_indices]

    probs = torch.softmax(logits, dim=-1)

    k = max(1, min(top_k, probs.numel()))
    values, indices = torch.topk(probs, k=k)

    predictions = []
    for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        predictions.append(
            LanguagePrediction(
                language_code=labels[index],
                scores=float(round(score, 4)),
                rank=rank,
            )
        )

    if (
        unknown_threshold > 0.0
        and len(predictions) >= 2
        and predictions[0].language_code == UNKNOWN_LABEL
        and predictions[0].scores < unknown_threshold
    ):
        predictions[0], predictions[1] = predictions[1], predictions[0]
        predictions[0].rank, predictions[1].rank = 1, 2

    # Singlish Accent Head: if top-1 is "en", run binary en/en_sg classifier.
    if predictions and predictions[0].language_code == "en":
        singlish_head = _get_singlish_head()
        if singlish_head is not None:
            head_labels = _get_model_labels(singlish_head)
            head_logits = _run_ambernet_logits(singlish_head, segment, TARGET_SR)
            head_probs = torch.softmax(head_logits, dim=-1)
            if "en_sg" in head_labels:
                ensg_idx = head_labels.index("en_sg")
                p_ensg = float(head_probs[ensg_idx])
                if p_ensg > SINGLISH_EN_SG_THRESHOLD:
                    predictions[0] = LanguagePrediction(
                        language_code="en_sg",
                        scores=float(round(p_ensg, 4)),
                        rank=1,
                    )

    return predictions


def _merge_adjacent_segments(labels: List[LanguageSegment], gap_threshold: float) -> List[LanguageSegment]:
    if not labels:
        return []

    merged = [_copy_model(labels[0])]
    for current in labels[1:]:
        previous = merged[-1]
        gap = current.start_time - previous.end_time

        if previous.language_code == current.language_code and gap <= gap_threshold:
            total_duration = previous.duration + current.duration
            weighted_score = (
                previous.scores * previous.duration + current.scores * current.duration
            ) / total_duration if total_duration > 0 else max(previous.scores, current.scores)

            previous.end_time = current.end_time
            previous.duration = float(round(previous.end_time - previous.start_time, 3))
            previous.scores = float(round(weighted_score, 4))
            previous.predictions = current.predictions
        else:
            merged.append(_copy_model(current))

    return merged


def _merge_consecutive_same_language(labels: List[LanguageSegment]) -> List[LanguageSegment]:
    if not labels:
        return []

    merged = [_copy_model(labels[0])]
    for current in labels[1:]:
        previous = merged[-1]

        if previous.language_code == current.language_code:
            total_duration = previous.duration + current.duration
            weighted_score = (
                previous.scores * previous.duration + current.scores * current.duration
            ) / total_duration if total_duration > 0 else max(previous.scores, current.scores)

            previous.end_time = current.end_time
            previous.duration = float(round(previous.end_time - previous.start_time, 3))
            previous.scores = float(round(weighted_score, 4))
            previous.predictions = current.predictions
        else:
            merged.append(_copy_model(current))

    return merged


def _smooth_language_islands(
    labels: List[LanguageSegment],
    max_island_duration_sec: float,
    island_score_threshold: float,
) -> List[LanguageSegment]:
    if len(labels) < 3:
        return labels

    smoothed = [_copy_model(label) for label in labels]
    for index in range(1, len(smoothed) - 1):
        previous = smoothed[index - 1]
        current = smoothed[index]
        following = smoothed[index + 1]

        if (
            previous.language_code == following.language_code
            and current.language_code != previous.language_code
            and current.duration <= max_island_duration_sec
            and current.scores <= island_score_threshold
        ):
            current.language_code = previous.language_code
            current.scores = min(previous.scores, following.scores)
            current.predictions = previous.predictions

    return _merge_consecutive_same_language(smoothed)


def _filter_low_score_segments(labels: List[LanguageSegment], min_score: float) -> List[LanguageSegment]:
    if min_score <= 0.0:
        return labels
    return [_copy_model(label) for label in labels if label.scores >= min_score]


def _copy_model(model: BaseModel) -> BaseModel:
    if hasattr(model, "model_copy"):
        return model.model_copy(deep=True)
    return model.copy(deep=True)


def _dump_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _detect_speech_segments(
    wav: torch.Tensor,
    vad_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
) -> List[dict[str, int]]:
    vad_model, get_speech_timestamps = _get_vad_bundle()
    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        threshold=vad_threshold,
        sampling_rate=TARGET_SR,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=False,
    )
    return [dict(item) for item in timestamps]


def _iter_lid_windows(
    speech_start_sample: int,
    speech_end_sample: int,
    lid_window_sec: float,
    lid_hop_sec: float,
) -> List[tuple[int, int]]:
    speech_len = speech_end_sample - speech_start_sample
    if speech_len <= 0:
        return []

    window_samples = int(lid_window_sec * TARGET_SR)
    if lid_window_sec <= 0 or window_samples <= 0 or speech_len <= window_samples:
        return [(speech_start_sample, speech_end_sample)]

    hop_samples = int((lid_hop_sec if lid_hop_sec > 0 else lid_window_sec) * TARGET_SR)
    hop_samples = max(1, hop_samples)

    windows = []
    start = speech_start_sample
    min_window_samples = min(TARGET_SR, window_samples)
    while start < speech_end_sample:
        end = min(start + window_samples, speech_end_sample)
        if end - start >= min_window_samples:
            windows.append((start, end))
        if end >= speech_end_sample:
            break
        start += hop_samples

    return windows


def _build_segment_labels(
    wav: torch.Tensor,
    top_k: int,
    vad_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    merge_gap_sec: float,
    merge_same_language: bool = MERGE_SAME_LANGUAGE,
    smooth_language_islands: bool = SMOOTH_LANGUAGE_ISLANDS,
    max_island_duration_sec: float = MAX_ISLAND_DURATION_SEC,
    island_score_threshold: float = ISLAND_SCORE_THRESHOLD,
    lid_window_sec: float = LID_WINDOW_SEC,
    lid_hop_sec: float = LID_HOP_SEC,
    allowed_languages: str | None = ALLOWED_LANGUAGES,
    unknown_threshold: float = UNKNOWN_THRESHOLD,
    min_score: float = MIN_SCORE,
) -> List[LanguageSegment]:
    speech_segments = _detect_speech_segments(
        wav=wav,
        vad_threshold=vad_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    labels: List[LanguageSegment] = []
    for speech in speech_segments:
        speech_start_sample = int(speech["start"])
        speech_end_sample = int(speech["end"])
        for start_sample, end_sample in _iter_lid_windows(
            speech_start_sample=speech_start_sample,
            speech_end_sample=speech_end_sample,
            lid_window_sec=lid_window_sec,
            lid_hop_sec=lid_hop_sec,
        ):
            segment = wav[start_sample:end_sample]
            if segment.numel() == 0:
                continue

            predictions = _predict_segment_languages(
                segment,
                top_k=top_k,
                allowed_languages=allowed_languages,
                unknown_threshold=unknown_threshold,
            )
            start_time = round(start_sample / TARGET_SR, 3)
            end_time = round(end_sample / TARGET_SR, 3)
            labels.append(
                LanguageSegment(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    duration=float(round(end_time - start_time, 3)),
                    language_code=predictions[0].language_code,
                    scores=float(predictions[0].scores),
                    predictions=predictions,
                )
            )

    labels = _merge_adjacent_segments(labels, gap_threshold=merge_gap_sec)
    if merge_same_language:
        labels = _merge_consecutive_same_language(labels)
    if smooth_language_islands:
        labels = _smooth_language_islands(
            labels,
            max_island_duration_sec=max_island_duration_sec,
            island_score_threshold=island_score_threshold,
        )
    labels = _filter_low_score_segments(labels, min_score=min_score)
    return labels


def _build_result(
    audio_file_id: str,
    labels: List[LanguageSegment],
    classifier_id: str,
    model_version: str,
    run_id: str,
    created_at: str,
) -> ResultOut:
    language_statistics = _build_language_statistics(labels)
    return ResultOut(
        audio_file_id=audio_file_id,
        classifier_id=classifier_id,
        model_version=model_version,
        run_id=run_id,
        event_type="language id",
        labels=labels,
        language_statistics=language_statistics,
        created_at=created_at,
    )


def _build_language_statistics(labels: List[LanguageSegment]) -> LanguageStatistics:
    total_duration = sum(max(label.duration, 0.0) for label in labels)
    period_distributions = [
        PeriodLanguageDistribution(
            start_time=label.start_time,
            end_time=label.end_time,
            duration=label.duration,
            language_code=label.language_code,
            scores=label.scores,
            distribution=label.predictions,
        )
        for label in labels
    ]

    top1_duration_by_language: dict[str, float] = {}
    segment_count_by_language: dict[str, int] = {}
    confidence_weight_by_language: dict[str, float] = {}

    for label in labels:
        duration = max(label.duration, 0.0)
        top1_duration_by_language[label.language_code] = (
            top1_duration_by_language.get(label.language_code, 0.0) + duration
        )
        segment_count_by_language[label.language_code] = (
            segment_count_by_language.get(label.language_code, 0) + 1
        )

        reported_probability = 0.0
        for prediction in label.predictions:
            score = max(float(prediction.scores), 0.0)
            reported_probability += score
            confidence_weight_by_language[prediction.language_code] = (
                confidence_weight_by_language.get(prediction.language_code, 0.0) + score * duration
            )

        residual_probability = max(0.0, 1.0 - reported_probability)
        if residual_probability > 1e-6:
            confidence_weight_by_language["unreported"] = (
                confidence_weight_by_language.get("unreported", 0.0) + residual_probability * duration
            )

    top1_duration_distribution = [
        Top1DurationDistribution(
            language_code=language_code,
            duration=float(round(duration, 3)),
            ratio=float(round(duration / total_duration, 4)) if total_duration > 0 else 0.0,
            percentage=float(round(duration / total_duration * 100.0, 2)) if total_duration > 0 else 0.0,
            segment_count=segment_count_by_language[language_code],
        )
        for language_code, duration in sorted(
            top1_duration_by_language.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    confidence_total = sum(confidence_weight_by_language.values())
    confidence_weighted_distribution = [
        ConfidenceWeightedDistribution(
            language_code=language_code,
            score_seconds=float(round(score_seconds, 4)),
            ratio=float(round(score_seconds / confidence_total, 4)) if confidence_total > 0 else 0.0,
            percentage=float(round(score_seconds / confidence_total * 100.0, 2))
            if confidence_total > 0
            else 0.0,
        )
        for language_code, score_seconds in sorted(
            confidence_weight_by_language.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    final_language_code = None
    for item in confidence_weighted_distribution:
        if item.language_code != "unreported":
            final_language_code = item.language_code
            break

    return LanguageStatistics(
        final_language_code=final_language_code,
        total_speech_duration=float(round(total_duration, 3)),
        total_periods=len(labels),
        period_distributions=period_distributions,
        top1_duration_distribution=top1_duration_distribution,
        confidence_weighted_distribution=confidence_weighted_distribution,
    )


app = FastAPI(title="Language Identification API", version="1.0.0")


@app.post("/detect", response_model=List[ResultOut])
def detect(
    files: List[UploadFile] = File(..., description="audio or video files"),
    top_k: int = Query(3, ge=1, le=20, description="number of top language predictions per segment"),
    vad_threshold: float = Query(VAD_THRESHOLD, ge=0.0, le=1.0, description="Silero VAD threshold"),
    min_speech_duration_ms: int = Query(
        MIN_SPEECH_DURATION_MS,
        ge=50,
        le=10000,
        description="minimum speech segment duration in milliseconds",
    ),
    min_silence_duration_ms: int = Query(
        MIN_SILENCE_DURATION_MS,
        ge=50,
        le=10000,
        description="minimum silence duration used by VAD in milliseconds",
    ),
    merge_gap_sec: float = Query(
        MERGE_GAP_SEC,
        ge=0.0,
        le=10.0,
        description="merge adjacent segments with the same language if the gap is small",
    ),
    merge_same_language: bool = Query(
        MERGE_SAME_LANGUAGE,
        description="merge consecutive same-language segments regardless of the silence gap",
    ),
    smooth_language_islands: bool = Query(
        SMOOTH_LANGUAGE_ISLANDS,
        description="replace short low-confidence language islands between same-language neighbors",
    ),
    max_island_duration_sec: float = Query(
        MAX_ISLAND_DURATION_SEC,
        ge=0.0,
        le=30.0,
        description="maximum duration for a short language island to be smoothed",
    ),
    island_score_threshold: float = Query(
        ISLAND_SCORE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="maximum score for a language island to be considered low confidence",
    ),
    lid_window_sec: float = Query(
        LID_WINDOW_SEC,
        ge=0.0,
        le=120.0,
        description="split long VAD speech segments into fixed-size LID windows; 0 disables windowing",
    ),
    lid_hop_sec: float = Query(
        LID_HOP_SEC,
        ge=0.0,
        le=120.0,
        description="hop size for LID windows; 0 uses lid_window_sec",
    ),
    allowed_languages: str = Query(
        ALLOWED_LANGUAGES,
        description="comma-separated list of allowed language labels; empty means all checkpoint labels",
    ),
    unknown_threshold: float = Query(
        UNKNOWN_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "if > 0, fall back to the second-best language when 'unknown' wins with a score "
            "below this threshold; 0 disables the fallback"
        ),
    ),
    min_score: float = Query(
        MIN_SCORE,
        ge=0.0,
        le=1.0,
        description="drop final language segments whose top-1 score is below this threshold",
    ),
    classifier_id: str = Query(CLASSIFIER_ID),
    model_version: str = Query(MODEL_VERSION),
):
    run_id = f"{classifier_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:3]}"
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    results = []
    for file in files:
        wav = _read_uploadfile_to_waveform(file, TARGET_SR)
        labels = _build_segment_labels(
            wav=wav,
            top_k=top_k,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            merge_gap_sec=merge_gap_sec,
            merge_same_language=merge_same_language,
            smooth_language_islands=smooth_language_islands,
            max_island_duration_sec=max_island_duration_sec,
            island_score_threshold=island_score_threshold,
            lid_window_sec=lid_window_sec,
            lid_hop_sec=lid_hop_sec,
            allowed_languages=allowed_languages,
            unknown_threshold=unknown_threshold,
            min_score=min_score,
        )
        audio_file_id = file.filename or f"audio_{uuid.uuid4().hex[:8]}"
        results.append(
            _build_result(
                audio_file_id=audio_file_id,
                labels=labels,
                classifier_id=classifier_id,
                model_version=model_version,
                run_id=run_id,
                created_at=created_at,
            )
        )

    output_filename = f"lid_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w") as file:
        json.dump([_dump_model(result) for result in results], file, indent=2)

    return results


def detect_local_file(
    file_path: str,
    top_k: int = 3,
    vad_threshold: float = VAD_THRESHOLD,
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
    merge_gap_sec: float = MERGE_GAP_SEC,
    merge_same_language: bool = MERGE_SAME_LANGUAGE,
    smooth_language_islands: bool = SMOOTH_LANGUAGE_ISLANDS,
    max_island_duration_sec: float = MAX_ISLAND_DURATION_SEC,
    island_score_threshold: float = ISLAND_SCORE_THRESHOLD,
    lid_window_sec: float = LID_WINDOW_SEC,
    lid_hop_sec: float = LID_HOP_SEC,
    allowed_languages: str | None = ALLOWED_LANGUAGES,
    unknown_threshold: float = UNKNOWN_THRESHOLD,
    min_score: float = MIN_SCORE,
    classifier_id: str = CLASSIFIER_ID,
    model_version: str = MODEL_VERSION,
    output_json: str | None = None,
) -> ResultOut:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    wav = _read_local_file_to_waveform(file_path, TARGET_SR)
    labels = _build_segment_labels(
        wav=wav,
        top_k=top_k,
        vad_threshold=vad_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        merge_gap_sec=merge_gap_sec,
        merge_same_language=merge_same_language,
        smooth_language_islands=smooth_language_islands,
        max_island_duration_sec=max_island_duration_sec,
        island_score_threshold=island_score_threshold,
        lid_window_sec=lid_window_sec,
        lid_hop_sec=lid_hop_sec,
        allowed_languages=allowed_languages,
        unknown_threshold=unknown_threshold,
        min_score=min_score,
    )

    run_id = f"{classifier_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:3]}"
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    result = _build_result(
        audio_file_id=os.path.basename(file_path),
        labels=labels,
        classifier_id=classifier_id,
        model_version=model_version,
        run_id=run_id,
        created_at=created_at,
    )

    output_path = output_json or f"lid_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump([_dump_model(result)], file, indent=2)

    return result
