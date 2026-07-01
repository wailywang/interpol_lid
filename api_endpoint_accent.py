#!/usr/bin/env python3
"""FastAPI endpoint that extends api_endpoint with a CommonAccent second-stage
English accent classifier.

Run instead of api_endpoint.py:
    uvicorn api_endpoint_accent:app --host 0.0.0.0 --port 8000

Environment variables:
    ACCENT_HEAD_MODEL      HuggingFace model ID or local path
                           (default: Jzuluaga/accent-id-commonaccent_ecapa)
    ACCENT_HEAD_SAVEDIR    Local directory to cache the downloaded model
                           (default: checkpoints/commonaccent_ecapa/)
    ACCENT_HEAD_MIN_SCORE  Minimum accent-head confidence to override "en";
                           0.0 = always override (default)

All other env vars from api_endpoint.py (NEMO_LID_CKPT, SINGLISH_HEAD_CKPT, etc.)
are still respected but SINGLISH_HEAD_CKPT is superseded by this accent head when
both are set — set SINGLISH_HEAD_CKPT="" to avoid running both.
"""
import os
from functools import lru_cache

import torch

# ── CommonAccent config ──────────────────────────────────────────────────────
ACCENT_HEAD_MODEL = os.environ.get(
    "ACCENT_HEAD_MODEL", "Jzuluaga/accent-id-commonaccent_ecapa"
)
ACCENT_HEAD_SAVEDIR = os.environ.get(
    "ACCENT_HEAD_SAVEDIR",
    os.path.join(os.path.dirname(__file__), "checkpoints", "commonaccent_ecapa"),
)
ACCENT_HEAD_MIN_SCORE = float(os.environ.get("ACCENT_HEAD_MIN_SCORE", "0.0"))

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


@lru_cache(maxsize=1)
def _get_accent_head():
    """Load the CommonAccent SpeechBrain accent classifier. Returns None if not configured."""
    if not ACCENT_HEAD_MODEL:
        return None
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError as exc:
        raise ImportError(
            "speechbrain is required for the CommonAccent accent head. "
            "Install it with: pip install speechbrain"
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EncoderClassifier.from_hparams(
        source=ACCENT_HEAD_MODEL,
        savedir=ACCENT_HEAD_SAVEDIR,
        run_opts={"device": device},
    )


def _run_accent_head(segment: torch.Tensor) -> tuple[str, float] | None:
    """Run CommonAccent on a waveform segment. Returns (accent_code, probability) or None."""
    model = _get_accent_head()
    if model is None:
        return None
    wavs = segment.unsqueeze(0).cpu()
    wav_lens = torch.tensor([1.0])
    out_prob, score, index, text_lab = model.classify_batch(wavs, wav_lens)
    accent_raw = text_lab[0].strip().lower()
    accent_code = _COMMONACCENT_TO_CODE.get(accent_raw, f"en_{accent_raw}")
    # score is log-softmax of the winning class; .exp() converts to probability
    prob = float(torch.clamp(score[0].exp(), 0.0, 1.0))
    return accent_code, prob


# ── Patch api_endpoint before its app is used ────────────────────────────────
import api_endpoint as _base  # noqa: E402  (import after env setup above)

_original_predict = _base._predict_segment_languages


def _predict_with_accent(
    segment: torch.Tensor,
    top_k: int,
    allowed_languages=None,
    unknown_threshold: float = 0.0,
):
    predictions = _original_predict(segment, top_k, allowed_languages, unknown_threshold)
    if predictions and predictions[0].language_code == "en":
        accent_result = _run_accent_head(segment)
        if accent_result is not None:
            accent_code, accent_score = accent_result
            if accent_score >= ACCENT_HEAD_MIN_SCORE:
                predictions[0] = _base.LanguagePrediction(
                    language_code=accent_code,
                    scores=float(round(accent_score, 4)),
                    rank=1,
                )
    return predictions


_base._predict_segment_languages = _predict_with_accent

# ── Export the (now-patched) FastAPI app ─────────────────────────────────────
app = _base.app


@app.on_event("startup")
async def _preload_accent_head() -> None:
    """Download and cache the CommonAccent model at startup so the first request is fast."""
    import logging
    logger = logging.getLogger("api_endpoint_accent")
    if ACCENT_HEAD_MODEL:
        logger.info("Loading CommonAccent model: %s", ACCENT_HEAD_MODEL)
        _get_accent_head()
        logger.info("CommonAccent model ready.")
    else:
        logger.info("ACCENT_HEAD_MODEL is empty — accent head disabled.")


@app.get("/health")
def health() -> dict:
    """Return the load status of both the LID model and the accent head."""
    lid_loaded = _base._get_lid_model.cache_info().currsize > 0
    accent_loaded = _get_accent_head.cache_info().currsize > 0
    return {
        "status": "ok",
        "lid_model_loaded": lid_loaded,
        "accent_head_loaded": accent_loaded,
        "accent_head_model": ACCENT_HEAD_MODEL or "(disabled)",
    }
