"""
Analysis endpoints for prosody-driven KPI prediction.

Audio → prosodic features → KPI predictions.  No emotion labels.

Clients define their KPIs on the Next.js dashboard.  This API reads those
definitions and predicts outcomes based on raw prosodic signals.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import base64
import os
import uuid

logger = logging.getLogger(__name__)

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl

from config import settings
from db import create_session
from model_client import get_model_client, ModelPrediction
from kpis import get_kpi_loader
from kpi_predictor import get_kpi_predictor, ProsodySignals
from schemas import (
    AnalysisResponse,
    AnalysisRequest,
    ProsodyFeatures,
    ProsodyMarkers,
    KPIPredictionResult,
    KPIAlertResult,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# GCS prediction logging
# ---------------------------------------------------------------------------
FEEDBACK_BUCKET = os.getenv("FEEDBACK_BUCKET", "prosodyssm-prosody-datasets")
FEEDBACK_PREFIX = "feedback"

try:
    from google.cloud import storage as gcs_storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def _hash_api_key(api_key: str) -> str:
    """Return SHA-256 hex digest of an API key."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def _resolve_session_id(
    session_id: Optional[str],
    api_key_hash: Optional[str],
) -> Optional[str]:
    """If session_id is missing but we have api_key_hash, create a session and return its id."""
    if session_id:
        return session_id
    if not api_key_hash:
        return None
    loader = get_kpi_loader()
    org_id = await loader.get_organization_id(api_key_hash)
    if not org_id:
        return None
    return await create_session(org_id, metadata=None)


def _log_prediction(
    prediction_id: str,
    session_id: str | None,
    prediction: ModelPrediction,
    prosody_signals: dict,
    kpi_predictions: list[dict] | None,
    api_key_hash: str | None = None,
):
    """Write prediction record to GCS (or local fallback) as JSONL."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    record = json.dumps(
        {
            "prediction_id": prediction_id,
            "session_id": session_id,
            "api_key_hash": api_key_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prosody_signals": prosody_signals,
            "kpi_predictions": kpi_predictions,
            "text": prediction.text,
            "duration": prediction.duration,
        },
        default=str,
    ) + "\n"

    if GCS_AVAILABLE and os.getenv("USE_GCS_FEEDBACK", "true").lower() == "true":
        try:
            client = gcs_storage.Client()
            bucket = client.bucket(FEEDBACK_BUCKET)
            blob_path = f"{FEEDBACK_PREFIX}/predictions/{date_str}.jsonl"
            blob = bucket.blob(blob_path)

            existing = ""
            if blob.exists():
                existing = blob.download_as_text()

            blob.upload_from_string(existing + record)
            return
        except Exception as e:
            logger.warning(f"GCS write failed, falling back to local: {e}")

    # Local fallback
    local_dir = os.path.join("feedback_data", "predictions")
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, f"{date_str}.jsonl"), "a") as f:
        f.write(record)


async def _build_response(
    prediction: ModelPrediction,
    prediction_id: str,
    api_key_hash: str | None,
    session_id: str | None = None,
) -> AnalysisResponse:
    """Build API response from model prediction with KPI predictions."""

    # Extract prosodic signals (no emotion labels)
    signals = ProsodySignals.from_model_prediction(prediction)
    features = prediction.prosody_features or {}

    prosody = ProsodyFeatures(
        valence=prediction.valence,
        arousal=prediction.arousal,
        dominance=prediction.dominance,
        prosody=ProsodyMarkers(
            pitch_trend=prediction.pitch_trend,
            intensity=prediction.intensity,
            tempo=prediction.tempo,
        ) if any([prediction.pitch_trend, prediction.intensity, prediction.tempo]) else None,
        pitch_mean=features.get("f0_mean"),
        pitch_std=features.get("f0_std"),
        pitch_range=features.get("f0_range"),
        energy_mean=features.get("energy_mean"),
        energy_std=features.get("energy_std"),
        speech_rate=features.get("speech_rate"),
        jitter=features.get("jitter"),
        shimmer=features.get("shimmer"),
        hnr=features.get("hnr"),
        spectral_centroid=features.get("spectral_centroid_mean"),
        spectral_rolloff=features.get("spectral_rolloff_mean"),
    )

    # Base response
    response = AnalysisResponse(
        prediction_id=prediction_id,
        session_id=session_id,
        text=prediction.text,
        prosody=prosody,
        duration=prediction.duration,
        word_count=prediction.word_count,
    )

    # KPI predictions (if client has KPIs configured)
    if api_key_hash:
        try:
            loader = get_kpi_loader()
            kpis = await loader.get_kpis_for_api_key(api_key_hash)

            if kpis:
                predictor = get_kpi_predictor()
                predictions = predictor.predict(signals, kpis)

                response.kpi_predictions = [
                    KPIPredictionResult(**p.to_dict()) for p in predictions
                ]

                # Collect alerts
                alerts = [
                    KPIAlertResult(
                        threshold=p.alert.threshold,
                        direction=p.alert.direction,
                        message=p.alert.message,
                    )
                    for p in predictions
                    if p.alert is not None
                ]
                if alerts:
                    response.alerts = alerts

        except Exception as e:
            # Don't fail the whole request if KPI prediction fails
            logger.warning(f"KPI prediction failed (returning prosody only): {e}")

    return response


@router.post(
    "/audio",
    response_model=AnalysisResponse,
    summary="Analyze audio file",
    description="Upload audio for prosody analysis and KPI outcome prediction.",
)
async def analyze_audio_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC, OGG)"),
    language: str = Form(default="en", description="Language code for ASR"),
    session_id: Optional[str] = Form(default=None, description="Session ID for conversation tracking"),
):
    """
    Analyze an uploaded audio file.

    The audio is sent to the model service which:
    1. Transcribes the audio using Whisper ASR
    2. Extracts prosodic features (pitch, energy, rhythm, voice quality)
    3. Returns VAD (valence-arousal-dominance) scores

    The API then predicts outcomes for the client's configured KPIs
    based on the prosodic signals and returns recommended actions.
    """
    api_key = request.headers.get("X-API-Key", "")
    api_key_hash = _hash_api_key(api_key) if api_key else None
    session_id = await _resolve_session_id(session_id, api_key_hash) or session_id

    # Validate file size
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size // (1024*1024)}MB",
        )

    # Save to temp file
    os.makedirs(settings.temp_dir, exist_ok=True)
    temp_path = os.path.join(
        settings.temp_dir,
        f"{uuid.uuid4()}{os.path.splitext(file.filename or '.wav')[1]}"
    )

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        # Call model service
        model_client = get_model_client()
        prediction = await model_client.predict_from_file(temp_path, language)

        # Build response with KPI predictions
        prediction_id = str(uuid.uuid4())
        response = await _build_response(
            prediction, prediction_id, api_key_hash, session_id,
        )

        # Log prediction asynchronously
        background_tasks.add_task(
            _log_prediction,
            prediction_id=prediction_id,
            session_id=session_id,
            prediction=prediction,
            prosody_signals=response.prosody.model_dump(),
            kpi_predictions=[p.model_dump() for p in response.kpi_predictions] if response.kpi_predictions else None,
            api_key_hash=api_key_hash,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Model inference failed during audio file analysis")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again or contact support.",
        )

    finally:
        background_tasks.add_task(cleanup_temp_file, temp_path)


@router.post(
    "/base64",
    response_model=AnalysisResponse,
    summary="Analyze base64-encoded audio",
    description="Submit base64-encoded audio for prosody analysis and KPI prediction.",
)
async def analyze_base64_audio(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    raw_request: Request = None,
):
    """
    Analyze base64-encoded audio data.

    Useful for browser-based recording or when file upload isn't practical.
    """
    api_key = raw_request.headers.get("X-API-Key", "") if raw_request else ""
    api_key_hash = _hash_api_key(api_key) if api_key else None
    session_id = await _resolve_session_id(request.session_id, api_key_hash) or request.session_id

    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 is required")

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    if len(audio_bytes) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large. Maximum size is {settings.max_file_size // (1024*1024)}MB",
        )

    try:
        model_client = get_model_client()
        prediction = await model_client.predict_from_base64(
            request.audio_base64, request.language,
        )

        prediction_id = str(uuid.uuid4())
        response = await _build_response(
            prediction, prediction_id, api_key_hash, request.session_id,
        )

        background_tasks.add_task(
            _log_prediction,
            prediction_id=prediction_id,
            session_id=request.session_id,
            prediction=prediction,
            prosody_signals=response.prosody.model_dump(),
            kpi_predictions=[p.model_dump() for p in response.kpi_predictions] if response.kpi_predictions else None,
            api_key_hash=api_key_hash,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Model inference failed during base64 audio analysis")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again or contact support.",
        )


@router.post(
    "/url",
    response_model=AnalysisResponse,
    summary="Analyze audio from URL",
    description="Fetch and analyze audio from a URL.",
)
async def analyze_audio_url(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    raw_request: Request = None,
):
    """Analyze audio from a publicly accessible URL."""
    api_key = raw_request.headers.get("X-API-Key", "") if raw_request else ""
    api_key_hash = _hash_api_key(api_key) if api_key else None
    session_id = await _resolve_session_id(request.session_id, api_key_hash) or request.session_id

    if not request.audio_url:
        raise HTTPException(status_code=400, detail="audio_url is required")

    import httpx

    os.makedirs(settings.temp_dir, exist_ok=True)
    temp_path = os.path.join(settings.temp_dir, f"{uuid.uuid4()}.wav")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(str(request.audio_url))
            resp.raise_for_status()

            content = resp.content
            if len(content) > settings.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio too large. Maximum size is {settings.max_file_size // (1024*1024)}MB",
                )

            with open(temp_path, "wb") as f:
                f.write(content)

        model_client = get_model_client()
        prediction = await model_client.predict_from_file(temp_path, request.language)

        prediction_id = str(uuid.uuid4())
        response = await _build_response(
            prediction, prediction_id, api_key_hash, session_id,
        )

        background_tasks.add_task(
            _log_prediction,
            prediction_id=prediction_id,
            session_id=session_id,
            prediction=prediction,
            prosody_signals=response.prosody.model_dump(),
            kpi_predictions=[p.model_dump() for p in response.kpi_predictions] if response.kpi_predictions else None,
            api_key_hash=api_key_hash,
        )

        return response

    except httpx.HTTPError:
        logger.exception("Failed to download audio from URL")
        raise HTTPException(
            status_code=400,
            detail="Failed to download audio from the provided URL.",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Model inference failed during URL audio analysis")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again or contact support.",
        )
    finally:
        background_tasks.add_task(cleanup_temp_file, temp_path)


@router.post(
    "/gcs",
    response_model=AnalysisResponse,
    summary="Analyze audio from GCS",
    description="Analyze audio directly from Google Cloud Storage.",
)
async def analyze_gcs_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    gcs_uri: str = Form(..., description="GCS URI (gs://bucket/path/to/audio.wav)"),
    language: str = Form(default="en", description="Language code for ASR"),
    session_id: Optional[str] = Form(default=None, description="Session ID for conversation tracking"),
):
    """Analyze audio directly from GCS (efficient for large files)."""
    api_key = request.headers.get("X-API-Key", "")
    api_key_hash = _hash_api_key(api_key) if api_key else None
    session_id = await _resolve_session_id(session_id, api_key_hash) or session_id

    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="Invalid GCS URI. Must start with gs://")

    try:
        model_client = get_model_client()
        prediction = await model_client.predict_from_gcs(gcs_uri, language)

        prediction_id = str(uuid.uuid4())
        response = await _build_response(
            prediction, prediction_id, api_key_hash, session_id,
        )

        background_tasks.add_task(
            _log_prediction,
            prediction_id=prediction_id,
            session_id=session_id,
            prediction=prediction,
            prosody_signals=response.prosody.model_dump(),
            kpi_predictions=[p.model_dump() for p in response.kpi_predictions] if response.kpi_predictions else None,
            api_key_hash=api_key_hash,
        )

        return response

    except HTTPException:
        raise
    except Exception:
        logger.exception("Model inference failed during GCS audio analysis")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again or contact support.",
        )


def cleanup_temp_file(path: str):
    """Remove temporary file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
