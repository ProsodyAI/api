"""
Feedback endpoints for outcome collection and prosody corrections.

These endpoints close the training loop:
1. /correction -- Human corrects a prosody prediction (VAD values)
2. /session_outcome -- Actual KPI outcomes after a conversation ends

KPI outcomes are written to the shared PostgreSQL database (KpiOutcome table)
so both the Next.js dashboard and the training pipeline can access them.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request

from schemas import (
    FeedbackCorrectionRequest,
    SessionOutcomeRequest,
)
from kpis import get_kpi_loader

logger = logging.getLogger(__name__)

router = APIRouter()

# GCS bucket for feedback data
FEEDBACK_BUCKET = os.getenv("FEEDBACK_BUCKET", "prosodyssm-prosody-datasets")
FEEDBACK_PREFIX = "feedback"

try:
    from google.cloud import storage as gcs_storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def _write_feedback(category: str, data: dict):
    """Write feedback record to GCS (or local fallback)."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    record = json.dumps(data, default=str) + "\n"

    if GCS_AVAILABLE and os.getenv("USE_GCS_FEEDBACK", "true").lower() == "true":
        try:
            client = gcs_storage.Client()
            bucket = client.bucket(FEEDBACK_BUCKET)
            blob_path = f"{FEEDBACK_PREFIX}/{category}/{date_str}.jsonl"
            blob = bucket.blob(blob_path)

            existing = ""
            if blob.exists():
                existing = blob.download_as_text()

            blob.upload_from_string(existing + record)
            return
        except Exception as e:
            logger.warning(f"GCS write failed, falling back to local: {e}")

    # Local fallback
    local_dir = os.path.join("feedback_data", category)
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, f"{date_str}.jsonl"), "a") as f:
        f.write(record)


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Write KPI outcomes to the shared database
# ---------------------------------------------------------------------------

_OUTCOME_INSERT_SQL = """
INSERT INTO "KpiOutcome" (id, "sessionId", "scalarValue", "booleanValue", "categoryValue", "kpiId", "createdAt")
VALUES ($1, $2, $3, $4, $5, $6, NOW())
"""


async def _write_kpi_outcomes_to_db(session_id: str, outcomes: list[dict]):
    """Write KPI outcomes to the shared PostgreSQL database."""
    import uuid

    try:
        loader = get_kpi_loader()
        pool = await loader._get_pool()

        async with pool.acquire() as conn:
            for outcome in outcomes:
                await conn.execute(
                    _OUTCOME_INSERT_SQL,
                    str(uuid.uuid4()),  # id
                    session_id,
                    outcome.get("scalar_value"),
                    outcome.get("boolean_value"),
                    outcome.get("category_value"),
                    outcome["kpi_id"],
                )
    except Exception as e:
        logger.error(f"Failed to write KPI outcomes to DB: {e}")
        # Fall through to JSONL fallback (handled by caller)
        raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/correction",
    summary="Submit prosody correction",
    description="Human corrects a prosody prediction (VAD values). High-quality training signal.",
)
async def submit_correction(
    body: FeedbackCorrectionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Submit a correction for a previous prediction.

    Corrects the VAD (valence, arousal, dominance) values.
    These corrections feed into the next training batch.
    """
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    api_key_hash = _hash_api_key(api_key)

    record = {
        "type": "correction",
        "prediction_id": body.prediction_id,
        "corrected_valence": body.corrected_valence,
        "corrected_arousal": body.corrected_arousal,
        "corrected_dominance": body.corrected_dominance,
        "notes": body.notes,
        "api_key_hash": api_key_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    background_tasks.add_task(_write_feedback, "corrections", record)

    return {"status": "accepted", "prediction_id": body.prediction_id}


@router.post(
    "/session_outcome",
    summary="Submit KPI outcomes for a conversation",
    description=(
        "Report actual KPI values after a conversation ends. "
        "This is the primary training signal for outcome prediction."
    ),
)
async def submit_session_outcome(
    body: SessionOutcomeRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Submit actual KPI outcomes for a conversation session.

    These ground-truth values train the system to learn which prosodic
    patterns predict which KPI outcomes for each client.

    Outcomes are written to:
    1. The shared PostgreSQL database (KpiOutcome table) — for the
       dashboard and training pipeline.
    2. JSONL logs (GCS or local) — for audit and batch processing.
    """
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    api_key_hash = _hash_api_key(api_key)

    # Build outcome records
    outcome_records = []
    for entry in body.outcomes:
        outcome_records.append({
            "kpi_id": entry.kpi_id,
            "scalar_value": entry.scalar_value,
            "boolean_value": entry.boolean_value,
            "category_value": entry.category_value,
        })

    # Write to database (primary)
    try:
        await _write_kpi_outcomes_to_db(body.session_id, outcome_records)
    except Exception as e:
        logger.warning(f"DB write failed, outcomes saved to JSONL only: {e}")

    # Write to JSONL logs (secondary, for audit)
    log_record = {
        "type": "session_outcome",
        "session_id": body.session_id,
        "outcomes": outcome_records,
        "notes": body.notes,
        "api_key_hash": api_key_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    background_tasks.add_task(_write_feedback, "session_outcomes", log_record)

    return {
        "status": "accepted",
        "session_id": body.session_id,
        "outcomes_recorded": len(body.outcomes),
    }
