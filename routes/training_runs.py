"""
Admin endpoints for listing and fetching training run metrics from GCS.

Training runs are stored at gs://prosodyssm-prosody-models/training_runs/{timestamp}/
with a training_metrics.json file produced by scripts/training/metrics.py.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from routes.admin import require_admin

logger = logging.getLogger(__name__)

router = APIRouter()

MODEL_BUCKET = "prosodyssm-prosody-models"
RUNS_PREFIX = "training_runs/"


def _get_gcs_client():
    from google.cloud import storage
    return storage.Client()


@router.get("/training-runs", dependencies=[Depends(require_admin)])
async def list_training_runs():
    """List all training runs from GCS, newest first."""
    try:
        client = _get_gcs_client()
        bucket = client.bucket(MODEL_BUCKET)

        runs: list[dict] = []
        seen_prefixes: set[str] = set()

        for blob in bucket.list_blobs(prefix=RUNS_PREFIX):
            parts = blob.name[len(RUNS_PREFIX):].split("/")
            if len(parts) < 2:
                continue
            run_id = parts[0]
            if run_id in seen_prefixes:
                if parts[1] == "training_metrics.json":
                    for r in runs:
                        if r["id"] == run_id:
                            r["has_metrics"] = True
                            r["metrics_size"] = blob.size
                continue
            seen_prefixes.add(run_id)

            is_metrics = parts[1] == "training_metrics.json"
            runs.append({
                "id": run_id,
                "has_metrics": is_metrics,
                "metrics_size": blob.size if is_metrics else None,
                "updated_at": blob.updated.isoformat() if blob.updated else None,
            })

        runs.sort(key=lambda r: r["id"], reverse=True)
        return {"runs": runs}

    except Exception as e:
        logger.error(f"Failed to list training runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-runs/{run_id}", dependencies=[Depends(require_admin)])
async def get_training_run(run_id: str):
    """Fetch training_metrics.json for a specific training run."""
    try:
        client = _get_gcs_client()
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob(f"{RUNS_PREFIX}{run_id}/training_metrics.json")

        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"No metrics found for run {run_id}")

        data = json.loads(blob.download_as_text())
        data["run_id"] = run_id
        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch training run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-runs/latest/metrics", dependencies=[Depends(require_admin)])
async def get_latest_metrics():
    """Fetch the latest training_metrics.json (models/training_metrics.json)."""
    try:
        client = _get_gcs_client()
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob("models/training_metrics.json")

        if not blob.exists():
            raise HTTPException(status_code=404, detail="No latest metrics found")

        data = json.loads(blob.download_as_text())
        data["run_id"] = "latest"
        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch latest metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
