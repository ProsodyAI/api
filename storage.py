"""
Org-scoped GCS storage for audio and transcripts.

Default bucket: gs://prosodyai-org-data/{org-slug}/audio|transcripts/
If an org has its own storageBucket configured, that is used instead.
"""

import json
import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage as gcs_storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def _get_client():
    if not GCS_AVAILABLE:
        raise RuntimeError("google-cloud-storage not installed")
    return gcs_storage.Client()


def _resolve_bucket(org_storage_bucket: Optional[str] = None) -> str:
    return org_storage_bucket or settings.org_bucket


async def get_org_slug(api_key_hash: str) -> Optional[str]:
    """Resolve API key hash to org slug via DB."""
    try:
        from kpis import get_kpi_loader
        loader = get_kpi_loader()
        pool = await loader._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT o.slug, o."storageBucket" FROM "ApiKey" ak '
                'JOIN "Organization" o ON o.id = ak."organizationId" '
                'WHERE ak."keyHash" = $1',
                api_key_hash,
            )
            if row:
                return row["slug"], row["storageBucket"]
    except Exception as e:
        logger.warning("Failed to resolve org slug: %s", e)
    return None, None


def upload_audio(
    org_slug: str,
    session_id: str,
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    org_storage_bucket: Optional[str] = None,
) -> str:
    """Upload audio to GCS. Returns the gs:// URI."""
    bucket_name = _resolve_bucket(org_storage_bucket)
    blob_path = f"{org_slug}/audio/{session_id}.wav"

    try:
        client = _get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(audio_bytes, content_type=content_type)
        uri = f"gs://{bucket_name}/{blob_path}"
        logger.info("Uploaded audio: %s", uri)
        return uri
    except Exception as e:
        logger.error("Failed to upload audio to GCS: %s", e)
        return ""


def upload_transcript(
    org_slug: str,
    session_id: str,
    result: dict,
    org_storage_bucket: Optional[str] = None,
) -> str:
    """Upload analysis result as JSON to GCS. Returns the gs:// URI."""
    bucket_name = _resolve_bucket(org_storage_bucket)
    blob_path = f"{org_slug}/transcripts/{session_id}.json"

    try:
        client = _get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(result, default=str),
            content_type="application/json",
        )
        uri = f"gs://{bucket_name}/{blob_path}"
        logger.info("Uploaded transcript: %s", uri)
        return uri
    except Exception as e:
        logger.error("Failed to upload transcript to GCS: %s", e)
        return ""
