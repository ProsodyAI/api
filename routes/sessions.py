"""
Session, transcript, and audio endpoints for the multitenant DB.

All endpoints require X-API-Key. Session and transcript/audio writes are scoped
to the organization resolved from the API key (tenant isolation).
"""

import hashlib
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from db import create_session, get_session_organization_id
from kpis import get_kpi_loader
from middleware.auth import get_api_key_header

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key_header)])


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


async def _get_org_id_from_request(request: Request) -> str:
    """Resolve organization ID from X-API-Key. Raises 403 if not found."""
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    loader = get_kpi_loader()
    org_id = await loader.get_organization_id(_hash_api_key(api_key))
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key or organization not found",
        )
    return org_id


async def _ensure_session_belongs_to_org(session_id: str, organization_id: str) -> None:
    """Raise 404 if session does not exist or does not belong to the organization."""
    session_org = await get_session_organization_id(session_id)
    if not session_org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or session table not available",
        )
    if session_org != organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateSessionResponse(BaseModel):
    session_id: str = Field(..., description="ID of the created session")


class TranscriptRequest(BaseModel):
    content: str = Field(..., description="Transcript text")
    language: Optional[str] = Field(None, description="Language code (e.g. en)")


class AudioRequest(BaseModel):
    storage_path: str = Field(..., description="Path or URL to stored audio (e.g. S3/GCS)")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    format: Optional[str] = Field(None, description="Audio format (e.g. wav, mp3)")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=CreateSessionResponse,
    summary="Create a session",
    description="Create a new conversation session for the current tenant. Use the returned session_id in analysis and feedback.",
)
async def post_create_session(request: Request):
    """Create a session. Requires X-API-Key. Returns session_id for use in /analyze, /feedback, and transcript/audio."""
    org_id = await _get_org_id_from_request(request)
    session_id = await create_session(org_id, metadata=None)
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session creation unavailable (database or Session table not configured)",
        )
    return CreateSessionResponse(session_id=session_id)


@router.post(
    "/{session_id}/transcript",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Add transcript for a session",
)
async def post_transcript(
    session_id: str,
    body: TranscriptRequest,
    request: Request,
):
    """Store transcript text for a session. Session must belong to the caller's organization."""
    org_id = await _get_org_id_from_request(request)
    await _ensure_session_belongs_to_org(session_id, org_id)

    loader = get_kpi_loader()
    pool = await loader._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO "ConversationTranscript" ("sessionId", "organizationId", content, language, "createdAt")
            VALUES ($1, $2, $3, $4, NOW())
            """,
            session_id,
            org_id,
            body.content,
            body.language,
        )


@router.post(
    "/{session_id}/audio",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Register audio metadata for a session",
)
async def post_audio(
    session_id: str,
    body: AudioRequest,
    request: Request,
):
    """Register audio metadata (storage path, duration, format) for a session. Blobs go in object storage; this stores the reference."""
    org_id = await _get_org_id_from_request(request)
    await _ensure_session_belongs_to_org(session_id, org_id)

    loader = get_kpi_loader()
    pool = await loader._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO "ConversationAudio" ("sessionId", "organizationId", "storagePath", "durationSeconds", format, "createdAt")
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            session_id,
            org_id,
            body.storage_path,
            body.duration_seconds,
            body.format,
        )
