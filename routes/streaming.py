"""
WebSocket endpoint for real-time ProsodySSM inference.

Audio in (PCM) -> model inference -> directives out (VAD, emotion, KPIs).
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config import settings
from db import create_session
from middleware.auth import validate_api_key
from kpis import get_kpi_loader, KPIDefinition
from kpi_predictor import get_kpi_predictor, ProsodySignals
from storage import upload_audio, upload_transcript, get_org_slug
from streaming.pipeline import get_pipeline
from streaming.session import InMemorySessionStore, SessionState

logger = logging.getLogger(__name__)

STREAMING_SAMPLE_RATE = 16000

router = APIRouter()

# Session store singleton
_store: Optional[InMemorySessionStore] = None


def _get_store() -> InMemorySessionStore:
    global _store
    if _store is None:
        _store = InMemorySessionStore()
    return _store


# Observer registry
_observers: dict[str, list[WebSocket]] = {}


def _add_observer(session_id: str, ws: WebSocket):
    _observers.setdefault(session_id, []).append(ws)


def _remove_observer(session_id: str, ws: WebSocket):
    if session_id in _observers:
        _observers[session_id] = [w for w in _observers[session_id] if w is not ws]
        if not _observers[session_id]:
            del _observers[session_id]


async def _broadcast(session_id: str, data: dict):
    if session_id not in _observers:
        return
    dead = []
    for ws in _observers[session_id]:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _remove_observer(session_id, ws)


def _flush_to_gcs(org_slug, org_bucket, session_id, pipeline):
    """Store session audio + transcript to GCS."""
    if not org_slug:
        return
    audio = pipeline.get_all_audio(session_id)
    history = pipeline.get_prosody_history(session_id)
    if not audio:
        return
    try:
        from streaming.pipeline import _pcm_to_wav
        wav = _pcm_to_wav(audio)
        upload_audio(org_slug, session_id, wav, org_storage_bucket=org_bucket)

        session = pipeline.get_session(session_id)
        transcript = {
            "session_id": session_id,
            "frames": len(history),
            "duration_seconds": round(len(audio) / (STREAMING_SAMPLE_RATE * 2), 1),
            "prosody_summary": {
                "avg_valence": round(sum(s.valence for s in history) / len(history), 3) if history else 0,
                "avg_arousal": round(sum(s.arousal for s in history) / len(history), 3) if history else 0,
                "avg_dominance": round(sum(s.dominance for s in history) / len(history), 3) if history else 0,
            },
        }
        if session and session.last_directive:
            transcript["last_emotion"] = session.last_directive.emotion
            transcript["last_confidence"] = session.last_directive.confidence
            transcript["text"] = session.last_directive.text

        upload_transcript(org_slug, session_id, transcript, org_storage_bucket=org_bucket)
        logger.info(f"Session {session_id}: flushed to GCS ({org_slug})")
    except Exception as e:
        logger.error(f"Session {session_id}: GCS flush failed: {e}")


class StreamConfig(BaseModel):
    session_id: Optional[str] = None
    sample_rate: int = 16000


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    store = _get_store()
    pipeline = get_pipeline()

    api_key_hash: Optional[str] = None
    client_kpis: list[KPIDefinition] = []
    org_slug: Optional[str] = None
    org_bucket: Optional[str] = None

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
                continue

            

            if "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "config":
                    if msg.get("session_id"):
                        session_id = msg["session_id"]

                    api_key = msg.get("api_key", "")
                    if api_key:
                        try:
                            validate_api_key(api_key)
                            api_key_hash = _hash_key(api_key)
                            try:
                                loader = get_kpi_loader()
                                client_kpis = await loader.get_kpis_for_api_key(api_key_hash)
                                org_id = await loader.get_organization_id(api_key_hash)
                                if org_id:
                                    db_sid = await create_session(org_id)
                                    if db_sid:
                                        session_id = db_sid
                                result = await get_org_slug(api_key_hash)
                                if result and result[0]:
                                    org_slug, org_bucket = result
                                ss = SessionState(session_id=session_id, org_id=org_id, org_slug=org_slug)
                                await store.set(ss)
                            except Exception as e:
                                logger.warning(f"Session {session_id}: KPI/org load failed: {e}")
                        except Exception:
                            await websocket.send_json({"type": "error", "message": "Invalid API key"})
                            await websocket.close(code=4001)
                            return

                    await websocket.send_json({
                        "type": "config_ack",
                        "session_id": session_id,
                        "kpis_loaded": len(client_kpis),
                    })
                    continue

                elif msg_type == "end":
                    _flush_to_gcs(org_slug, org_bucket, session_id, pipeline)
                    session = pipeline.get_session(session_id)
                    await websocket.send_json({
                        "type": "session_end",
                        "session_id": session_id,
                        "frames_processed": session.frames_processed if session else 0,
                    })
                    break

            else:
                raw = data.get("bytes") or data.get("data")
                if raw:
                    directive = await pipeline.process_audio(session_id, raw)
                else:
                    continue
                if directive is None:
                    continue

                ss = await store.get(session_id)
                if ss:
                    ss.last_emotion = directive.emotion
                    ss.last_text = directive.text
                    ss.frames_processed = directive.frames_processed
                    await store.set(ss)

                response = {
                    "type": "directive",
                    "session_id": session_id,
                    "prosody": {
                        "valence": round(directive.valence, 3),
                        "arousal": round(directive.arousal, 3),
                        "dominance": round(directive.dominance, 3),
                    },
                    "emotion": directive.emotion,
                    "confidence": round(directive.confidence, 3),
                    "text": directive.text,
                    "frames_processed": directive.frames_processed,
                    "timestamp_ms": directive.timestamp_ms,
                }

                if client_kpis:
                    signals = ProsodySignals(valence=directive.valence, arousal=directive.arousal, dominance=directive.dominance)
                    history = pipeline.get_prosody_history(session_id)
                    preds = get_kpi_predictor().predict(signals, client_kpis, history)
                    response["kpi_predictions"] = [p.to_dict() for p in preds]
                    alerts = [{"kpi_name": p.alert.kpi_name, "message": p.alert.message} for p in preds if p.alert]
                    if alerts:
                        response["alerts"] = alerts
                    actions = [{"kpi": p.kpi_name, "action": a.action, "expected_impact": round(a.expected_impact, 3)} for p in preds for a in p.recommended_actions]
                    if actions:
                        response["recommended_actions"] = actions

                await websocket.send_json(response)
                await _broadcast(session_id, response)

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
        _flush_to_gcs(org_slug, org_bucket, session_id, pipeline)
    except Exception as e:
        logger.error(f"Session {session_id} error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await pipeline.close_session(session_id)
        await store.delete(session_id)


@router.get("/health")
async def streaming_health():
    store = _get_store()
    return {"status": "healthy", "active_sessions": await store.active_count()}


@router.get("/sessions")
async def list_active_sessions(org_id: Optional[str] = None):
    store = _get_store()
    sessions = await store.list_by_org(org_id) if org_id else await store.list_all()
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "org_id": s.org_id,
                "org_slug": s.org_slug,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "frames_processed": s.frames_processed,
                "last_emotion": s.last_emotion,
                "last_text": s.last_text,
                "duration_seconds": round(time.time() - s.created_at, 1),
            }
            for s in sessions
        ],
    }


@router.websocket("/observe/{session_id}")
async def observe_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _add_observer(session_id, websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                if "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _remove_observer(session_id, websocket)


@router.get("/history/{org_slug}")
async def list_session_history(org_slug: str, limit: int = 50):
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket(settings.org_bucket)
        blobs = list(bucket.list_blobs(prefix=f"{org_slug}/transcripts/", max_results=limit))
        return {
            "sessions": [
                {
                    "session_id": b.name.split("/")[-1].replace(".json", ""),
                    "stored_at": b.updated.isoformat() if b.updated else None,
                    "size_bytes": b.size,
                }
                for b in sorted(blobs, key=lambda b: b.updated or 0, reverse=True)
            ]
        }
    except Exception as e:
        return {"sessions": [], "error": str(e)}


@router.get("/history/{org_slug}/{session_id}")
async def get_session_transcript(org_slug: str, session_id: str):
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        blob = client.bucket(settings.org_bucket).blob(f"{org_slug}/transcripts/{session_id}.json")
        if not blob.exists():
            return {"error": "Not found"}
        return json.loads(blob.download_as_text())
    except Exception as e:
        return {"error": str(e)}
