"""
WebSocket endpoint for real-time prosodic feedback and KPI prediction.

Receives PCM audio frames, extracts prosodic features, predicts KPI
outcomes, and returns directives for voice agent adaptation.

No emotion labels — raw prosodic signals drive everything.

This is the API layer. The core logic lives in prosody_ssm.streaming.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Optional
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config import settings
from db import create_session
from middleware.auth import validate_api_key
from kpis import get_kpi_loader, KPIDefinition
from kpi_predictor import get_kpi_predictor, ProsodySignals

logger = logging.getLogger(__name__)

# Lazy-load pipeline components
_pipeline = None
_bus = None
_store = None


def _get_pipeline():
    """Lazy-initialize the ProsodicPipeline singleton."""
    global _pipeline, _bus, _store

    if _pipeline is not None:
        return _pipeline, _bus, _store

    from prosody_ssm.streaming.bus import WebSocketAudioBus
    from prosody_ssm.streaming.session import InMemorySessionStore
    from prosody_ssm.streaming.pipeline import ProsodicPipeline

    _bus = WebSocketAudioBus()
    _store = InMemorySessionStore()

    # Try to load model
    model = None
    try:
        import torch
        from prosody_ssm.model import ProsodySSMClassifier

        model = ProsodySSMClassifier()
        model.eval()
        logger.info("ProsodySSM model loaded for streaming pipeline")
    except Exception as e:
        logger.warning(f"Running pipeline in heuristic mode (no model): {e}")

    _pipeline = ProsodicPipeline(
        model=model,
        predictor=None,  # KPI predictions handled at API layer now
        bus=_bus,
        store=_store,
    )

    return _pipeline, _bus, _store


router = APIRouter()


class StreamConfig(BaseModel):
    """Configuration sent by client at session start."""
    session_id: Optional[str] = None
    sample_rate: int = 16000
    directive_interval_ms: int = 200


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    Real-time prosodic feedback loop via WebSocket.

    Protocol:
    1. Client connects and sends config JSON:
       {"type": "config", "session_id": "...", "sample_rate": 16000, "api_key": "..."}

    2. Client sends audio as binary (raw PCM int16, mono) or JSON:
       {"type": "audio", "audio": "<base64_pcm>"}

    3. Server pushes directives as JSON:
       {
         "type": "directive",
         "prosody": {"valence": -0.3, "arousal": 0.7, ...},
         "kpi_predictions": [...],
         "alerts": [...],
         "recommended_actions": [...]
       }

    4. Client sends end signal:
       {"type": "end"}
    """
    await websocket.accept()

    session_id = str(uuid.uuid4())
    pipeline, bus, store = _get_pipeline()

    from prosody_ssm.streaming.adapters.websocket import WebSocketAdapter
    adapter = WebSocketAdapter(bus)

    config = StreamConfig()
    api_key_hash: Optional[str] = None
    client_kpis: list[KPIDefinition] = []
    prosody_history: list[ProsodySignals] = []

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive(), timeout=30.0
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
                continue

            # Handle text messages (config, end)
            if "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "config":
                    config = StreamConfig(**{k: v for k, v in msg.items() if k != "type" and k != "api_key"})
                    if config.session_id:
                        session_id = config.session_id

                    # Auth
                    api_key = msg.get("api_key", "")
                    if api_key:
                        try:
                            validate_api_key(api_key)
                            api_key_hash = _hash_api_key(api_key)

                            # Load client KPIs and create DB session for this stream
                            try:
                                loader = get_kpi_loader()
                                client_kpis = await loader.get_kpis_for_api_key(api_key_hash)
                                org_id = await loader.get_organization_id(api_key_hash)
                                if org_id:
                                    db_session_id = await create_session(org_id, metadata=None)
                                    if db_session_id:
                                        session_id = db_session_id
                                logger.info(f"Session {session_id}: loaded {len(client_kpis)} KPIs")
                            except Exception as e:
                                logger.warning(f"Failed to load KPIs for session {session_id}: {e}")

                        except Exception:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Invalid API key",
                            })
                            await websocket.close(code=4001)
                            return

                    await websocket.send_json({
                        "type": "config_ack",
                        "session_id": session_id,
                        "kpis_loaded": len(client_kpis),
                    })
                    continue

                elif msg_type == "end":
                    await adapter.close_session(session_id)
                    await websocket.send_json({
                        "type": "session_end",
                        "session_id": session_id,
                        "frames_processed": len(prosody_history),
                    })
                    break

                elif msg_type == "audio":
                    await adapter.handle_message(session_id, data["text"])
                    continue

            # Handle binary messages (raw PCM)
            elif "bytes" in data and data["bytes"]:
                pcm_data = data["bytes"]

                output = await pipeline.process_frame(session_id, pcm_data)

                if output is not None:
                    directive = output.directive

                    # Build prosody signals from pipeline output
                    signals = ProsodySignals(
                        valence=directive.valence,
                        arousal=directive.arousal,
                        dominance=getattr(directive, "dominance", 0.5),
                    )
                    prosody_history.append(signals)

                    # Build directive response
                    response_data = {
                        "type": "directive",
                        "session_id": session_id,
                        # Prosodic signals (the core data)
                        "prosody": {
                            "valence": round(directive.valence, 3),
                            "arousal": round(directive.arousal, 3),
                            "dominance": round(getattr(directive, "dominance", 0.5), 3),
                        },
                        # TTS adaptation (prosody-driven, not emotion-driven)
                        "tts_speed": directive.tts_speed,
                        # Metadata
                        "confidence": round(directive.confidence, 3),
                        "frames_processed": directive.frames_processed,
                        "timestamp_ms": directive.timestamp_ms,
                    }

                    # KPI predictions
                    if client_kpis:
                        predictor = get_kpi_predictor()
                        kpi_preds = predictor.predict(
                            signals, client_kpis, prosody_history,
                        )

                        response_data["kpi_predictions"] = [
                            p.to_dict() for p in kpi_preds
                        ]

                        # Collect alerts
                        alerts = [
                            {
                                "kpi_name": p.alert.kpi_name,
                                "message": p.alert.message,
                            }
                            for p in kpi_preds
                            if p.alert is not None
                        ]
                        if alerts:
                            response_data["alerts"] = alerts

                        # Collect recommended actions across all KPIs
                        all_actions = []
                        for p in kpi_preds:
                            for a in p.recommended_actions:
                                all_actions.append({
                                    "kpi": p.kpi_name,
                                    "action": a.action,
                                    "expected_impact": round(a.expected_impact, 3),
                                })
                        if all_actions:
                            response_data["recommended_actions"] = all_actions

                    # LLM context (prosody-based, not emotion-based)
                    if hasattr(directive, "llm_context") and directive.llm_context:
                        response_data["llm_context"] = directive.llm_context

                    await websocket.send_json(response_data)

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Session {session_id} error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass
    finally:
        await pipeline.close_session(session_id)


@router.get("/health")
async def streaming_health():
    """Health check for streaming endpoint."""
    _, _, store = _get_pipeline()
    active = await store.active_count()
    return {
        "status": "healthy",
        "active_sessions": active,
    }
