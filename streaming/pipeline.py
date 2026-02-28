"""
ProsodySSM streaming pipeline.

Accumulates PCM audio, sends 1-second chunks to the deployed model for
inference, returns VAD scores + emotion + confidence as directives.
"""

import asyncio
import base64
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from kpi_predictor import ProsodySignals

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 5
CHUNK_BYTES = 16000 * 2 * CHUNK_SECONDS  # 5 seconds at 16kHz mono int16
INFERENCE_TIMEOUT = 10.0
VAD_RMS_THRESHOLD = 300


@dataclass
class AgentDirective:
    """Output from a single inference pass."""
    session_id: str
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    emotion: str = "neutral"
    confidence: float = 0.0
    text: str = ""
    frames_processed: int = 0
    timestamp_ms: int = 0


@dataclass
class PipelineSession:
    """Per-session state for the pipeline."""
    session_id: str
    audio_buffer: bytearray = field(default_factory=bytearray)
    all_audio: bytearray = field(default_factory=bytearray)
    prosody_history: list[ProsodySignals] = field(default_factory=list)
    frames_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_directive: Optional[AgentDirective] = None


def _pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    n = len(pcm)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + n, b"WAVE", b"fmt ", 16,
        1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", n,
    ) + pcm


def _rms(pcm: bytes) -> float:
    samples = struct.unpack(f'<{len(pcm)//2}h', pcm)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 if samples else 0


async def _transcribe_chunk(wav_bytes: bytes) -> str:
    """Transcribe a WAV chunk using OpenAI Whisper API."""
    import os
    import httpx

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        from dotenv import dotenv_values
        vals = dotenv_values(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        api_key = vals.get("OPENAI_API_KEY", "")
    if not api_key:
        return ""

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("chunk.wav", wav_bytes, "audio/wav")},
            data={"model": "whisper-1", "response_format": "text"},
        )
        resp.raise_for_status()
        return resp.text.strip()


class ProsodicPipeline:
    """
    Streaming pipeline that sends audio chunks to the Baseten ProsodySSM model.

    Usage:
        pipeline = BasetenPipeline()
        directive = await pipeline.process_audio(session_id, pcm_bytes)
        if directive:
            # send to client
    """

    def __init__(self):
        self._sessions: dict[str, PipelineSession] = {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            from model_client import get_model_client
            self._client = get_model_client()
        return self._client

    def _get_session(self, session_id: str) -> PipelineSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = PipelineSession(session_id=session_id)
        return self._sessions[session_id]

    async def process_audio(self, session_id: str, pcm_data: bytes) -> Optional[AgentDirective]:
        """
        Feed PCM audio into the pipeline. Returns an AgentDirective when
        enough audio has accumulated (1s) and the Baseten model returns a result.
        Returns None if still accumulating or audio is silence.
        """
        session = self._get_session(session_id)
        session.audio_buffer.extend(pcm_data)
        session.all_audio.extend(pcm_data)

        if len(session.audio_buffer) < CHUNK_BYTES:
            return None

        chunk = bytes(session.audio_buffer[:CHUNK_BYTES])
        session.audio_buffer = session.audio_buffer[CHUNK_BYTES:]

        rms = _rms(chunk)
        if rms < VAD_RMS_THRESHOLD:
            return None

        try:
            client = self._get_client()
            wav = _pcm_to_wav(chunk)
            b64 = base64.b64encode(wav).decode("utf-8")

            print(f"CALLING MODEL: url={client.service_url} wav_bytes={len(wav)} b64_len={len(b64)} chunk_bytes={len(chunk)}", flush=True)

            pred = await asyncio.wait_for(
                client.predict_from_base64(b64, language="en"),
                timeout=INFERENCE_TIMEOUT,
            )
            print(f"MODEL RESULT: emotion={pred.emotion} conf={pred.confidence} v={pred.valence} a={pred.arousal} d={pred.dominance}", flush=True)

            try:
                pred.text = await _transcribe_chunk(wav)
            except Exception:
                pass

            session.frames_processed += 1
            elapsed_ms = int((time.time() - session.start_time) * 1000)

            signals = ProsodySignals(
                valence=pred.valence,
                arousal=pred.arousal,
                dominance=pred.dominance,
            )
            session.prosody_history.append(signals)

            directive = AgentDirective(
                session_id=session_id,
                valence=pred.valence,
                arousal=pred.arousal,
                dominance=pred.dominance,
                emotion=pred.emotion,
                confidence=pred.confidence,
                text=pred.text,
                frames_processed=session.frames_processed,
                timestamp_ms=elapsed_ms,
            )
            session.last_directive = directive

            logger.info(
                f"Session {session_id}: {pred.emotion} ({pred.confidence:.2f}) "
                f"v={pred.valence:.2f} a={pred.arousal:.2f} d={pred.dominance:.2f}"
            )
            return directive

        except Exception as e:
            print(f"MODEL FAILED: {e}", flush=True)
            return None

    def get_session(self, session_id: str) -> Optional[PipelineSession]:
        return self._sessions.get(session_id)

    def get_prosody_history(self, session_id: str) -> list[ProsodySignals]:
        session = self._sessions.get(session_id)
        return session.prosody_history if session else []

    def get_all_audio(self, session_id: str) -> bytes:
        session = self._sessions.get(session_id)
        return bytes(session.all_audio) if session else b""

    async def close_session(self, session_id: str):
        self._sessions.pop(session_id, None)


_pipeline: Optional[ProsodicPipeline] = None


def get_pipeline() -> ProsodicPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ProsodicPipeline()
    return _pipeline
