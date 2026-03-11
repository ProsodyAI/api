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
from typing import Any, Optional

from kpi_predictor import ProsodySignals
from streaming.speaker_utils import assign_speaker, get_embedding

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 5
CHUNK_BYTES = 16000 * 2 * CHUNK_SECONDS  # 5 seconds at 16kHz mono int16
INFERENCE_TIMEOUT = 10.0
VAD_RMS_THRESHOLD = 300

# Temporal smoothing: EMA factor (higher = more weight on new frame, less smoothing)
SMOOTH_ALPHA = 0.35  # ~3–4 chunks to settle; tune down for stronger smoothing


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
    speaker_id: str = "unknown"  # "agent" | "caller" | "speaker_0" | "speaker_1" | ...


def _ema(alpha: float, raw: float, prev: Optional[float]) -> float:
    """Exponential moving average. Returns raw if prev is None."""
    if prev is None:
        return raw
    return alpha * raw + (1.0 - alpha) * prev


def _smooth_emotion_probs(
    alpha: float,
    raw_probs: dict[str, float],
    prev_probs: Optional[dict[str, float]],
) -> dict[str, float]:
    """EMA over emotion probability dict. Returns normalized smoothed probs."""
    all_labels = set(raw_probs) | (set(prev_probs) if prev_probs else set())
    if not all_labels:
        return dict(raw_probs)
    smoothed = {}
    for label in all_labels:
        r = raw_probs.get(label, 0.0)
        p = prev_probs.get(label, 0.0) if prev_probs else None
        smoothed[label] = _ema(alpha, r, p)
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {k: v / total for k, v in smoothed.items()}
    return smoothed


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
    # Temporal smoothing state (EMA of last output)
    _smooth_valence: Optional[float] = None
    _smooth_arousal: Optional[float] = None
    _smooth_dominance: Optional[float] = None
    _smooth_confidence: Optional[float] = None
    _smooth_emotion_probs: Optional[dict[str, float]] = None
    # Speaker detection: optional agent embedding for agent vs caller; else multi-speaker centroids
    agent_embedding: Any = None  # np.ndarray 256-d (set at session start for agent vs caller)
    _speaker_centroids: Any = None  # list of (label, np.ndarray) for multi-speaker clustering


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

            # Speaker detection: agent vs caller (if agent_embedding set) or multi-speaker clustering
            speaker_id = "unknown"
            emb = get_embedding(wav)
            if emb is not None:
                agent_emb = session.agent_embedding
                centroids = session._speaker_centroids
                speaker_id, updated_centroids = assign_speaker(
                    emb,
                    agent_embedding=agent_emb,
                    centroids=centroids,
                )
                session._speaker_centroids = updated_centroids

            # Temporal smoothing: EMA over VAD, confidence, and emotion probabilities
            smooth_probs = _smooth_emotion_probs(
                SMOOTH_ALPHA,
                pred.emotion_probabilities or {pred.emotion: pred.confidence},
                session._smooth_emotion_probs,
            )
            session._smooth_emotion_probs = smooth_probs
            smooth_valence = _ema(SMOOTH_ALPHA, pred.valence, session._smooth_valence)
            smooth_arousal = _ema(SMOOTH_ALPHA, pred.arousal, session._smooth_arousal)
            smooth_dominance = _ema(SMOOTH_ALPHA, pred.dominance, session._smooth_dominance)
            smooth_confidence = _ema(SMOOTH_ALPHA, pred.confidence, session._smooth_confidence)
            session._smooth_valence = smooth_valence
            session._smooth_arousal = smooth_arousal
            session._smooth_dominance = smooth_dominance
            session._smooth_confidence = smooth_confidence
            # Smoothed emotion = argmax of smoothed probabilities
            if smooth_probs:
                smooth_emotion = max(smooth_probs, key=smooth_probs.get)
            else:
                smooth_emotion = pred.emotion
                smooth_confidence = pred.confidence

            session.frames_processed += 1
            elapsed_ms = int((time.time() - session.start_time) * 1000)

            signals = ProsodySignals(
                valence=smooth_valence,
                arousal=smooth_arousal,
                dominance=smooth_dominance,
            )
            session.prosody_history.append(signals)

            directive = AgentDirective(
                session_id=session_id,
                valence=smooth_valence,
                arousal=smooth_arousal,
                dominance=smooth_dominance,
                emotion=smooth_emotion,
                confidence=smooth_confidence,
                text=pred.text,
                frames_processed=session.frames_processed,
                timestamp_ms=elapsed_ms,
                speaker_id=speaker_id,
            )
            session.last_directive = directive

            logger.info(
                f"Session {session_id}: {smooth_emotion} ({smooth_confidence:.2f}) speaker={speaker_id} "
                f"v={smooth_valence:.2f} a={smooth_arousal:.2f} d={smooth_dominance:.2f}"
            )
            return directive

        except Exception as e:
            print(f"MODEL FAILED: {e}", flush=True)
            return None

    def set_agent_embedding_from_audio(self, session_id: str, wav_bytes: bytes) -> bool:
        """Enroll agent voice from reference WAV (16 kHz mono). Enables agent vs caller labeling."""
        emb = get_embedding(wav_bytes)
        if emb is None:
            return False
        session = self._get_session(session_id)
        session.agent_embedding = emb
        return True

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
