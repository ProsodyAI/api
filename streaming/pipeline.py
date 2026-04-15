"""
ProsodySSM streaming pipeline.

Accumulates PCM audio, sends 1-second chunks to the deployed model for
inference, returns VAD scores + emotion + confidence as directives.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import struct
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
from kpi_predictor import ProsodySignals
from streaming.speaker_utils import assign_speaker, get_embedding

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 2
TARGET_SAMPLE_RATE = 16000
CHUNK_BYTES = TARGET_SAMPLE_RATE * 2 * CHUNK_SECONDS  # 2 seconds at 16kHz mono int16
INFERENCE_TIMEOUT = 10.0
VAD_RMS_THRESHOLD = 200
VAD_RMS_THRESHOLD_AFTER_NORM = 3000  # threshold after peak normalization
NORMALIZE_PEAK = 0.9  # target peak as fraction of int16 range

# Temporal smoothing: EMA factor (higher = more weight on new frame, less smoothing)
SMOOTH_ALPHA = 0.6  # Higher = more responsive; with 2s chunks we need less smoothing


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
    signals: dict[str, float] = field(default_factory=dict)
    sequence_signals: dict[str, float] = field(default_factory=dict)
    # Phoneme and prosodic embeddings (from same chunk)
    phonemes: list[str] = field(default_factory=list)
    ipa_transcript: str = ""  # Space-separated IPA phonemes (from phonemizer when available)
    prosody_embedding: Optional[dict] = None  # mfcc_means, f0_contour, energy_contour (downsampled)


@dataclass
class TranscriptSegment:
    """One chunk in the aligned transcript."""
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str
    emotion: str
    confidence: float
    valence: float
    arousal: float
    dominance: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class TranscriptTurn:
    """Consecutive segments from the same speaker, merged into a turn."""
    start_ms: int
    end_ms: int
    speaker_id: str
    text: str
    segments: list[TranscriptSegment]
    avg_valence: float
    avg_arousal: float
    avg_dominance: float
    dominant_emotion: str
    avg_confidence: float


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
    segments: list[TranscriptSegment] = field(default_factory=list)
    frames_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_directive: Optional[AgentDirective] = None
    # Audio format config (set from WebSocket config message)
    input_sample_rate: int = 16000
    input_encoding: str = "pcm16"  # "pcm16" | "mulaw" | "alaw"
    # Temporal smoothing state (EMA of last output)
    _smooth_valence: Optional[float] = None
    _smooth_arousal: Optional[float] = None
    _smooth_dominance: Optional[float] = None
    _smooth_confidence: Optional[float] = None
    _smooth_emotion_probs: Optional[dict[str, float]] = None
    _smooth_signals: Optional[dict[str, float]] = None
    # Speaker detection: optional agent embedding for agent vs caller; else multi-speaker centroids
    agent_embedding: Any = None  # np.ndarray 256-d (set at session start for agent vs caller)
    _speaker_centroids: Any = None  # list of (label, np.ndarray) for multi-speaker clustering
    # Previous transcript (used as Whisper prompt for continuity)
    last_transcript: str = ""


def _decode_mulaw(data: bytes) -> bytes:
    """Decode G.711 μ-law to 16-bit linear PCM."""
    MULAW_BIAS = 33
    out = bytearray(len(data) * 2)
    for i, byte in enumerate(data):
        byte = ~byte & 0xFF
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        sample = ((mantissa << 3) + MULAW_BIAS) << exponent
        sample -= MULAW_BIAS
        if sign:
            sample = -sample
        sample = max(-32768, min(32767, sample))
        struct.pack_into('<h', out, i * 2, sample)
    return bytes(out)


def _decode_alaw(data: bytes) -> bytes:
    """Decode G.711 A-law to 16-bit linear PCM."""
    out = bytearray(len(data) * 2)
    for i, byte in enumerate(data):
        byte ^= 0x55
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        if exponent == 0:
            sample = (mantissa << 4) + 8
        else:
            sample = ((mantissa << 4) + 0x108) << (exponent - 1)
        if sign:
            sample = -sample
        sample = max(-32768, min(32767, sample))
        struct.pack_into('<h', out, i * 2, sample)
    return bytes(out)


def _resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample 16-bit mono PCM via linear interpolation."""
    if src_rate == dst_rate:
        return pcm
    n_samples = len(pcm) // 2
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    ratio = dst_rate / src_rate
    n_out = int(n_samples * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, n_samples - 1)
    idx_floor = indices.astype(np.int32)
    idx_ceil = np.minimum(idx_floor + 1, n_samples - 1)
    frac = indices - idx_floor
    resampled = samples[idx_floor] * (1 - frac) + samples[idx_ceil] * frac
    return resampled.astype(np.int16).tobytes()


def _peak_normalize(pcm: bytes, target_peak: float = NORMALIZE_PEAK) -> bytes:
    """Peak-normalize 16-bit PCM so phone (quiet) and mic (loud) audio are comparable."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    peak = np.abs(samples).max()
    if peak < 1.0:
        return pcm
    target = target_peak * 32767
    scale = target / peak
    normalized = np.clip(samples * scale, -32768, 32767).astype(np.int16)
    return normalized.tobytes()


def preprocess_audio(pcm_data: bytes, sample_rate: int = 16000, encoding: str = "pcm16") -> bytes:
    """
    Preprocess incoming audio to 16kHz 16-bit linear PCM, peak-normalized.

    Args:
        pcm_data: raw audio bytes
        sample_rate: input sample rate (8000, 16000, etc.)
        encoding: "pcm16" (default), "mulaw", "alaw"

    Returns:
        16kHz 16-bit mono PCM bytes, peak-normalized
    """
    if encoding == "mulaw":
        pcm_data = _decode_mulaw(pcm_data)
    elif encoding == "alaw":
        pcm_data = _decode_alaw(pcm_data)

    if sample_rate != TARGET_SAMPLE_RATE:
        pcm_data = _resample_pcm(pcm_data, sample_rate, TARGET_SAMPLE_RATE)

    pcm_data = _peak_normalize(pcm_data)
    return pcm_data


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


TRANSCRIPT_PROMPT_MAX_CHARS = 224  # Whisper prompt limit (for context continuity)

# Max points in contour arrays sent to client (downsample to keep payload small)
PROSODY_CONTOUR_POINTS = 32


def _extract_phonemes_and_prosody(text: str, wav_bytes: bytes) -> tuple[list[str], str, Optional[dict]]:
    """Extract phonemes from text and prosodic embedding from audio. Returns (phonemes, ipa_transcript, prosody_embedding)."""
    phonemes: list[str] = []
    ipa_transcript = ""
    prosody_embedding: Optional[dict] = None
    try:
        if text and text.strip():
            from routes.features import get_phonetic_extractor
            ext = get_phonetic_extractor()
            feats = ext.extract_from_text(text.strip())
            phonemes = feats.phonemes or []
            ipa_transcript = " ".join(phonemes) if phonemes else ""
    except Exception:
        pass
    try:
        import io

        import soundfile as sf
        from routes.features import get_prosody_extractor

        audio, sr = sf.read(io.BytesIO(wav_bytes))
        ext = get_prosody_extractor()
        pf = ext.extract(audio, sr)
        # Build embedding dict: mfcc_means (13-d) + downsampled contours
        def downsample(arr: "np.ndarray", n: int) -> list[float]:
            if arr is None or len(arr) == 0:
                return []
            a = np.asarray(arr)
            if len(a) <= n:
                return a.tolist()
            indices = np.linspace(0, len(a) - 1, n, dtype=int)
            return a[indices].tolist()
        prosody_embedding = {
            "mfcc_means": pf.mfcc_means.tolist() if hasattr(pf.mfcc_means, "tolist") else list(pf.mfcc_means),
            "f0_contour": downsample(pf.f0_contour, PROSODY_CONTOUR_POINTS),
            "energy_contour": downsample(pf.energy_contour, PROSODY_CONTOUR_POINTS),
            "f0_mean": pf.f0_mean,
            "f0_std": pf.f0_std,
            "energy_mean": pf.energy_mean,
            "energy_std": pf.energy_std,
        }
    except Exception:
        pass
    return phonemes, ipa_transcript, prosody_embedding


async def _transcribe_chunk(wav_bytes: bytes, language: str = "en", prompt: Optional[str] = None) -> str:
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

    data: dict = {
        "model": "whisper-1",
        "response_format": "text",
        "language": language,
    }
    if prompt and prompt.strip():
        data["prompt"] = prompt.strip()[-TRANSCRIPT_PROMPT_MAX_CHARS:]

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("chunk.wav", wav_bytes, "audio/wav")},
            data=data,
        )
        resp.raise_for_status()
        return resp.text.strip()


def _merge_segments(group: list[TranscriptSegment]) -> TranscriptTurn:
    """Collapse a list of same-speaker segments into a single turn."""
    n = len(group)
    emotions = Counter(s.emotion for s in group)
    return TranscriptTurn(
        start_ms=group[0].start_ms,
        end_ms=group[-1].end_ms,
        speaker_id=group[0].speaker_id,
        text=" ".join(s.text for s in group if s.text).strip(),
        segments=list(group),
        avg_valence=round(sum(s.valence for s in group) / n, 3),
        avg_arousal=round(sum(s.arousal for s in group) / n, 3),
        avg_dominance=round(sum(s.dominance for s in group) / n, 3),
        dominant_emotion=emotions.most_common(1)[0][0],
        avg_confidence=round(sum(s.confidence for s in group) / n, 3),
    )


class ProsodicPipeline:
    """
    Streaming pipeline: audio chunks → model inference → directives.

    Usage:
        pipeline = ProsodicPipeline()
        directive = await pipeline.process_audio(session_id, pcm_bytes)
        if directive:
            # send to client
    """

    def __init__(self):
        self._sessions: dict[str, PipelineSession] = {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            from client import get_model_client
            self._client = get_model_client()
        return self._client

    def _get_session(self, session_id: str) -> PipelineSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = PipelineSession(session_id=session_id)
        return self._sessions[session_id]

    def configure_session(self, session_id: str, sample_rate: int = 16000, encoding: str = "pcm16"):
        """Set audio format for a session (call after WebSocket config)."""
        session = self._get_session(session_id)
        session.input_sample_rate = sample_rate
        session.input_encoding = encoding
        logger.info(f"Session {session_id}: audio config sr={sample_rate} enc={encoding}")

    async def process_audio(self, session_id: str, pcm_data: bytes) -> Optional[AgentDirective]:
        """
        Feed audio into the pipeline. Handles resampling, codec decoding,
        and peak normalization automatically based on session config.

        Returns an AgentDirective when enough audio has accumulated and
        the model returns a result. Returns None if still accumulating
        or audio is silence.
        """
        session = self._get_session(session_id)

        needs_preprocessing = (
            session.input_sample_rate != TARGET_SAMPLE_RATE
            or session.input_encoding != "pcm16"
        )
        if needs_preprocessing:
            pcm_data = preprocess_audio(pcm_data, session.input_sample_rate, session.input_encoding)
        else:
            pcm_data = _peak_normalize(pcm_data)

        session.audio_buffer.extend(pcm_data)
        session.all_audio.extend(pcm_data)

        if len(session.audio_buffer) < CHUNK_BYTES:
            return None

        chunk = bytes(session.audio_buffer[:CHUNK_BYTES])
        session.audio_buffer = session.audio_buffer[CHUNK_BYTES:]

        rms = _rms(chunk)
        if rms < VAD_RMS_THRESHOLD_AFTER_NORM:
            return None

        wav = _pcm_to_wav(chunk)
        b64 = base64.b64encode(wav).decode("utf-8")
        prompt = session.last_transcript[-TRANSCRIPT_PROMPT_MAX_CHARS:] if session.last_transcript else None

        # Run inference and transcription in parallel.
        # If the model fails (cold start, timeout, etc.) we still return a
        # directive with Whisper text and neutral defaults so the client
        # always gets a response for every voiced chunk.
        client = self._get_client()
        model_task = asyncio.ensure_future(asyncio.wait_for(
            client.predict_from_base64(b64, language="en"),
            timeout=INFERENCE_TIMEOUT,
        ))
        transcribe_task = asyncio.ensure_future(_transcribe_chunk(wav, language="en", prompt=prompt))

        pred: Optional["ModelPrediction"] = None
        try:
            pred = await model_task
        except Exception as e:
            logger.error(f"Session {session_id}: model inference failed: {e}")

        transcript_text = ""
        try:
            transcript_text = await asyncio.wait_for(transcribe_task, timeout=8.0)
        except Exception:
            transcribe_task.cancel()

        if pred is None:
            from client import ModelPrediction
            pred = ModelPrediction(
                text="", language="en", duration=0.0, word_count=0,
                emotion="neutral", confidence=0.0, emotion_probabilities={"neutral": 1.0},
                valence=0.0, arousal=0.5, dominance=0.5,
            )

        pred.text = transcript_text
        if pred.text:
            session.last_transcript = pred.text

        # Speaker detection
        speaker_id = "unknown"
        try:
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
        except Exception:
            pass

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
        if smooth_probs:
            smooth_emotion = max(smooth_probs, key=smooth_probs.get)
        else:
            smooth_emotion = pred.emotion
            smooth_confidence = pred.confidence

        # Smooth prosodic signals
        raw_signals = pred.signals or {}
        smooth_signals = {}
        prev_signals = session._smooth_signals or {}
        for key, val in raw_signals.items():
            smooth_signals[key] = _ema(SMOOTH_ALPHA, val, prev_signals.get(key))
        session._smooth_signals = smooth_signals

        session.frames_processed += 1
        elapsed_ms = int((time.time() - session.start_time) * 1000)

        phonemes, ipa_transcript, prosody_embedding = _extract_phonemes_and_prosody(pred.text or "", wav)

        prosody_signals = ProsodySignals(
            valence=smooth_valence,
            arousal=smooth_arousal,
            dominance=smooth_dominance,
        )
        session.prosody_history.append(prosody_signals)

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
            signals=smooth_signals,
            sequence_signals=pred.sequence_signals or {},
            phonemes=phonemes,
            ipa_transcript=ipa_transcript,
            prosody_embedding=prosody_embedding,
        )
        session.last_directive = directive

        chunk_start_ms = (session.frames_processed - 1) * CHUNK_SECONDS * 1000
        session.segments.append(TranscriptSegment(
            start_ms=chunk_start_ms,
            end_ms=chunk_start_ms + CHUNK_SECONDS * 1000,
            text=pred.text or "",
            speaker_id=speaker_id,
            emotion=smooth_emotion,
            confidence=round(smooth_confidence, 3),
            valence=round(smooth_valence, 3),
            arousal=round(smooth_arousal, 3),
            dominance=round(smooth_dominance, 3),
            signals={k: round(v, 3) for k, v in smooth_signals.items()},
        ))

        logger.info(
            f"Session {session_id}: {smooth_emotion} ({smooth_confidence:.2f}) speaker={speaker_id} "
            f"v={smooth_valence:.2f} a={smooth_arousal:.2f} d={smooth_dominance:.2f}"
        )
        return directive

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

    def get_aligned_transcript(self, session_id: str) -> list[TranscriptSegment]:
        session = self._sessions.get(session_id)
        return list(session.segments) if session else []

    def get_turns(self, session_id: str) -> list[TranscriptTurn]:
        """Merge consecutive same-speaker segments into turns."""
        segments = self.get_aligned_transcript(session_id)
        if not segments:
            return []
        turns: list[TranscriptTurn] = []
        group: list[TranscriptSegment] = [segments[0]]
        for seg in segments[1:]:
            if seg.speaker_id == group[-1].speaker_id:
                group.append(seg)
            else:
                turns.append(_merge_segments(group))
                group = [seg]
        turns.append(_merge_segments(group))
        return turns

    def get_session_transcript_dict(self, session_id: str, duration_seconds: float = 0.0) -> dict:
        """Full aligned transcript as a JSON-serializable dict."""
        return {
            "session_id": session_id,
            "duration_seconds": round(duration_seconds, 1),
            "turns": [asdict(t) for t in self.get_turns(session_id)],
            "segments": [asdict(s) for s in self.get_aligned_transcript(session_id)],
        }

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
