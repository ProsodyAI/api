"""
Speaker detection for agent vs caller or multiple speakers on one audio stream.

Uses resemblyzer (optional) for speaker embeddings. If not installed, speaker_id
remains "unknown". Supports:
- Agent vs caller: set agent_embedding in session (from reference audio); chunks
  are labeled "agent" or "caller" by cosine similarity.
- Multiple speakers: no enrollment; running centroid clustering labels
  "speaker_0", "speaker_1", ... by order of first appearance.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded encoder (resemblyzer optional)
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from resemblyzer import VoiceEncoder  # type: ignore[import-untyped]

        _encoder = VoiceEncoder(verbose=False)
        return _encoder
    except ImportError:
        logger.debug("resemblyzer not installed; speaker detection disabled")
        return None


def wav_bytes_to_float(wav_bytes: bytes, target_sr: int = 16000) -> Optional[tuple[np.ndarray, int]]:
    """Decode WAV bytes to float32 mono. Returns (samples, sr) or None."""
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data, sr
    except Exception as e:
        logger.debug("wav_bytes_to_float failed: %s", e)
        return None


def get_embedding(wav_bytes: bytes) -> Optional[np.ndarray]:
    """
    Compute 256-d speaker embedding from WAV bytes (16 kHz mono preferred).
    Returns None if resemblyzer is not installed or audio is too short.
    """
    enc = _get_encoder()
    if enc is None:
        return None
    out = wav_bytes_to_float(wav_bytes)
    if out is None:
        return None
    wav, sr = out
    # resemblyzer expects ~1.6s+ for partials; short clips still get one embedding
    if len(wav) < 8000:  # 0.5s at 16k
        return None
    try:
        return enc.embed_utterance(wav, return_partials=False)
    except Exception as e:
        logger.debug("embed_utterance failed: %s", e)
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] for L2-normed embeddings."""
    return float(np.dot(a, b))


def assign_speaker(
    embedding: np.ndarray,
    agent_embedding: Optional[np.ndarray] = None,
    agent_threshold: float = 0.60,
    centroids: Optional[list[tuple[str, np.ndarray]]] = None,
    centroid_ema_alpha: float = 0.3,
    max_speakers: int = 4,
) -> tuple[str, Optional[list[tuple[str, np.ndarray]]]]:
    """
    Label the chunk as agent, caller, or speaker_0, speaker_1, ...

    - If agent_embedding is set: return "agent" if cos_sim(embedding, agent_embedding) >= agent_threshold else "caller".
    - Else: use running centroids (list of (label, centroid)). Assign to nearest; if distance to all > threshold, add new centroid. Update assigned centroid with EMA. Returns (speaker_id, updated_centroids).
    """
    if agent_embedding is not None:
        sim = cosine_similarity(embedding, agent_embedding)
        return ("agent" if sim >= agent_threshold else "caller", centroids)

    # Multi-speaker: assign to nearest centroid or create new one
    centroids = list(centroids) if centroids else []
    best_idx = -1
    best_sim = -1.0
    for i, (_label, cent) in enumerate(centroids):
        sim = cosine_similarity(embedding, cent)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    # New speaker if no centroid or similarity below threshold
    if best_idx < 0 or best_sim < agent_threshold:
        if len(centroids) < max_speakers:
            new_label = f"speaker_{len(centroids)}"
            centroids.append((new_label, embedding.copy()))
            return new_label, centroids
        if best_idx < 0:
            return "unknown", centroids
        # Fallback: assign to nearest centroid even if below threshold

    # Update centroid with EMA
    label, old_cent = centroids[best_idx]
    new_cent = centroid_ema_alpha * embedding + (1.0 - centroid_ema_alpha) * old_cent
    new_cent = new_cent / np.linalg.norm(new_cent)
    centroids[best_idx] = (label, new_cent)
    return label, centroids
