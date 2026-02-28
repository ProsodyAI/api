"""Emotion labels used by the streaming pipeline (heuristic / model output)."""

from enum import Enum


class EmotionLabel(str, Enum):
    """Emotion labels used for prosody pipeline output."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CONTEMPT = "contempt"
