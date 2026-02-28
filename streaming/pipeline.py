"""
ProsodicPipeline: continuous prosodic feedback loop.

FrameExtractor -> (optional SSM model) -> AgentDirective.
Runs in heuristic mode when no model is loaded (no torch dependency).
"""

from dataclasses import dataclass
from typing import Optional, AsyncIterator
import logging

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .frame_extractor import FrameExtractor, ProsodyFrame
from .bus import AudioBus, AudioFrame
from .session import SessionStore, SessionState
from .emotions import EmotionLabel

logger = logging.getLogger(__name__)


@dataclass
class AgentDirective:
    """Real-time instruction to the voice agent."""

    tts_emotion: str = "neutral"
    tts_speed: float = 1.0
    llm_context: str = ""
    escalation_prob: float = 0.0
    churn_risk: float = 0.0
    resolution_prob: float = 0.5
    predicted_csat: float = 3.0
    sentiment_forecast: float = 0.0
    intervention_type: str = "none"
    intervention_urgency: float = 0.0
    current_emotion: str = "neutral"
    emotion_confidence: float = 0.0
    valence: float = 0.0
    arousal: float = 0.5
    confidence: float = 0.0
    timestamp_ms: float = 0.0
    frames_processed: int = 0


@dataclass
class PipelineOutput:
    """Full output from a pipeline processing cycle."""
    directive: AgentDirective
    prosody_frame: Optional[ProsodyFrame] = None
    emotion_probs: Optional[dict[str, float]] = None
    raw_valence: float = 0.0
    raw_arousal: float = 0.0


_TTS_EMOTION_MAP = {
    "empathetic": "sad",
    "calm": "neutral",
    "enthusiastic": "happy",
    "professional": "neutral",
    "reassuring": "neutral",
    "apologetic": "sad",
}

_TTS_SPEED_MAP = {
    "empathetic": 0.9,
    "calm": 0.85,
    "enthusiastic": 1.1,
    "professional": 1.0,
    "reassuring": 0.95,
    "apologetic": 0.9,
}


def _determine_tone(valence: float, arousal: float, emotion: str) -> str:
    if valence < -0.5 and arousal > 0.6:
        return "calm"
    if emotion == "angry":
        return "calm"
    if emotion == "sad":
        return "empathetic"
    if emotion == "fearful":
        return "reassuring"
    if emotion == "happy":
        return "enthusiastic"
    if valence < -0.3:
        return "empathetic"
    return "professional"


def _determine_intervention(
    escalation_prob: float,
    consecutive_negative: int,
    valence: float,
) -> tuple[str, float]:
    if escalation_prob > 0.8 and consecutive_negative >= 5:
        return "escalate_to_human", 1.0
    if escalation_prob > 0.6 and consecutive_negative >= 3:
        return "offer_callback", 0.7
    if escalation_prob > 0.4 or valence < -0.5:
        return "tone_shift", 0.4
    return "none", 0.0


class ProsodicPipeline:
    """
    Continuous prosodic feedback loop.
    Runs in heuristic mode when model is None (no torch/prosody_ssm required).
    """

    DIRECTIVE_INTERVAL = 4

    def __init__(
        self,
        model=None,
        predictor=None,
        bus: Optional[AudioBus] = None,
        store: Optional[SessionStore] = None,
        sample_rate: int = 16000,
        directive_interval: int = 4,
    ):
        self.model = model
        self.predictor = predictor
        self.bus = bus
        self.store = store
        self.sample_rate = sample_rate
        self.DIRECTIVE_INTERVAL = directive_interval
        self._extractors: dict[str, FrameExtractor] = {}

    def _get_extractor(self, session_id: str) -> FrameExtractor:
        if session_id not in self._extractors:
            self._extractors[session_id] = FrameExtractor(sample_rate=self.sample_rate)
        return self._extractors[session_id]

    async def process_frame(
        self,
        session_id: str,
        pcm_data: bytes,
    ) -> Optional[PipelineOutput]:
        """Process a single audio chunk for a session."""
        extractor = self._get_extractor(session_id)
        frames = extractor.process_frames(pcm_data)
        if not frames:
            return None

        state = None
        if self.store:
            state = await self.store.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)

        last_output = None
        for frame in frames:
            state.frames_processed += 1
            last_output = self._process_prosody_frame(frame, state)

        if self.store:
            await self.store.set(state)

        if state.frames_processed % self.DIRECTIVE_INTERVAL == 0 and last_output:
            return last_output
        return None

    def _process_prosody_frame(
        self,
        frame: ProsodyFrame,
        state: SessionState,
    ) -> PipelineOutput:
        emotion_probs_dict = {}
        valence = 0.0
        arousal = 0.5
        dominance = 0.5
        current_emotion = "neutral"
        confidence = 0.0

        if TORCH_AVAILABLE and self.model is not None:
            prosody_vec = torch.from_numpy(frame.to_vector()).float().unsqueeze(0)
            device = next(self.model.parameters()).device
            prosody_vec = prosody_vec.to(device)
            probs, vad, new_ssm_state = self.model.step(
                prosody_vec,
                state=state.ssm_state,
            )
            state.ssm_state = new_ssm_state
            probs_np = probs[0].cpu().numpy()
            vad_np = vad[0].cpu().numpy()
            emotions = list(EmotionLabel)
            emotion_probs_dict = {e.value: float(probs_np[i]) for i, e in enumerate(emotions) if i < len(probs_np)}
            best_idx = int(np.argmax(probs_np))
            current_emotion = emotions[best_idx].value if best_idx < len(emotions) else "neutral"
            confidence = float(probs_np[best_idx])
            valence = float(vad_np[0]) if len(vad_np) > 0 else 0.0
            arousal = float(vad_np[1]) if len(vad_np) > 1 else 0.5
            dominance = float(vad_np[2]) if len(vad_np) > 2 else 0.5
        else:
            if frame.energy > 0.1 and frame.f0_mean > 200:
                current_emotion = "angry"
                valence = -0.5
                arousal = 0.8
            elif frame.energy < 0.02:
                current_emotion = "sad"
                valence = -0.3
                arousal = 0.3
            else:
                current_emotion = "neutral"
                valence = 0.0
                arousal = 0.5
            confidence = 0.5
            emotion_probs_dict = {current_emotion: confidence}

        state.prosody_history.append({
            "emotion_probs": list(emotion_probs_dict.values()),
            "vad": [valence, arousal, dominance],
            "confidence": confidence,
            "timestamp": frame.timestamp_ms,
        })
        if len(state.prosody_history) > 50:
            state.prosody_history = state.prosody_history[-50:]

        escalation_prob = 0.0
        churn_risk = 0.0
        resolution_prob = 0.5
        predicted_csat = 3.0
        sentiment_forecast = 0.0
        recommended_tone = "professional"

        if self.predictor is not None and len(state.prosody_history) >= 3:
            try:
                pred = self.predictor.predict(
                    [h["emotion_probs"] for h in state.prosody_history],
                    [h["vad"] for h in state.prosody_history],
                    [h["confidence"] for h in state.prosody_history],
                )
                if pred is not None:
                    escalation_prob = getattr(pred, "will_escalate", 0.0)
                    churn_risk = getattr(pred, "churn_risk", 0.0)
                    resolution_prob = getattr(pred, "resolution_prob", 0.5)
                    predicted_csat = getattr(pred, "final_csat", 3.0)
                    sentiment_forecast = getattr(pred, "sentiment_forecast", 0.0)
                    recommended_tone = getattr(getattr(pred, "recommended_tone", None), "value", "professional") or "professional"
            except Exception as e:
                logger.debug("ConversationPredictor error: %s", e)

        if recommended_tone == "professional":
            recommended_tone = _determine_tone(valence, arousal, current_emotion)

        consecutive_neg = 0
        for h in reversed(state.prosody_history):
            if h["vad"][0] < -0.3:
                consecutive_neg += 1
            else:
                break
        intervention_type, intervention_urgency = _determine_intervention(
            escalation_prob, consecutive_neg, valence
        )

        tts_emotion = _TTS_EMOTION_MAP.get(recommended_tone, "neutral")
        tts_speed = _TTS_SPEED_MAP.get(recommended_tone, 1.0)
        llm_context = (
            f"Customer emotion: {current_emotion} (valence={valence:+.2f}, "
            f"arousal={arousal:.2f}). Escalation risk: {escalation_prob:.0%}. "
            f"Use {recommended_tone} tone."
        )
        if intervention_type != "none":
            llm_context += f" Intervention: {intervention_type}."

        directive = AgentDirective(
            tts_emotion=tts_emotion,
            tts_speed=tts_speed,
            llm_context=llm_context,
            escalation_prob=escalation_prob,
            churn_risk=churn_risk,
            resolution_prob=resolution_prob,
            predicted_csat=predicted_csat,
            sentiment_forecast=sentiment_forecast,
            intervention_type=intervention_type,
            intervention_urgency=intervention_urgency,
            current_emotion=current_emotion,
            emotion_confidence=confidence,
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            timestamp_ms=frame.timestamp_ms,
            frames_processed=state.frames_processed,
        )
        return PipelineOutput(
            directive=directive,
            prosody_frame=frame,
            emotion_probs=emotion_probs_dict,
            raw_valence=valence,
            raw_arousal=arousal,
        )

    async def run(self, session_id: str) -> AsyncIterator[PipelineOutput]:
        if self.bus is None:
            raise RuntimeError("AudioBus required for run(). Use process_frame() for direct calls.")
        async for audio_frame in self.bus.subscribe(session_id):
            output = await self.process_frame(session_id, audio_frame.pcm_data)
            if output is not None:
                yield output

    async def close_session(self, session_id: str) -> None:
        self._extractors.pop(session_id, None)
        if self.store:
            await self.store.delete(session_id)
        if self.bus:
            await self.bus.close_session(session_id)
