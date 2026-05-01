"""Agent-modulation engine for the streaming pipeline.

Consumes per-chunk ``AgentDirective``s and produces:

1. A continuous ``agent_modulation`` payload attached to every directive
   (TTS shaping params + a short system-prompt fragment + intensity).
2. A discrete ``agent_steering`` transition event whenever the modulation
   state changes (entering or leaving ``de_escalate`` / ``mirror_calm`` /
   ``agent_overheated`` / ``normal``). The event carries an LLM-ready
   system-prompt block that the voice agent can inject into its next turn.

This module is read-only of ``AgentDirective`` and has no I/O. It is meant to
be cheap to call once per chunk per session.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Tunable thresholds and TTS deltas (single source of truth)
# ---------------------------------------------------------------------------

# Rolling-window length per speaker (number of chunks remembered).
WINDOW_LEN = 4

# Number of consecutive chunks required to enter / leave a state.
# 2 chunks ≈ 4s with the current 2s pipeline chunk size.
ENTER_STREAK = 2
EXIT_STREAK = 2
NORMAL_STREAK = 3

# Caller escalation thresholds.
CALLER_AROUSAL_HIGH = 0.7
CALLER_VALENCE_NEG = -0.3
CALLER_EMOTION_NEG = frozenset({"angry", "fearful", "disgusted", "contempt"})
CALLER_EMOTION_CONF = 0.5

# Caller cooldown thresholds (used to leave caller_escalating).
CALLER_AROUSAL_COOL = 0.5
CALLER_VALENCE_COOL = -0.1

# Mirror-calm exit thresholds (return to normal).
NORMAL_AROUSAL_MAX = 0.55

# Agent overheated thresholds (only evaluated when speaker_id == 'agent').
AGENT_AROUSAL_HIGH = 0.65
AGENT_VALENCE_NEG = 0.0

# TTS shaping baselines and deltas.
BASE_TTS = {
    "speed": 1.0,
    "pitch_shift_semitones": 0.0,
    "emotion": "neutral",
    "target_intensity": 0.5,
    "pre_pause_ms": 0,
}

VALID_MODES = ("normal", "caller_escalating", "mirror_calm", "agent_overheated")

# Speaker tag classification.
AGENT_TAGS = frozenset({"agent"})
# Anything else (caller, speaker_0/1/..., unknown) feeds the caller window.
# Agent overheating only triggers from explicit "agent" tag, which requires
# the WS client to have called enroll_agent first.


# ---------------------------------------------------------------------------
# Snapshot + state types
# ---------------------------------------------------------------------------


@dataclass
class Snap:
    """One chunk's prosody snapshot, kept on a rolling window."""

    arousal: float = 0.5
    valence: float = 0.0
    dominance: float = 0.5
    emotion: str = "neutral"
    confidence: float = 0.0
    stress: float = 0.0
    timestamp_ms: int = 0


@dataclass
class ModulationState:
    """Per-session modulation state. Holds rolling per-speaker windows."""

    mode: str = "normal"
    previous_mode: str = "normal"
    mode_entered_at_ms: int = 0
    last_caller: Optional[Snap] = None
    last_agent: Optional[Snap] = None
    caller_window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_LEN))
    agent_window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_LEN))
    agent_enrolled: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ModulationEngine:
    """Stateless logic — operates on a caller-supplied ``ModulationState``.

    Kept stateless so the pipeline can own one ``ModulationState`` per session
    while sharing a single engine instance across the process.
    """

    def update(
        self,
        state: ModulationState,
        directive: Any,
    ) -> tuple[dict, Optional[dict]]:
        """Push a new directive into the rolling state and emit modulation.

        Returns a tuple of:
          - ``continuous`` dict to attach to the directive as ``agent_modulation``
          - ``transition`` event dict (or ``None`` if the mode did not change)
        """
        snap = self._snap_from_directive(directive)
        speaker = getattr(directive, "speaker_id", None) or "unknown"
        if speaker in AGENT_TAGS:
            state.agent_window.append(snap)
            state.last_agent = snap
            state.agent_enrolled = True
        else:
            state.caller_window.append(snap)
            state.last_caller = snap

        new_mode = self._classify(state)
        intensity = self._intensity(state, new_mode)
        ts_ms = int(getattr(directive, "timestamp_ms", 0) or 0)

        transition: Optional[dict] = None
        if new_mode != state.mode:
            transition = self._transition_event(
                session_id=str(getattr(directive, "session_id", "") or ""),
                previous_mode=state.mode,
                mode=new_mode,
                intensity=intensity,
                state=state,
                timestamp_ms=ts_ms,
            )
            state.previous_mode = state.mode
            state.mode = new_mode
            state.mode_entered_at_ms = ts_ms

        continuous = self._continuous(state, intensity)
        return continuous, transition

    # --- classification ------------------------------------------------------

    def _classify(self, state: ModulationState) -> str:
        """Apply the state-machine rules to the current rolling state."""
        # agent_overheated takes priority, but only when the agent voice is
        # actually enrolled — we do NOT want unknown / clustered speakers to
        # masquerade as the agent.
        if state.agent_enrolled and self._agent_overheated(state):
            return "agent_overheated"

        # Stay in agent_overheated until we have ENTER_STREAK calm agent chunks.
        if state.mode == "agent_overheated":
            if self._agent_calm_streak(state, ENTER_STREAK):
                return "normal"
            return "agent_overheated"

        if state.mode == "caller_escalating":
            if self._caller_cooldown_streak(state, EXIT_STREAK):
                return "mirror_calm"
            return "caller_escalating"

        if state.mode == "mirror_calm":
            if self._caller_normal_streak(state, NORMAL_STREAK):
                return "normal"
            # Re-escalation while mirroring → jump back to caller_escalating.
            if self._caller_escalating_streak(state, ENTER_STREAK):
                return "caller_escalating"
            return "mirror_calm"

        # state.mode == "normal" (or any unknown mode) — look for entry.
        if self._caller_escalating_streak(state, ENTER_STREAK):
            return "caller_escalating"
        return "normal"

    # --- streak predicates ---------------------------------------------------

    @staticmethod
    def _is_caller_escalating(s: Snap) -> bool:
        if s.arousal >= CALLER_AROUSAL_HIGH and s.valence <= CALLER_VALENCE_NEG:
            return True
        if (
            s.emotion in CALLER_EMOTION_NEG
            and s.confidence >= CALLER_EMOTION_CONF
            and s.arousal >= CALLER_AROUSAL_HIGH - 0.1
        ):
            return True
        return False

    @staticmethod
    def _is_caller_cool(s: Snap) -> bool:
        return s.arousal < CALLER_AROUSAL_COOL and s.valence > CALLER_VALENCE_COOL

    @staticmethod
    def _is_caller_normal(s: Snap) -> bool:
        return s.arousal <= NORMAL_AROUSAL_MAX

    @staticmethod
    def _is_agent_overheated(s: Snap) -> bool:
        return s.arousal >= AGENT_AROUSAL_HIGH and s.valence < AGENT_VALENCE_NEG

    def _caller_escalating_streak(self, state: ModulationState, n: int) -> bool:
        return self._tail_streak(state.caller_window, self._is_caller_escalating, n)

    def _caller_cooldown_streak(self, state: ModulationState, n: int) -> bool:
        return self._tail_streak(state.caller_window, self._is_caller_cool, n)

    def _caller_normal_streak(self, state: ModulationState, n: int) -> bool:
        return self._tail_streak(state.caller_window, self._is_caller_normal, n)

    def _agent_overheated(self, state: ModulationState) -> bool:
        return self._tail_streak(state.agent_window, self._is_agent_overheated, ENTER_STREAK)

    def _agent_calm_streak(self, state: ModulationState, n: int) -> bool:
        return self._tail_streak(
            state.agent_window,
            lambda s: s.arousal < AGENT_AROUSAL_HIGH - 0.05 and s.valence >= AGENT_VALENCE_NEG,
            n,
        )

    @staticmethod
    def _tail_streak(window: deque, predicate, n: int) -> bool:
        if len(window) < n:
            return False
        for s in list(window)[-n:]:
            if not predicate(s):
                return False
        return True

    # --- intensity / payload builders ---------------------------------------

    def _intensity(self, state: ModulationState, mode: str) -> float:
        """Map state into a 0..1 intensity used to scale TTS deltas and prompts."""
        if mode == "caller_escalating":
            s = state.last_caller
            if s is None:
                return 0.5
            ar = max(0.0, min(1.0, (s.arousal - CALLER_AROUSAL_HIGH) / 0.3))
            va = max(0.0, min(1.0, (CALLER_VALENCE_NEG - s.valence) / 0.7))
            base = 0.5 + 0.5 * max(ar, va)
            return round(min(1.0, base + 0.2 * s.stress), 3)
        if mode == "agent_overheated":
            s = state.last_agent
            if s is None:
                return 0.6
            ar = max(0.0, min(1.0, (s.arousal - AGENT_AROUSAL_HIGH) / 0.35))
            return round(min(1.0, 0.5 + 0.5 * ar), 3)
        if mode == "mirror_calm":
            return 0.4
        return 0.0

    def _tts_for(self, mode: str, intensity: float) -> dict:
        """Map (mode, intensity) → TTS shaping params for the agent's voice."""
        if mode == "caller_escalating":
            return {
                "speed": round(1.0 - 0.2 * intensity, 3),
                "pitch_shift_semitones": round(-2.0 * intensity, 2),
                "emotion": "calm",
                "target_intensity": round(max(0.2, 0.5 - 0.3 * intensity), 3),
                "pre_pause_ms": int(150 + 200 * intensity),
            }
        if mode == "mirror_calm":
            return {
                "speed": round(1.0 - 0.08 * intensity, 3),
                "pitch_shift_semitones": round(-0.5 * intensity, 2),
                "emotion": "warm",
                "target_intensity": round(0.4 + 0.05 * intensity, 3),
                "pre_pause_ms": 100,
            }
        if mode == "agent_overheated":
            return {
                "speed": round(1.0 - 0.25 * intensity, 3),
                "pitch_shift_semitones": round(-3.0 * intensity, 2),
                "emotion": "calm",
                "target_intensity": round(max(0.15, 0.4 - 0.25 * intensity), 3),
                "pre_pause_ms": int(200 + 250 * intensity),
            }
        return dict(BASE_TTS)

    def _short_fragment(self, mode: str, state: ModulationState) -> str:
        if mode == "caller_escalating":
            return (
                "Caller voice indicates rising frustration. Acknowledge their feelings "
                "explicitly before substance. Speak slowly and softly. Keep your reply "
                "to one short sentence; no lists or menus."
            )
        if mode == "mirror_calm":
            return (
                "Caller has cooled down. Stay warm and brief; mirror their lower "
                "energy. Avoid abruptly returning to a sales or upsell tone."
            )
        if mode == "agent_overheated":
            return (
                "Agent voice has trended high-arousal/negative. Soften delivery, "
                "slow down, and lower pitch. Avoid argumentative phrasing."
            )
        return "No modulation needed; respond naturally."

    def _full_prompt(self, mode: str, state: ModulationState, reason: str) -> str:
        if mode == "caller_escalating":
            return (
                "[ProsodyAI escalation alert]\n"
                f"Reason: {reason}\n\n"
                "Steering for your next reply:\n"
                "- Acknowledge the caller's feelings in the first clause "
                "(e.g. 'I hear you', 'That's frustrating').\n"
                "- Slow your pace; lower pitch; keep volume soft.\n"
                "- Reply with ONE short sentence. No lists, no prices, no menu items.\n"
                "- If the caller is still rising next turn, offer to connect them "
                "with a human."
            )
        if mode == "mirror_calm":
            return (
                "[ProsodyAI cooldown]\n"
                f"Reason: {reason}\n\n"
                "Steering for your next reply:\n"
                "- The caller has cooled down. Stay warm and brief.\n"
                "- Mirror their reduced energy; do not jump back to upsell or "
                "sales tone.\n"
                "- Confirm understanding before moving on."
            )
        if mode == "agent_overheated":
            return (
                "[ProsodyAI agent self-check]\n"
                f"Reason: {reason}\n\n"
                "Your own voice has trended high-arousal and negative. "
                "Steering for your next reply:\n"
                "- Take a breath; soften delivery and slow down.\n"
                "- Lower pitch; reduce vocal intensity.\n"
                "- Avoid argumentative phrasing; lead with collaborative language."
            )
        return ""

    def _reason_for(self, mode: str, state: ModulationState) -> str:
        if mode == "caller_escalating":
            s = state.last_caller
            if s is None:
                return "caller escalation detected"
            return (
                f"caller arousal={round(s.arousal, 2)} valence={round(s.valence, 2)} "
                f"emotion={s.emotion} sustained {ENTER_STREAK} chunks"
            )
        if mode == "mirror_calm":
            s = state.last_caller
            if s is None:
                return "caller cooled down"
            return (
                f"caller arousal dropped to {round(s.arousal, 2)} "
                f"and valence rose to {round(s.valence, 2)}"
            )
        if mode == "agent_overheated":
            s = state.last_agent
            if s is None:
                return "agent voice high arousal / negative valence"
            return (
                f"agent arousal={round(s.arousal, 2)} valence={round(s.valence, 2)} "
                f"sustained {ENTER_STREAK} chunks"
            )
        s = state.last_caller
        if s is None:
            return "stable"
        return (
            f"caller arousal={round(s.arousal, 2)} valence={round(s.valence, 2)} stable"
        )

    def _continuous(self, state: ModulationState, intensity: float) -> dict:
        mode = state.mode
        return {
            "mode": mode,
            "intensity": round(intensity, 3),
            "tts": self._tts_for(mode, intensity),
            "system_prompt_fragment": self._short_fragment(mode, state),
        }

    def _transition_event(
        self,
        *,
        session_id: str,
        previous_mode: str,
        mode: str,
        intensity: float,
        state: ModulationState,
        timestamp_ms: int,
    ) -> dict:
        reason = self._reason_for(mode, state)
        return {
            "type": "agent_steering",
            "session_id": session_id,
            "previous_mode": previous_mode,
            "mode": mode,
            "intensity": round(intensity, 3),
            "reason": reason,
            "tts": self._tts_for(mode, intensity),
            "system_prompt": self._full_prompt(mode, state, reason),
            "timestamp_ms": timestamp_ms,
        }

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _snap_from_directive(directive: Any) -> Snap:
        signals = getattr(directive, "signals", None) or {}
        return Snap(
            arousal=float(getattr(directive, "arousal", 0.5) or 0.0),
            valence=float(getattr(directive, "valence", 0.0) or 0.0),
            dominance=float(getattr(directive, "dominance", 0.5) or 0.0),
            emotion=str(getattr(directive, "emotion", "neutral") or "neutral"),
            confidence=float(getattr(directive, "confidence", 0.0) or 0.0),
            stress=float(signals.get("stress", 0.0) or 0.0),
            timestamp_ms=int(getattr(directive, "timestamp_ms", 0) or 0),
        )


_engine_singleton: Optional[ModulationEngine] = None


def get_modulation_engine() -> ModulationEngine:
    """Return the process-wide stateless engine instance."""
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ModulationEngine()
    return _engine_singleton
