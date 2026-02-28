"""
KPI outcome predictor for ProsodyAI.

Takes raw prosodic signals (NOT emotion labels) and predicts outcomes for
client-defined KPIs.  The pipeline is:

    audio → prosodic features → KPI predictions

There is no emotion classification step.  The prosodic feature vector is the
universal input that drives all predictions.

Prosodic feature vector (from the model service):
    - valence      (-1 to +1)  overall positive/negative tone
    - arousal      (0 to 1)    energy / activation level
    - dominance    (0 to 1)    assertiveness / control
    - pitch_mean, pitch_std, pitch_range   fundamental frequency stats
    - energy_mean, energy_std              RMS energy stats
    - speech_rate                          syllables per second
    - jitter, shimmer, hnr                 voice quality
    - mfcc_means (13-dim)                  spectral shape
    - spectral_centroid, spectral_rolloff  brightness / timbre

For each client-defined KPI, the predictor produces:
    - predicted_value   (scalar, probability, or category)
    - confidence        how sure we are
    - trajectory        improving / stable / declining (in conversation context)
    - impact_factors    which prosodic signals are driving the prediction
    - recommended_actions  what to change to improve the KPI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from kpis import KPIDefinition, KPIType, KPIDirection, AlertDirection


# ---------------------------------------------------------------------------
# Prediction result types
# ---------------------------------------------------------------------------


@dataclass
class ImpactFactor:
    """A prosodic signal that is influencing a KPI prediction."""

    signal: str          # e.g., "valence", "speech_rate", "pitch_range"
    value: float         # current value of this signal
    impact: float        # how much it's pushing the KPI (signed)
    description: str     # human-readable explanation


@dataclass
class RecommendedAction:
    """An action that could improve the KPI outcome."""

    action: str              # what to do
    expected_impact: float   # estimated improvement
    signal_target: str       # which prosodic signal this targets


@dataclass
class KPIAlert:
    """Alert when a predicted KPI crosses its threshold."""

    kpi_name: str
    predicted_value: float
    threshold: float
    direction: str           # "above" or "below"
    message: str


@dataclass
class KPIPrediction:
    """Prediction for a single client-defined KPI."""

    kpi_id: str
    kpi_name: str
    kpi_type: str

    # Prediction
    predicted_value: Any         # float for scalar, bool for binary, str for categorical
    confidence: float            # 0-1

    # Trajectory (requires conversation history)
    trajectory: Optional[str] = None  # "improving", "stable", "declining"

    # Explainability
    impact_factors: list[ImpactFactor] = field(default_factory=list)
    recommended_actions: list[RecommendedAction] = field(default_factory=list)

    # Alert (if threshold crossed)
    alert: Optional[KPIAlert] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kpi_id": self.kpi_id,
            "kpi_name": self.kpi_name,
            "kpi_type": self.kpi_type,
            "predicted_value": self.predicted_value,
            "confidence": round(self.confidence, 3),
        }
        if self.trajectory:
            result["trajectory"] = self.trajectory
        if self.impact_factors:
            result["impact_factors"] = [
                {
                    "signal": f.signal,
                    "value": round(f.value, 3),
                    "impact": round(f.impact, 3),
                    "description": f.description,
                }
                for f in self.impact_factors
            ]
        if self.recommended_actions:
            result["recommended_actions"] = [
                {
                    "action": a.action,
                    "expected_impact": round(a.expected_impact, 3),
                    "signal_target": a.signal_target,
                }
                for a in self.recommended_actions
            ]
        if self.alert:
            result["alert"] = {
                "threshold": self.alert.threshold,
                "direction": self.alert.direction,
                "message": self.alert.message,
            }
        return result


@dataclass
class ProsodySignals:
    """
    Raw prosodic feature vector — the universal input for all predictions.

    This is what the model service returns.  No emotion labels.
    """

    # Core VAD dimensions
    valence: float = 0.0        # -1 (negative) to +1 (positive)
    arousal: float = 0.5        # 0 (calm) to 1 (activated)
    dominance: float = 0.5      # 0 (submissive) to 1 (dominant)

    # Pitch (fundamental frequency)
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    pitch_range: Optional[float] = None

    # Energy
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None

    # Rhythm
    speech_rate: Optional[float] = None       # syllables/sec

    # Voice quality
    jitter: Optional[float] = None
    shimmer: Optional[float] = None
    hnr: Optional[float] = None               # harmonics-to-noise ratio

    # Spectral
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    mfcc_means: Optional[list[float]] = None  # 13-dim

    # Prosody markers (categorical, from model service)
    pitch_trend: Optional[str] = None         # rising, falling, flat, varied
    intensity: Optional[str] = None           # soft, normal, loud, emphasized
    tempo: Optional[str] = None               # slow, normal, fast

    @classmethod
    def from_model_prediction(cls, prediction) -> "ProsodySignals":
        """
        Build from a ModelPrediction (model_client.ModelPrediction).

        Extracts all available prosodic features — NO emotion fields.
        """
        features = prediction.prosody_features or {}

        return cls(
            valence=prediction.valence,
            arousal=prediction.arousal,
            dominance=prediction.dominance,
            pitch_mean=features.get("f0_mean"),
            pitch_std=features.get("f0_std"),
            pitch_range=features.get("f0_range"),
            energy_mean=features.get("energy_mean"),
            energy_std=features.get("energy_std"),
            speech_rate=features.get("speech_rate"),
            jitter=features.get("jitter"),
            shimmer=features.get("shimmer"),
            hnr=features.get("hnr"),
            spectral_centroid=features.get("spectral_centroid_mean"),
            spectral_rolloff=features.get("spectral_rolloff_mean"),
            mfcc_means=features.get("mfcc_means"),
            pitch_trend=prediction.pitch_trend,
            intensity=prediction.intensity,
            tempo=prediction.tempo,
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class KPIPredictor:
    """
    Predicts KPI outcomes from raw prosodic signals.

    Currently uses heuristic mappings based on established psycholinguistic
    research (e.g., low valence + high arousal correlates with poor satisfaction
    outcomes).  As clients report actual outcomes via the feedback loop, these
    heuristics will be replaced by per-client learned models.

    The predictor is stateless per call — conversation-level trajectory
    tracking is handled by passing in a history of ProsodySignals.
    """

    def predict(
        self,
        signals: ProsodySignals,
        kpis: list[KPIDefinition],
        history: Optional[list[ProsodySignals]] = None,
    ) -> list[KPIPrediction]:
        """
        Predict outcomes for all client-defined KPIs.

        Args:
            signals: Current prosodic feature vector.
            kpis: Client's KPI definitions (from the database).
            history: Previous prosodic signals in this conversation (for trajectory).

        Returns:
            List of KPIPredictions, one per KPI.
        """
        predictions = []
        for kpi in kpis:
            if not kpi.enabled:
                continue
            pred = self._predict_single(signals, kpi, history)
            predictions.append(pred)
        return predictions

    def _predict_single(
        self,
        signals: ProsodySignals,
        kpi: KPIDefinition,
        history: Optional[list[ProsodySignals]],
    ) -> KPIPrediction:
        """Predict a single KPI from prosodic signals."""

        if kpi.kpi_type == KPIType.SCALAR:
            return self._predict_scalar(signals, kpi, history)
        elif kpi.kpi_type == KPIType.BINARY:
            return self._predict_binary(signals, kpi, history)
        elif kpi.kpi_type == KPIType.CATEGORICAL:
            return self._predict_categorical(signals, kpi, history)
        else:
            # Fallback
            return KPIPrediction(
                kpi_id=kpi.id,
                kpi_name=kpi.name,
                kpi_type=kpi.kpi_type.value,
                predicted_value=None,
                confidence=0.0,
            )

    # -- Scalar KPIs --------------------------------------------------------

    def _predict_scalar(
        self,
        signals: ProsodySignals,
        kpi: KPIDefinition,
        history: Optional[list[ProsodySignals]],
    ) -> KPIPrediction:
        """
        Predict a scalar KPI (e.g., CSAT 1-5, NPS -100 to 100).

        Heuristic: valence is the primary driver, arousal and dominance
        modulate.  The prediction is scaled to the KPI's range.
        """
        range_min = kpi.range_min if kpi.range_min is not None else 0.0
        range_max = kpi.range_max if kpi.range_max is not None else 1.0
        midpoint = (range_min + range_max) / 2.0
        half_range = (range_max - range_min) / 2.0

        # Base prediction: valence drives the core signal
        # valence is -1 to +1, we map it to the KPI range
        if kpi.direction == KPIDirection.HIGHER_IS_BETTER:
            base = midpoint + signals.valence * half_range * 0.6
        else:
            # Lower is better: invert valence relationship
            base = midpoint - signals.valence * half_range * 0.6

        # Arousal modulation: high arousal pushes predictions toward extremes
        arousal_factor = (signals.arousal - 0.5) * 0.2
        if kpi.direction == KPIDirection.HIGHER_IS_BETTER:
            # High arousal + positive valence = even better
            # High arousal + negative valence = even worse
            base += signals.valence * arousal_factor * half_range
        else:
            base -= signals.valence * arousal_factor * half_range

        # Dominance modulation: high dominance slightly improves outcomes
        # (speaker control/confidence tends to correlate with better outcomes)
        dominance_nudge = (signals.dominance - 0.5) * 0.1 * half_range
        if kpi.direction == KPIDirection.HIGHER_IS_BETTER:
            base += dominance_nudge
        else:
            base -= dominance_nudge

        # Voice quality modulation (when available)
        if signals.jitter is not None and signals.shimmer is not None:
            # High jitter/shimmer = vocal stress, generally worsens outcomes
            stress = (signals.jitter + signals.shimmer) / 2
            stress_impact = stress * 0.1 * half_range
            if kpi.direction == KPIDirection.HIGHER_IS_BETTER:
                base -= stress_impact
            else:
                base += stress_impact

        # Clamp to range
        predicted = max(range_min, min(range_max, base))

        # Impact factors
        factors = self._compute_impact_factors(signals, kpi)

        # Trajectory
        trajectory = self._compute_trajectory(signals, history) if history else None

        # Confidence: higher with more available signals
        confidence = self._compute_confidence(signals)

        # Recommended actions
        actions = self._recommend_actions(signals, kpi, predicted)

        # Alert check
        alert = self._check_alert(kpi, predicted)

        return KPIPrediction(
            kpi_id=kpi.id,
            kpi_name=kpi.name,
            kpi_type=kpi.kpi_type.value,
            predicted_value=round(predicted, 2),
            confidence=confidence,
            trajectory=trajectory,
            impact_factors=factors,
            recommended_actions=actions,
            alert=alert,
        )

    # -- Binary KPIs --------------------------------------------------------

    def _predict_binary(
        self,
        signals: ProsodySignals,
        kpi: KPIDefinition,
        history: Optional[list[ProsodySignals]],
    ) -> KPIPrediction:
        """
        Predict a binary KPI (e.g., will_escalate, deal_closed).

        Returns a probability (0-1).  The direction field tells us
        whether True is "good" or "bad":
        - higher_is_better: P(True) is the outcome we want
        - lower_is_better: P(True) is the outcome we want to avoid
        """
        # Base probability from valence
        if kpi.direction == KPIDirection.HIGHER_IS_BETTER:
            # Positive outcome: higher valence = more likely
            base_prob = 0.5 + signals.valence * 0.3
        else:
            # Negative outcome (e.g., churn, escalation): negative signals = more likely
            base_prob = 0.5 - signals.valence * 0.3

        # Arousal modulation
        # High arousal increases probability of extreme outcomes (both good and bad)
        arousal_shift = (signals.arousal - 0.5) * 0.15
        if kpi.direction == KPIDirection.HIGHER_IS_BETTER:
            base_prob += signals.valence * arousal_shift
        else:
            base_prob -= signals.valence * arousal_shift

        # Conversation history: sustained negative patterns increase risk
        if history and len(history) >= 3:
            recent_valences = [s.valence for s in history[-5:]]
            avg_recent = sum(recent_valences) / len(recent_valences)
            if kpi.direction == KPIDirection.LOWER_IS_BETTER:
                # Sustained negativity increases probability of bad outcome
                if avg_recent < -0.3:
                    base_prob += 0.15
            else:
                if avg_recent > 0.3:
                    base_prob += 0.1

        prob = max(0.0, min(1.0, base_prob))

        factors = self._compute_impact_factors(signals, kpi)
        trajectory = self._compute_trajectory(signals, history) if history else None
        confidence = self._compute_confidence(signals)
        actions = self._recommend_actions(signals, kpi, prob)
        alert = self._check_alert(kpi, prob)

        return KPIPrediction(
            kpi_id=kpi.id,
            kpi_name=kpi.name,
            kpi_type=kpi.kpi_type.value,
            predicted_value=round(prob, 3),
            confidence=confidence,
            trajectory=trajectory,
            impact_factors=factors,
            recommended_actions=actions,
            alert=alert,
        )

    # -- Categorical KPIs ---------------------------------------------------

    def _predict_categorical(
        self,
        signals: ProsodySignals,
        kpi: KPIDefinition,
        history: Optional[list[ProsodySignals]],
    ) -> KPIPrediction:
        """
        Predict a categorical KPI (e.g., resolution_type).

        Returns the most likely category.  Without per-client learned models,
        this is a coarse mapping based on valence/arousal quadrants.
        """
        categories = kpi.categories or []
        if not categories:
            return KPIPrediction(
                kpi_id=kpi.id,
                kpi_name=kpi.name,
                kpi_type=kpi.kpi_type.value,
                predicted_value=None,
                confidence=0.0,
            )

        # Distribute categories across the valence-arousal space
        # This is a placeholder — with outcome data, we learn real boundaries
        n = len(categories)
        # Map valence (-1 to 1) to index space
        normalized = (signals.valence + 1) / 2  # 0 to 1
        idx = int(normalized * (n - 1) + 0.5)
        idx = max(0, min(n - 1, idx))

        predicted_category = categories[idx]
        confidence = self._compute_confidence(signals) * 0.7  # lower for categorical

        factors = self._compute_impact_factors(signals, kpi)
        trajectory = self._compute_trajectory(signals, history) if history else None

        return KPIPrediction(
            kpi_id=kpi.id,
            kpi_name=kpi.name,
            kpi_type=kpi.kpi_type.value,
            predicted_value=predicted_category,
            confidence=confidence,
            trajectory=trajectory,
            impact_factors=factors,
        )

    # -- Shared helpers -----------------------------------------------------

    def _compute_impact_factors(
        self, signals: ProsodySignals, kpi: KPIDefinition,
    ) -> list[ImpactFactor]:
        """Identify which prosodic signals are most influencing this prediction."""
        factors: list[ImpactFactor] = []
        is_higher_better = kpi.direction == KPIDirection.HIGHER_IS_BETTER

        # Valence: always the primary driver
        if abs(signals.valence) > 0.1:
            positive = signals.valence > 0
            good_for_kpi = (positive and is_higher_better) or (not positive and not is_higher_better)
            factors.append(ImpactFactor(
                signal="valence",
                value=signals.valence,
                impact=signals.valence if is_higher_better else -signals.valence,
                description=(
                    "Positive vocal tone is helping this KPI"
                    if good_for_kpi
                    else "Negative vocal tone is hurting this KPI"
                ),
            ))

        # Arousal: significant when extreme
        if abs(signals.arousal - 0.5) > 0.15:
            high_arousal = signals.arousal > 0.5
            factors.append(ImpactFactor(
                signal="arousal",
                value=signals.arousal,
                impact=(signals.arousal - 0.5) * signals.valence * 0.5,
                description=(
                    "High activation level is amplifying the current trend"
                    if high_arousal
                    else "Low energy may indicate disengagement"
                ),
            ))

        # Speech rate: when available and extreme
        if signals.speech_rate is not None:
            if signals.speech_rate > 5.0:  # fast
                factors.append(ImpactFactor(
                    signal="speech_rate",
                    value=signals.speech_rate,
                    impact=-0.1 if is_higher_better else 0.1,
                    description="Rapid speech suggests urgency or agitation",
                ))
            elif signals.speech_rate < 2.5:  # slow
                factors.append(ImpactFactor(
                    signal="speech_rate",
                    value=signals.speech_rate,
                    impact=0.05 if is_higher_better else -0.05,
                    description="Slow speech may indicate careful thought or fatigue",
                ))

        # Voice quality stress markers
        if signals.jitter is not None and signals.jitter > 0.02:
            factors.append(ImpactFactor(
                signal="jitter",
                value=signals.jitter,
                impact=-0.1 if is_higher_better else 0.1,
                description="Vocal instability (jitter) suggests stress",
            ))

        if signals.hnr is not None and signals.hnr < 10:
            factors.append(ImpactFactor(
                signal="hnr",
                value=signals.hnr,
                impact=-0.05 if is_higher_better else 0.05,
                description="Low voice clarity (HNR) may indicate strain",
            ))

        return factors

    def _compute_trajectory(
        self,
        current: ProsodySignals,
        history: Optional[list[ProsodySignals]],
    ) -> Optional[str]:
        """Compute trajectory from conversation history."""
        if not history or len(history) < 3:
            return None

        mid = len(history) // 2
        first_half = history[:mid]
        second_half = history[mid:]

        first_avg = sum(s.valence for s in first_half) / len(first_half)
        second_avg = sum(s.valence for s in second_half) / len(second_half)
        trend = second_avg - first_avg

        if trend > 0.15:
            return "improving"
        elif trend < -0.15:
            return "declining"
        return "stable"

    def _compute_confidence(self, signals: ProsodySignals) -> float:
        """
        Confidence based on how many prosodic signals are available.

        VAD alone gives ~0.5 confidence.  Full feature set approaches 0.85.
        (With per-client learned models + outcome data, confidence can reach higher.)
        """
        available = 3  # valence, arousal, dominance are always present
        total = 15

        for attr in [
            "pitch_mean", "pitch_std", "pitch_range",
            "energy_mean", "energy_std", "speech_rate",
            "jitter", "shimmer", "hnr",
            "spectral_centroid", "spectral_rolloff", "mfcc_means",
        ]:
            if getattr(signals, attr, None) is not None:
                available += 1

        return 0.3 + 0.55 * (available / total)

    def _recommend_actions(
        self,
        signals: ProsodySignals,
        kpi: KPIDefinition,
        predicted_value: float,
    ) -> list[RecommendedAction]:
        """Suggest actions to improve the KPI based on current prosodic state."""
        actions: list[RecommendedAction] = []
        is_higher_better = kpi.direction == KPIDirection.HIGHER_IS_BETTER

        # If outcome is poor, suggest corrections
        poor_outcome = False
        if kpi.kpi_type == KPIType.SCALAR and kpi.range_min is not None and kpi.range_max is not None:
            midpoint = (kpi.range_min + kpi.range_max) / 2
            poor_outcome = (
                (predicted_value < midpoint and is_higher_better)
                or (predicted_value > midpoint and not is_higher_better)
            )
        elif kpi.kpi_type == KPIType.BINARY:
            poor_outcome = (
                (predicted_value < 0.5 and is_higher_better)
                or (predicted_value > 0.5 and not is_higher_better)
            )

        if not poor_outcome:
            return actions

        # Valence-based recommendations
        if signals.valence < -0.2:
            actions.append(RecommendedAction(
                action="Adopt a warmer, more supportive vocal tone",
                expected_impact=0.15,
                signal_target="valence",
            ))

        # Arousal-based recommendations
        if signals.arousal > 0.7:
            actions.append(RecommendedAction(
                action="Lower vocal intensity and slow pace to de-escalate",
                expected_impact=0.12,
                signal_target="arousal",
            ))
        elif signals.arousal < 0.3:
            actions.append(RecommendedAction(
                action="Increase vocal engagement and energy",
                expected_impact=0.08,
                signal_target="arousal",
            ))

        # Dominance-based recommendations
        if signals.dominance > 0.8:
            actions.append(RecommendedAction(
                action="Give the speaker more space and use active listening cues",
                expected_impact=0.08,
                signal_target="dominance",
            ))

        # Speech rate recommendations
        if signals.speech_rate is not None:
            if signals.speech_rate > 5.0:
                actions.append(RecommendedAction(
                    action="Slow down speech rate for clarity and calm",
                    expected_impact=0.1,
                    signal_target="speech_rate",
                ))

        return actions

    def _check_alert(
        self, kpi: KPIDefinition, predicted_value: float,
    ) -> Optional[KPIAlert]:
        """Check if the prediction crosses the KPI's alert threshold."""
        if kpi.alert_threshold is None:
            return None

        triggered = False
        if kpi.alert_direction == AlertDirection.BELOW:
            triggered = predicted_value < kpi.alert_threshold
        else:
            triggered = predicted_value > kpi.alert_threshold

        if not triggered:
            return None

        return KPIAlert(
            kpi_name=kpi.name,
            predicted_value=round(predicted_value, 3),
            threshold=kpi.alert_threshold,
            direction=kpi.alert_direction.value,
            message=(
                f"Predicted {kpi.name} ({predicted_value:.2f}) is "
                f"{'below' if kpi.alert_direction == AlertDirection.BELOW else 'above'} "
                f"threshold ({kpi.alert_threshold})"
            ),
        )


# Singleton
_predictor = KPIPredictor()


def get_kpi_predictor() -> KPIPredictor:
    return _predictor
