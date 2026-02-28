"""
Pydantic schemas for API request/response models.

The API works with raw prosodic signals — NOT emotion labels.
Pipeline: audio → prosodic features → KPI predictions.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl


# ==============================================================================
# Prosodic Signal Schemas
# ==============================================================================


class ProsodyMarkers(BaseModel):
    """High-level prosodic markers for a segment."""

    pitch_trend: Optional[str] = Field(None, description="Pitch trend: rising, falling, flat, varied")
    intensity: Optional[str] = Field(None, description="Intensity: soft, normal, loud, emphasized")
    tempo: Optional[str] = Field(None, description="Tempo: slow, normal, fast")


class ProsodyFeatures(BaseModel):
    """Raw prosodic feature vector — the universal signal."""

    # Core VAD dimensions
    valence: float = Field(..., ge=-1.0, le=1.0, description="Vocal tone: -1 (negative) to +1 (positive)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Activation: 0 (calm) to 1 (activated)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Assertiveness: 0 (submissive) to 1 (dominant)")

    # Prosodic markers
    prosody: Optional[ProsodyMarkers] = Field(None, description="Categorical prosodic markers")

    # Detailed features (when available)
    pitch_mean: Optional[float] = Field(None, description="Mean fundamental frequency (Hz)")
    pitch_std: Optional[float] = Field(None, description="F0 standard deviation")
    pitch_range: Optional[float] = Field(None, description="F0 range (Hz)")
    energy_mean: Optional[float] = Field(None, description="Mean RMS energy")
    energy_std: Optional[float] = Field(None, description="Energy standard deviation")
    speech_rate: Optional[float] = Field(None, description="Speech rate (syllables/sec)")
    jitter: Optional[float] = Field(None, description="Pitch perturbation")
    shimmer: Optional[float] = Field(None, description="Amplitude perturbation")
    hnr: Optional[float] = Field(None, description="Harmonics-to-Noise Ratio (dB)")
    spectral_centroid: Optional[float] = Field(None, description="Spectral centroid (Hz)")
    spectral_rolloff: Optional[float] = Field(None, description="Spectral rolloff (Hz)")


# ==============================================================================
# KPI Prediction Schemas
# ==============================================================================


class KPIImpactFactor(BaseModel):
    """A prosodic signal driving a KPI prediction."""

    signal: str = Field(..., description="Prosodic signal name (e.g., valence, speech_rate)")
    value: float = Field(..., description="Current signal value")
    impact: float = Field(..., description="Impact on KPI (signed)")
    description: str = Field(..., description="Human-readable explanation")


class KPIRecommendedAction(BaseModel):
    """Actionable recommendation to improve a KPI."""

    action: str = Field(..., description="What to do")
    expected_impact: float = Field(..., description="Estimated improvement")
    signal_target: str = Field(..., description="Which prosodic signal this targets")


class KPIAlertResult(BaseModel):
    """Alert fired when a predicted KPI crosses its threshold."""

    threshold: float = Field(..., description="Configured threshold")
    direction: str = Field(..., description="Alert direction: above or below")
    message: str = Field(..., description="Human-readable alert message")


class KPIPredictionResult(BaseModel):
    """Prediction for a single client-defined KPI."""

    kpi_id: str = Field(..., description="KPI identifier")
    kpi_name: str = Field(..., description="KPI display name")
    kpi_type: str = Field(..., description="KPI type: SCALAR, BINARY, CATEGORICAL")
    predicted_value: Any = Field(..., description="Predicted outcome value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    trajectory: Optional[str] = Field(None, description="Trend: improving, stable, declining")
    impact_factors: Optional[list[KPIImpactFactor]] = Field(None, description="Signals driving this prediction")
    recommended_actions: Optional[list[KPIRecommendedAction]] = Field(None, description="Actions to improve this KPI")
    alert: Optional[KPIAlertResult] = Field(None, description="Alert if threshold crossed")


# ==============================================================================
# Analysis Response
# ==============================================================================


class AnalysisResponse(BaseModel):
    """
    Response from analysis endpoints.

    Returns prosodic features and KPI predictions.  No emotion labels.
    """

    prediction_id: str = Field(..., description="Unique prediction ID for linking feedback/outcomes")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    text: str = Field(..., description="Transcribed text")

    # Prosodic signals — the core output
    prosody: ProsodyFeatures = Field(..., description="Raw prosodic feature vector")

    # Audio metadata
    duration: float = Field(default=0.0, ge=0.0, description="Audio duration in seconds")
    word_count: int = Field(default=0, ge=0, description="Number of words")

    # KPI predictions (populated when client has KPIs configured)
    kpi_predictions: Optional[list[KPIPredictionResult]] = Field(
        None,
        description="Predicted outcomes for client-defined KPIs",
    )

    # Aggregated alerts from all KPI predictions
    alerts: Optional[list[KPIAlertResult]] = Field(
        None,
        description="Alerts from KPI predictions that crossed thresholds",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_abc123",
                "session_id": "sess_xyz789",
                "text": "I've been waiting for 30 minutes and nobody has helped me!",
                "prosody": {
                    "valence": -0.72,
                    "arousal": 0.85,
                    "dominance": 0.65,
                    "prosody": {
                        "pitch_trend": "rising",
                        "intensity": "loud",
                        "tempo": "fast",
                    },
                    "speech_rate": 5.8,
                    "jitter": 0.03,
                },
                "duration": 4.2,
                "word_count": 11,
                "kpi_predictions": [
                    {
                        "kpi_id": "kpi_csat",
                        "kpi_name": "customer_satisfaction",
                        "kpi_type": "SCALAR",
                        "predicted_value": 1.8,
                        "confidence": 0.72,
                        "trajectory": "declining",
                        "impact_factors": [
                            {
                                "signal": "valence",
                                "value": -0.72,
                                "impact": -0.6,
                                "description": "Negative vocal tone is hurting this KPI",
                            },
                        ],
                        "recommended_actions": [
                            {
                                "action": "Adopt a warmer, more supportive vocal tone",
                                "expected_impact": 0.15,
                                "signal_target": "valence",
                            },
                        ],
                        "alert": {
                            "threshold": 2.0,
                            "direction": "BELOW",
                            "message": "Predicted customer_satisfaction (1.80) is below threshold (2.0)",
                        },
                    }
                ],
                "alerts": [
                    {
                        "threshold": 2.0,
                        "direction": "BELOW",
                        "message": "Predicted customer_satisfaction (1.80) is below threshold (2.0)",
                    }
                ],
            }
        }


# ==============================================================================
# Analysis Request
# ==============================================================================


class AnalysisRequest(BaseModel):
    """Request body for analysis endpoints."""

    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio data")
    audio_url: Optional[HttpUrl] = Field(None, description="URL to audio file")
    language: str = Field(default="en", description="Language code for ASR")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation tracking. Enables trajectory and history-aware predictions.",
    )


# ==============================================================================
# Feedback / Outcome Schemas
# ==============================================================================


class KPIOutcomeEntry(BaseModel):
    """Actual outcome value for a single KPI."""

    kpi_id: str = Field(..., description="KPI identifier (from the dashboard)")
    scalar_value: Optional[float] = Field(None, description="Actual scalar value (for SCALAR KPIs)")
    boolean_value: Optional[bool] = Field(None, description="Actual boolean value (for BINARY KPIs)")
    category_value: Optional[str] = Field(None, description="Actual category (for CATEGORICAL KPIs)")


class SessionOutcomeRequest(BaseModel):
    """
    Report actual KPI outcomes after a conversation ends.

    This is the ground truth that closes the training feedback loop.
    The system learns which prosodic patterns predict which KPI
    outcomes for each client.
    """

    session_id: str = Field(..., description="Session ID of the conversation")
    outcomes: list[KPIOutcomeEntry] = Field(
        ..., min_length=1, description="Actual KPI outcome values",
    )
    notes: Optional[str] = Field(None, description="Optional notes about the outcome")


class FeedbackCorrectionRequest(BaseModel):
    """Human correction of a prediction."""

    prediction_id: str = Field(..., description="ID of the prediction to correct")
    corrected_valence: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Corrected valence")
    corrected_arousal: Optional[float] = Field(None, ge=0.0, le=1.0, description="Corrected arousal")
    corrected_dominance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Corrected dominance")
    notes: Optional[str] = Field(None, description="Optional reviewer notes")


# ==============================================================================
# Prosody Feature Extraction Response
# ==============================================================================


class ProsodyFeaturesResponse(BaseModel):
    """Response from prosody feature extraction endpoint."""

    f0_mean: float = Field(..., description="Mean fundamental frequency (Hz)")
    f0_std: float = Field(..., description="F0 standard deviation")
    f0_min: float = Field(..., description="Minimum F0")
    f0_max: float = Field(..., description="Maximum F0")
    f0_range: float = Field(..., description="F0 range (max - min)")
    energy_mean: float = Field(..., description="Mean RMS energy")
    energy_std: float = Field(..., description="Energy standard deviation")
    jitter: float = Field(..., description="Pitch perturbation (jitter)")
    shimmer: float = Field(..., description="Amplitude perturbation (shimmer)")
    hnr: float = Field(..., description="Harmonics-to-Noise Ratio (dB)")
    speech_rate: float = Field(..., description="Speech rate (syllables/second)")
    pause_rate: float = Field(..., description="Pause rate (pauses/second)")
    pause_duration_mean: float = Field(..., description="Mean pause duration (seconds)")
    spectral_centroid_mean: float = Field(..., description="Mean spectral centroid (Hz)")
    spectral_rolloff_mean: float = Field(..., description="Mean spectral rolloff (Hz)")
    mfcc_means: list[float] = Field(..., description="Mean MFCC coefficients")


class PhoneticFeaturesResponse(BaseModel):
    """Response from phonetic feature extraction."""

    phonemes: list[str] = Field(..., description="List of phonemes")
    vowel_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of vowels")
    consonant_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of consonants")
    stressed_syllable_count: int = Field(..., ge=0, description="Number of stressed syllables")
    phoneme_count: int = Field(..., ge=0, description="Total phoneme count")


# ==============================================================================
# Error
# ==============================================================================


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: dict = Field(..., description="Error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": 401,
                    "message": "Invalid API key",
                    "type": "api_error",
                }
            }
        }
