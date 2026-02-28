"""
Client for the ProsodySSM model service.

Supports Baseten dedicated deployments (recommended) and optional Vertex AI.
"""

from dataclasses import dataclass
from typing import Any, Optional
import asyncio
import base64
import httpx

from config import settings


@dataclass
class ModelPrediction:
    """Raw prediction from the ProsodySSM model."""
    
    # Transcription
    text: str
    language: str
    duration: float
    word_count: int
    
    # Base emotion classification
    emotion: str
    confidence: float
    emotion_probabilities: dict[str, float]
    
    # VAD scores
    valence: float
    arousal: float
    dominance: float
    
    # Prosody markers
    pitch_trend: Optional[str] = None
    intensity: Optional[str] = None
    tempo: Optional[str] = None
    
    # Raw prosodic features (for advanced use)
    prosody_features: Optional[dict[str, float]] = None


class ModelServiceClient:
    """
    Client for a ProsodySSM model service (direct URL or Vertex AI).
    For new deployments, use Baseten via BasetenClient (get_model_client() selects it when credentials are set).
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ):
        url = (service_url or settings.service_url or "").strip().rstrip("/")
        self.service_url = url or ("http://localhost:8080" if settings.debug else "")
        self.timeout = timeout or settings.service_timeout
        self.api_key = api_key or settings.service_api_key
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            
            # Add internal service auth if configured
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                base_url=self.service_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def predict_from_file(
        self,
        file_path: str,
        language: str = "en",
    ) -> ModelPrediction:
        """
        Send audio file to model service for prediction.
        
        Args:
            file_path: Path to the audio file
            language: Language code for ASR
            
        Returns:
            ModelPrediction with emotion classification
        """
        # Read and encode file
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return await self.predict_from_base64(audio_base64, language)
    
    async def predict_from_base64(
        self,
        audio_base64: str,
        language: str = "en",
    ) -> ModelPrediction:
        """
        Send base64-encoded audio to model service.
        
        Args:
            audio_base64: Base64-encoded audio data
            language: Language code for ASR
            
        Returns:
            ModelPrediction with emotion classification
        """
        payload = {
            "audio_base64": audio_base64,
            "language": language,
            "return_features": True,  # Get raw prosody features
        }
        
        response = await self.client.post("/predict", json=payload)
        response.raise_for_status()
        
        return self._parse_response(response.json())
    
    async def predict_from_gcs(
        self,
        gcs_uri: str,
        language: str = "en",
    ) -> ModelPrediction:
        """
        Process audio directly from GCS (for large files).
        
        Args:
            gcs_uri: GCS URI (gs://bucket/path/to/audio.wav)
            language: Language code for ASR
            
        Returns:
            ModelPrediction with emotion classification
        """
        payload = {
            "gcs_uri": gcs_uri,
            "language": language,
            "return_features": True,
        }
        
        response = await self.client.post("/predict", json=payload)
        response.raise_for_status()
        
        return self._parse_response(response.json())
    
    async def health_check(self) -> bool:
        """Check if model service is healthy."""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def _parse_response(self, data: dict[str, Any]) -> ModelPrediction:
        """Parse model service response into ModelPrediction."""
        prosody = data.get("prosody", {})
        
        return ModelPrediction(
            # Transcription
            text=data.get("text", ""),
            language=data.get("language", "en"),
            duration=data.get("duration", 0.0),
            word_count=data.get("word_count", 0),
            
            # Emotion
            emotion=data.get("emotion", "neutral"),
            confidence=data.get("confidence", 0.0),
            emotion_probabilities=data.get("emotion_probabilities", {}),
            
            # VAD
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            
            # Prosody markers
            pitch_trend=prosody.get("pitch_trend"),
            intensity=prosody.get("intensity"),
            tempo=prosody.get("tempo"),
            
            # Raw features
            prosody_features=data.get("prosody_features"),
        )
    
    # Sync wrappers
    def predict_from_file_sync(
        self,
        file_path: str,
        language: str = "en",
    ) -> ModelPrediction:
        """Synchronous version of predict_from_file."""
        return asyncio.get_event_loop().run_until_complete(
            self.predict_from_file(file_path, language)
        )


class VertexAIClient(ModelServiceClient):
    """
    Client for ProsodySSM deployed on Vertex AI.
    
    Uses Vertex AI prediction endpoints with proper GCP authentication.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ):
        self.project_id = project_id or settings.gcp_project_id
        self.region = region or settings.gcp_region
        self.endpoint_id = endpoint_id or settings.vertex_endpoint_id
        
        if not all([self.project_id, self.endpoint_id]):
            raise ValueError("GCP project_id and endpoint_id required for Vertex AI")
        
        # Vertex AI endpoint URL
        service_url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/"
            f"endpoints/{self.endpoint_id}"
        )
        
        super().__init__(service_url=service_url)
        self._credentials = None
    
    async def _get_auth_token(self) -> str:
        """Get GCP auth token for Vertex AI."""
        try:
            import google.auth
            import google.auth.transport.requests
            
            if self._credentials is None:
                self._credentials, _ = google.auth.default()
            
            # Refresh if needed
            request = google.auth.transport.requests.Request()
            self._credentials.refresh(request)
            
            return self._credentials.token
        except ImportError:
            raise ImportError(
                "google-auth required for Vertex AI. "
                "Install with: pip install google-auth"
            )
    
    async def predict_from_base64(
        self,
        audio_base64: str,
        language: str = "en",
    ) -> ModelPrediction:
        """Send prediction request to Vertex AI endpoint."""
        # Get fresh auth token
        token = await self._get_auth_token()
        
        # Vertex AI expects specific format
        payload = {
            "instances": [{
                "audio_base64": audio_base64,
                "language": language,
            }]
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.service_url}:predict",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        
        # Vertex AI wraps response in predictions array
        data = response.json()
        prediction_data = data.get("predictions", [{}])[0]
        
        return self._parse_response(prediction_data)


class BasetenClient(ModelServiceClient):
    """
    Client for ProsodySSM deployed on Baseten (dedicated endpoint).

    Uses https://model-{model_id}.api.baseten.co/environments/{deployment}/predict
    with Authorization: Api-Key {key}.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
    ):
        model_id = model_id or settings.baseten_model_id
        api_key = api_key or settings.baseten_api_key
        deployment = (deployment or settings.baseten_deployment).strip("/")
        if not model_id or not api_key:
            raise ValueError("baseten_model_id and baseten_api_key required for Baseten")
        service_url = f"https://model-{model_id}.api.baseten.co/environments/{deployment}"
        super().__init__(service_url=service_url, api_key=api_key)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}",
            }
            self._client = httpx.AsyncClient(
                base_url=self.service_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    def _parse_response(self, data: dict[str, Any]) -> ModelPrediction:
        """Parse Baseten model response (emotion + VAD only; no ASR)."""
        if "error" in data:
            raise ValueError(data["error"])
        # Baseten deploy returns emotion, confidence, emotion_probabilities, valence, arousal, dominance
        return ModelPrediction(
            text="",
            language="en",
            duration=0.0,
            word_count=0,
            emotion=data.get("emotion", "neutral"),
            confidence=data.get("confidence", 0.0),
            emotion_probabilities=data.get("emotion_probabilities", {}),
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            pitch_trend=None,
            intensity=None,
            tempo=None,
            prosody_features=None,
        )


# Global model client instance
_model_client: Optional[ModelServiceClient] = None


def get_model_client() -> ModelServiceClient:
    """Get or create the model client instance. Prefers Baseten when credentials are set."""
    global _model_client

    if _model_client is None:
        use_baseten = settings.use_baseten or (
            bool(settings.baseten_model_id and settings.baseten_api_key)
        )
        if use_baseten:
            _model_client = BasetenClient()
        elif settings.use_vertex_ai:
            _model_client = VertexAIClient()
        else:
            _model_client = ModelServiceClient()

    return _model_client
