"""
API configuration settings.
"""

import os
from functools import lru_cache
from typing import Optional, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API configuration settings loaded from environment."""

    # Database — same as website. Use DIRECT_DATABASE_URL (direct Postgres) when set, else DATABASE_URL.
    database_url: Optional[str] = Field(default=None, validation_alias="DATABASE_URL")
    direct_database_url: Optional[str] = Field(default=None, validation_alias="DIRECT_DATABASE_URL")

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080  # Cloud Run default
    debug: bool = False
    
    # Authentication
    api_key_header: str = "X-API-Key"
    api_keys_file: Optional[str] = None

    # Admin API (tenant API keys, RBAC) — required for /v1/admin/* routes
    admin_api_key: Optional[str] = None
    admin_api_key_header: str = "X-Admin-Key"
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 86400  # window in seconds (24 hours)
    
    # CORS — set in prod (e.g. PROSODYAI_CORS_ORIGINS=https://prosodyai.app,https://www.prosodyai.app)
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # Model inference (Baseten under the hood; env vars are ProsodyAI-only)
    service_timeout: float = 60.0
    service_api_key: Optional[str] = None  # Optional auth for direct service_url (legacy)

    model_id: Optional[str] = None  # PROSODYAI_MODEL_ID
    model_api_key: Optional[str] = None  # PROSODYAI_MODEL_API_KEY
    model_deployment: str = "production"  # PROSODYAI_MODEL_DEPLOYMENT

    # Optional: direct model service URL (legacy)
    service_url: str = ""

    # GCP / Vertex AI (optional)
    gcp_project_id: Optional[str] = None
    gcp_region: str = "us-central1"
    use_vertex_ai: bool = False
    vertex_endpoint_id: Optional[str] = None
    
    # Audio processing
    sample_rate: int = 16000  # Audio sample rate for feature extraction
    
    # Storage
    temp_dir: str = "/tmp/prosodyai"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    gcs_bucket: Optional[str] = None  # For large file uploads
    
    # Redis (for rate limiting in production)
    redis_url: Optional[str] = None
    
    class Config:
        env_prefix = "PROSODYAI_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
