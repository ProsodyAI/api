"""
Authentication middleware for API key validation.

Validates API keys against the database first (ApiKey.keyHash → organizationId).
Falls back to env/file keys (PROSODYAI_API_KEYS, api_keys_file) for legacy or dev.
"""

from typing import Optional
import hashlib
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import settings


def _hash_key(api_key: str) -> str:
    """SHA-256 hex digest of the API key (matches dashboard/admin storage)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# API key header security scheme
api_key_header = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False,
    description="API key for authentication",
)


def _load_api_keys() -> set[str]:
    """Load valid API keys from file or environment."""
    keys = set()
    
    # Load from environment variable (comma-separated)
    env_keys = os.getenv("PROSODYAI_API_KEYS", "")
    if env_keys:
        keys.update(k.strip() for k in env_keys.split(",") if k.strip())
    
    # Load from file if specified
    if settings.api_keys_file and os.path.exists(settings.api_keys_file):
        with open(settings.api_keys_file, "r") as f:
            for line in f:
                key = line.strip()
                if key and not key.startswith("#"):
                    keys.add(key)
    
    # Development mode: allow demo key
    if settings.debug:
        keys.add("demo-api-key")
        keys.add("test-api-key")
    
    return keys


# Cache valid API keys
_valid_keys: Optional[set[str]] = None


def get_valid_keys() -> set[str]:
    """Get the set of valid API keys."""
    global _valid_keys
    if _valid_keys is None:
        _valid_keys = _load_api_keys()
    return _valid_keys


def reload_api_keys() -> None:
    """Reload API keys from source."""
    global _valid_keys
    _valid_keys = _load_api_keys()


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_keys = get_valid_keys()
    
    # Direct comparison
    if api_key in valid_keys:
        return True
    
    # Check hashed version (for storing keys securely)
    hashed = _hash_key(api_key)
    return hashed in valid_keys


async def _is_key_in_db(api_key: str) -> bool:
    """True if this key exists in ApiKey table (tenant key from dashboard/admin)."""
    try:
        from kpis import get_kpi_loader
        loader = get_kpi_loader()
        org_id = await loader.get_organization_id(_hash_key(api_key))
        return org_id is not None
    except Exception:
        return False


async def get_api_key_header(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """
    Dependency to validate API key from header.
    Checks DB first (tenant keys), then env/file. See SYSTEMS.md (multi-tenant).
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include your API key in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if await _is_key_in_db(api_key):
        return api_key
    if validate_api_key(api_key):
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def get_key_tier(api_key: str) -> str:
    """
    Get the tier/plan for an API key.
    
    Args:
        api_key: The API key
        
    Returns:
        Tier name: "free", "pro", or "enterprise"
    """
    # In production, look up key in database
    # For now, use prefix-based tiers
    if api_key.startswith("ent-"):
        return "enterprise"
    elif api_key.startswith("pro-"):
        return "pro"
    else:
        return "free"
