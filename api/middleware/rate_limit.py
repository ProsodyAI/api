"""
Rate limiting middleware.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.config import settings


class InMemoryRateLimiter:
    """Simple in-memory rate limiter for development/single-instance."""
    
    def __init__(self):
        self._requests: Dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(
        self,
        key: str,
        limit: int,
        window: int,
    ) -> Tuple[bool, int, int]:
        """
        Check if a request is allowed.
        
        Args:
            key: Rate limit key (usually API key or IP)
            limit: Maximum requests per window
            window: Time window in seconds
            
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        now = time.time()
        window_start = now - window
        
        # Remove old requests
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > window_start
        ]
        
        # Check limit
        current_count = len(self._requests[key])
        remaining = max(0, limit - current_count - 1)
        reset_time = int(window_start + window)
        
        if current_count >= limit:
            return False, 0, reset_time
        
        # Record this request
        self._requests[key].append(now)
        return True, remaining, reset_time
    
    def cleanup(self):
        """Remove expired entries."""
        now = time.time()
        cutoff = now - settings.rate_limit_window
        
        keys_to_remove = []
        for key, timestamps in self._requests.items():
            self._requests[key] = [ts for ts in timestamps if ts > cutoff]
            if not self._requests[key]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._requests[key]


# Global rate limiter instance
_rate_limiter: Optional[InMemoryRateLimiter] = None


def get_rate_limiter() -> InMemoryRateLimiter:
    """Get the rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = InMemoryRateLimiter()
    return _rate_limiter


PLAN_DAILY_LIMITS = {
    "free": 100,
    "starter": 5000,
    "pro": 50000,
    "enterprise": None,  # Unlimited
}

PLAN_FEATURES = {
    "free": {"streaming": False, "verticals": False, "max_keys": 1},
    "starter": {"streaming": False, "verticals": True, "max_keys": 3},
    "pro": {"streaming": True, "verticals": True, "max_keys": 10},
    "enterprise": {"streaming": True, "verticals": True, "max_keys": 999},
}

# Cache of api_key -> plan tier (avoids DB lookup on every request)
_key_plan_cache: Dict[str, Tuple[str, float]] = {}
_CACHE_TTL = 300  # 5 minutes


def get_plan_for_key(api_key: str) -> str:
    """
    Determine plan tier for an API key.
    
    Checks cache first, then falls back to the valid keys list.
    For MVP: all keys loaded from PROSODYAI_API_KEYS env are Enterprise.
    In production: look up key in the website DB via a shared cache.
    """
    now = time.time()
    
    # Check cache
    if api_key in _key_plan_cache:
        plan, cached_at = _key_plan_cache[api_key]
        if now - cached_at < _CACHE_TTL:
            return plan
    
    # All keys in the env var are Enterprise (internal/admin keys)
    from api.middleware.auth import get_valid_keys
    if api_key in get_valid_keys():
        _key_plan_cache[api_key] = ("enterprise", now)
        return "enterprise"
    
    # Default to free
    return "free"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Plan-aware rate limiting middleware."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)
        
        # Skip non-API paths
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        api_key = request.headers.get(settings.api_key_header)
        rate_key = api_key or (request.client.host if request.client else "unknown")
        
        plan = get_plan_for_key(api_key) if api_key else "free"
        limit = PLAN_DAILY_LIMITS.get(plan)
        
        # Enterprise: no limit
        if limit is None:
            response = await call_next(request)
            response.headers["X-Plan"] = plan
            return response
        
        # Check streaming access
        if request.url.path.startswith("/v1/stream") and not PLAN_FEATURES.get(plan, {}).get("streaming"):
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": 403,
                        "message": "WebSocket streaming requires Pro or Enterprise plan.",
                        "type": "plan_limit_error",
                    }
                },
            )
        
        limiter = get_rate_limiter()
        allowed, remaining, reset_time = limiter.is_allowed(
            rate_key, limit, settings.rate_limit_window,
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": 429,
                        "message": f"Rate limit exceeded ({limit} requests/day on {plan} plan). Upgrade at https://prosodyai.app/admin/settings",
                        "type": "rate_limit_error",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "X-Plan": plan,
                    "Retry-After": str(max(0, reset_time - int(time.time()))),
                },
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        response.headers["X-Plan"] = plan
        return response
