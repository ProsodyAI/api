"""
Health check endpoints.
"""

from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and version information.
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ProsodyAI API",
        "version": "0.1.0",
        "description": "Speech emotion recognition API",
        "docs": "/docs",
        "health": "/health",
    }
