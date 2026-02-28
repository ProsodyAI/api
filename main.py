"""
ProsodyAI API - FastAPI application for prosody-driven KPI prediction.

This API provides:
- Audio prosody analysis with KPI outcome prediction
- Real-time streaming prosodic feedback via WebSocket
- Prosodic feature extraction endpoints
- KPI outcome feedback for closing the training loop

No emotion labels — raw prosodic signals drive all predictions.
Clients define their KPIs on the Next.js dashboard (../prosodyai-website).
"""

from contextlib import asynccontextmanager
from typing import Optional
import os

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import admin, analysis, feedback, health, features, streaming, sessions
from middleware.rate_limit import RateLimitMiddleware
from middleware.auth import get_api_key_header
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup: Load models, initialize resources
    print("Starting ProsodyAI API...")
    yield
    # Shutdown: Cleanup resources
    print("Shutting down ProsodyAI API...")


app = FastAPI(
    title="ProsodyAI API",
    description="""
    Prosody-driven KPI prediction API powered by State Space Models.
    
    ## How It Works
    
    1. **Define KPIs** on the ProsodyAI dashboard (what you care about)
    2. **Send audio** to this API
    3. **Get predictions** for your KPIs based on raw prosodic signals
    4. **Report outcomes** to close the feedback loop and improve predictions
    
    ## Features
    
    - **Prosody Analysis**: Extract pitch, energy, rhythm, voice quality features
    - **KPI Prediction**: Predict outcomes for your custom KPIs
    - **Real-time Streaming**: WebSocket-based prosodic feedback for voice agents
    - **Actionable Recommendations**: What to change to improve KPI outcomes
    
    ## Authentication
    
    All endpoints require an API key passed via the `X-API-Key` header.
    Manage API keys on the ProsodyAI dashboard.
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(
    analysis.router,
    prefix="/v1/analyze",
    tags=["Analysis"],
    dependencies=[Depends(get_api_key_header)],
)
app.include_router(
    features.router,
    prefix="/v1/features",
    tags=["Features"],
    dependencies=[Depends(get_api_key_header)],
)
app.include_router(
    streaming.router,
    prefix="/v1/stream",
    tags=["Streaming"],
)
app.include_router(
    feedback.router,
    prefix="/v1/feedback",
    tags=["Feedback"],
    dependencies=[Depends(get_api_key_header)],
)
app.include_router(
    admin.router,
    prefix="/v1/admin",
    tags=["Admin"],
)
app.include_router(
    sessions.router,
    prefix="/v1/sessions",
    tags=["Sessions"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "api_error",
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "An internal error occurred",
                "type": "internal_error",
            }
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
