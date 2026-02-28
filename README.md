# ProsodyAI API

Public REST API for ProsodyAI speech emotion recognition service.

This API authenticates clients via API key and forwards requests to the ProsodySSM model. Inference is provided by **Baseten** (Truss bundle in `deploy/`). The dashboard (Next.js) and this API share one database: the dashboard writes KPIs and API keys; this API reads them per request. See **SYSTEMS.md** for coordination and env contract.

## Architecture

```
Client (SDK/LangChain) ──▶ API (auth + verticals) ──▶ Baseten (ProsodySSM inference)
                                    │
                                    ▼
                          Vertical Mapping
                          (contact_center, healthcare, sales, etc.)
```

## Quick Start

### Running Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Set environment variables (Baseten is used when model ID and API key are set)
export PROSODYAI_DEBUG=true
export PROSODYAI_API_KEYS=demo-api-key,test-api-key
export PROSODYAI_BASETEN_MODEL_ID=your-model-id
export PROSODYAI_BASETEN_API_KEY=your-baseten-api-key

# Run the server
uvicorn api.main:app --reload --port 8000
```

### Inference backend

- **Baseten** (default when credentials are set): Set `PROSODYAI_BASETEN_MODEL_ID` and `PROSODYAI_BASETEN_API_KEY`. Predict URL format: `https://model-<MODEL_ID>.api.baseten.co/environments/production/predict`.
- **Vertex AI** (optional): Set `PROSODYAI_USE_VERTEX_AI=true` and Vertex endpoint/config.

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Endpoints

### Health

```
GET /health
GET /
```

### Analysis (requires API key)

```
POST /v1/analyze/audio     - Upload audio file
POST /v1/analyze/base64    - Send base64-encoded audio
POST /v1/analyze/url       - Analyze audio from URL
POST /v1/analyze/gcs       - Analyze audio from GCS (gs://)
```

All analysis endpoints support the `vertical` parameter for domain-specific analysis:

```bash
curl -X POST http://localhost:8000/v1/analyze/audio \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.wav" \
  -F "vertical=contact_center"
```

### Supported Verticals

| Vertical | Key Metrics |
|----------|-------------|
| `contact_center` | CSAT prediction, escalation risk, sentiment trajectory |
| `healthcare` | Depression/anxiety markers, clinical attention level |
| `sales` | Buying intent, deal probability, recommended action |
| `education` | Comprehension, engagement, pacing recommendations |
| `hr_interviews` | Confidence, authenticity, communication score |
| `media_entertainment` | Engagement value, emotional intensity |
| `finance` | Trust level, suitability concerns, decision readiness |
| `legal` | Credibility score, consistency, evasion detection |

### Feature Extraction (requires API key)

```
POST /v1/features/prosody  - Extract prosodic features
POST /v1/features/phonetic - Extract phonetic features from text
```

### Sessions (requires API key)

When `DATABASE_URL` is set and the Session table exists, analysis and streaming create sessions automatically. You can also create and attach data explicitly:

```
POST /v1/sessions                    - Create a session (returns session_id)
POST /v1/sessions/{id}/transcript    - Add transcript for a session
POST /v1/sessions/{id}/audio          - Register audio metadata (storage path, duration, format)
```

### Admin (requires admin API key)

For enterprise tenant management. Set `PROSODYAI_ADMIN_API_KEY` (or `ADMIN_API_KEY`) and send it via `X-Admin-Key` or `Authorization: Bearer <key>`:

```
POST   /v1/admin/tenants/{org_id}/api-keys        - Create API key (returns raw key once)
GET    /v1/admin/tenants/{org_id}/api-keys        - List API keys (masked)
DELETE /v1/admin/tenants/{org_id}/api-keys/{id}   - Revoke API key
GET    /v1/admin/tenants/{org_id}/users           - List users (RBAC)
GET    /v1/admin/tenants/{org_id}/roles           - List roles
POST   /v1/admin/tenants/{org_id}/users/{id}/roles - Assign roles to user
GET    /v1/admin/tenants/{org_id}/permissions    - List permissions
```

RBAC and session/transcript/audio require the multitenant schema (see repo `schema/` and `schema/README.md`).

## Authentication

All `/v1/*` endpoints require an API key in the `X-API-Key` header:

```bash
curl -X POST http://localhost:8000/v1/analyze/audio \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.wav" \
  -F "language=en"
```

## Rate Limits

| Tier | Requests/Day |
|------|--------------|
| Free | 100 |
| Pro | 10,000 |
| Enterprise | Unlimited |

Rate limit headers are included in responses:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROSODYAI_HOST` | `0.0.0.0` | Server host |
| `PROSODYAI_PORT` | `8000` | Server port |
| `PROSODYAI_DEBUG` | `false` | Enable debug mode |
| `PROSODYAI_API_KEYS` | - | Comma-separated valid API keys |
| `PROSODYAI_USE_BASETEN` | `false` | Use Baseten for inference (auto-true if Baseten credentials set) |
| `PROSODYAI_BASETEN_MODEL_ID` | - | Baseten model ID (e.g. from Truss deploy) |
| `PROSODYAI_BASETEN_API_KEY` | - | Baseten API key |
| `PROSODYAI_BASETEN_DEPLOYMENT` | `production` | Baseten deployment name |
| `PROSODYAI_SERVICE_TIMEOUT` | `60.0` | Model request timeout (seconds) |
| `PROSODYAI_USE_VERTEX_AI` | `false` | Use Vertex AI instead of Baseten |
| `PROSODYAI_VERTEX_ENDPOINT_ID` | - | Vertex AI endpoint ID |
| `PROSODYAI_MAX_FILE_SIZE` | `52428800` | Max upload size (50MB) |
| `PROSODYAI_GCS_BUCKET` | - | GCS bucket for large uploads |
| `PROSODYAI_REDIS_URL` | - | Redis URL for rate limiting |
| `DATABASE_URL` / `PROSODYAI_DATABASE_URL` | - | PostgreSQL connection string (shared with dashboard; required for KPIs, sessions, admin API keys) |
| `PROSODYAI_ADMIN_API_KEY` / `ADMIN_API_KEY` | - | Admin API key for `/v1/admin/*` (tenant API keys, RBAC). Use header `X-Admin-Key` or `Authorization: Bearer <key>`. |

## Docker

```bash
# Build
docker build -t prosodyai-api .

# Run
docker run -p 8000:8000 \
  -e PROSODYAI_API_KEYS=your-key \
  prosodyai-api
```

## Example Response

### Basic Analysis

```json
{
  "text": "I don't know what to do anymore.",
  "emotion": {
    "primary": "sad",
    "confidence": 0.78,
    "probabilities": {
      "neutral": 0.12,
      "happy": 0.02,
      "sad": 0.78,
      "angry": 0.05,
      "fearful": 0.03
    }
  },
  "valence": -0.45,
  "arousal": 0.32,
  "dominance": 0.28,
  "prosody": {
    "pitch_trend": "falling",
    "intensity": "soft",
    "tempo": "slow"
  },
  "duration": 3.5,
  "word_count": 7,
  "format": "json"
}
```

### With Vertical Analysis (contact_center)

```json
{
  "text": "I've been waiting for 30 minutes and no one has helped me!",
  "emotion": {
    "primary": "angry",
    "confidence": 0.85,
    "probabilities": {
      "neutral": 0.05,
      "angry": 0.85,
      "frustrated": 0.08,
      "sad": 0.02
    }
  },
  "valence": -0.72,
  "arousal": 0.85,
  "dominance": 0.65,
  "prosody": {
    "pitch_trend": "rising",
    "intensity": "loud",
    "tempo": "fast"
  },
  "duration": 4.2,
  "word_count": 11,
  "vertical_analysis": {
    "vertical": "contact_center",
    "state": "angry",
    "metrics": {
      "csat_predicted": 1.6,
      "sentiment_trajectory": "declining",
      "escalation_risk": "critical",
      "first_call_resolution_likely": false
    },
    "alerts": [
      {
        "metric": "escalation_risk",
        "value": "critical",
        "threshold": "high"
      }
    ]
  }
}
```

### With Vertical Analysis (healthcare)

```json
{
  "text": "I just feel so tired all the time...",
  "emotion": {
    "primary": "sad",
    "confidence": 0.72
  },
  "valence": -0.55,
  "arousal": 0.25,
  "dominance": 0.30,
  "vertical_analysis": {
    "vertical": "healthcare",
    "state": "depressed",
    "metrics": {
      "depression_markers": 0.68,
      "anxiety_markers": 0.22,
      "distress_level": "moderate",
      "clinical_attention": "monitor",
      "mental_health_screening_recommended": true
    },
    "alerts": [
      {
        "metric": "distress_level",
        "value": "moderate",
        "threshold": "moderate"
      }
    ]
  }
}
```

## License

MIT
