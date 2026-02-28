# ProsodyAI API - deploy to Cloud Run (inference via Baseten)
# Build context = repo root. Standalone: pip installs prosody-ssm from git.
FROM python:3.11-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PORT=8080
ENV PYTHONPATH=/app

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
