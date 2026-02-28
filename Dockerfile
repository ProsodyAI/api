# ProsodyAI API - deploy to Cloud Run (inference via Baseten)
# Build context = repo root. pip installs prosody-ssm from git (needs git in image).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PORT=8080
ENV PYTHONPATH=/app

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
