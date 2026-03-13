# ProsodyAI API - deploy to Cloud Run (inference via Baseten; features + streaming in-repo)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    gcc python3-dev espeak-ng libespeak-ng-dev \
    && ln -sf /usr/bin/espeak-ng /usr/bin/espeak \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PORT=8080
ENV PYTHONPATH=/app

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
