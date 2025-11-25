# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencies for audio handling
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

ENV WHISPER_MODEL=/models/audio/whisper/ggml-base.en.bin

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082"]
