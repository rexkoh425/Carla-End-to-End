# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependencies for audio handling
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv and sync deps
RUN --mount=type=cache,target=/root/.cache/uv curl -LsSf https://astral.sh/uv/install.sh | sh
COPY pyproject.toml /app/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

ENV WHISPER_MODEL=/models/audio/whisper/ggml-base.en.bin \
    PATH="/app/.venv/bin:${PATH}"

COPY app.py .

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082"]
