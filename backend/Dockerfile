# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      build-essential curl git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv and set caches
ENV PATH="/root/.local/bin:${PATH}" \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv
RUN --mount=type=cache,target=/root/.cache/uv curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app/backend

# Sync backend deps
COPY backend/pyproject.toml /app/backend/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

# Install CUDA-enabled torch stack and CARLA wheel
COPY CarlaLinux/extracted/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl /tmp/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --index-url ${TORCH_INDEX} --extra-index-url https://pypi.nvidia.com \
      --index-strategy unsafe-best-match \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 && \
    uv pip install --python /opt/venv/bin/python /tmp/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl

# App code
COPY backend /app/backend
COPY WebUI/yaml_index.json /app/WebUI/yaml_index.json

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=7000

EXPOSE 7000

CMD ["/opt/venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7000"]
