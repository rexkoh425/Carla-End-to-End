# syntax=docker/dockerfile:1.6
# Stage 1: build the heavy base env once (environment.yml + torch/cu121 + CARLA)
FROM mambaorg/micromamba:1.5.9 AS env-base
SHELL ["/bin/bash", "-l", "-c"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121

COPY environment.yml /tmp/environment.yml
COPY CarlaLinux/extracted/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl /tmp/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl

# Heavy deps in one cached layer; requirements.txt will be installed later so this stays cached.
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    micromamba create -y -n app -f /tmp/environment.yml && \
    micromamba run -n app pip install --no-cache-dir \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
      --index-url ${TORCH_INDEX} \
      --extra-index-url https://pypi.nvidia.com && \
    micromamba run -n app pip install --no-cache-dir --no-build-isolation flash-attn==2.6.3 && \
    micromamba run -n app pip install --no-cache-dir /tmp/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl

# Stage 2: app layer installs the small/churny requirements and code
FROM env-base AS runtime
WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n app pip install --no-cache-dir -r /tmp/requirements.txt

# App code
COPY backend /app/backend
COPY WebUI/yaml_index.json /app/WebUI/yaml_index.json

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=7000

EXPOSE 7000

CMD ["micromamba", "run", "-n", "app", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7000"]
