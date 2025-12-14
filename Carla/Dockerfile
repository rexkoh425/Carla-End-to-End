FROM carlasim/carla:0.9.16

# Use root for package installs (base image runs as 'carla')
USER root

# Minimal user-space GL/Vulkan tools so NVIDIA runtime mounts can be used.
# Host still needs NVIDIA driver + nvidia-container-toolkit.
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      mesa-utils \
      vulkan-tools \
      libgl1 \
      libglvnd0 \
      libvulkan1 && \
    rm -rf /var/lib/apt/lists/*

# Enable GPU + headless rendering defaults
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /workspace

# Expose CARLA RPC/streaming ports
EXPOSE 2000-2002

# Start CARLA in offscreen/quiet mode as soon as the container launches
CMD ["bash", "CarlaUE4.sh", "-RenderOffScreen", "-nosound", "-quality-level=Low", "-carla-rpc-port=2000", "-carla-streaming-port=0"]

# Drop back to default user for runtime
USER carla
