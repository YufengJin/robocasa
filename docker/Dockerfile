# =========================================================
# RoboCasa Dockerfile (Micromamba-based, GPU-ready)
# =========================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# ---------------------------------------------------------
# System dependencies (SSL is CRITICAL) + debug tools
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libegl1 \
    libosmesa6 \
    ffmpeg \
    lsof \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install Micromamba
# ---------------------------------------------------------
RUN curl -Ls https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 -o /usr/bin/micromamba \
 && chmod +x /usr/bin/micromamba

# ---------------------------------------------------------
# Create conda environment
# ---------------------------------------------------------
RUN micromamba create -y -n robocasa python=3.10 -c conda-forge \
 && micromamba clean -a -y

# ---------------------------------------------------------
# Configure shell hook for micromamba
# ---------------------------------------------------------
RUN /bin/bash -c "micromamba shell init -s bash" \
 && echo "micromamba activate robocasa" >> /root/.bashrc

# ---------------------------------------------------------
# Upgrade pip
# ---------------------------------------------------------
RUN micromamba run -n robocasa python -m pip install --upgrade pip

# ---------------------------------------------------------
# Install PyTorch (CUDA 11.8)
# ---------------------------------------------------------
RUN micromamba run -n robocasa python -m pip install torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------
# Install gRPC and protobuf (policy service)
# ---------------------------------------------------------
RUN micromamba run -n robocasa python -m pip install \
    "grpcio>=1.60.0" \
    "grpcio-tools>=1.60.0" \
    "protobuf>=5.29.0"

# ---------------------------------------------------------
# Install robosuite v1.5.2 to /opt/third_party/ (required dependency)
# ---------------------------------------------------------
RUN mkdir -p /opt/third_party && \
    micromamba run -n robocasa bash -c "\
    git clone https://github.com/ARISE-Initiative/robosuite.git /opt/third_party/robosuite && \
    cd /opt/third_party/robosuite && \
    git checkout v1.5.2 && \
    pip install -e ."

# ---------------------------------------------------------
# Set up workspace directory (robocasa installed in entrypoint for in-container dev)
# ---------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
