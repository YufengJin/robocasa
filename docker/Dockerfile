# =========================================================
# RoboCasa Dockerfile (Micromamba-based, GPU-ready)
# =========================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# ---------------------------------------------------------
# System dependencies (SSL is CRITICAL)
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
# Set up workspace directory
# ---------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------
# Install robosuite v1.5.2 (required dependency)
# Clone to /workspace/robosuite and keep source code
# ---------------------------------------------------------
RUN micromamba run -n robocasa bash -c "\
    git clone https://github.com/ARISE-Initiative/robosuite.git /workspace/robosuite && \
    cd /workspace/robosuite && \
    git checkout v1.5.2 && \
    pip install -e ."

# ---------------------------------------------------------
# Clone and install RoboCasa
# Clone to /workspace/robocasa and keep source code
# Note: robocasa requires mujoco==3.2.6, so we install it after robosuite
# ---------------------------------------------------------
RUN micromamba run -n robocasa bash -c "\
    git clone https://github.com/robocasa/robocasa.git /workspace/robocasa && \
    cd /workspace/robocasa && \
    pip install -e ."

# ---------------------------------------------------------
# Optional: Install pre-commit and set up code formatter
# ---------------------------------------------------------
# RUN micromamba run -n robocasa bash -c "\
#     cd /workspace/robocasa && \
#     pip install pre-commit && \
#     pre-commit install"

# ---------------------------------------------------------
# Download kitchen assets and set up macros
# Use yes command to provide non-interactive input for all prompts
# ---------------------------------------------------------
RUN yes | micromamba run -n robocasa python /workspace/robocasa/robocasa/scripts/download_kitchen_assets.py
RUN micromamba run -n robocasa python /workspace/robocasa/robocasa/scripts/setup_macros.py

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
