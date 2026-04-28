# =========================================================
# RoboCasa Dockerfile (uv-based, GPU-ready)
# =========================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV VIRTUAL_ENV=/opt/venv
# uv sync defaults to .venv in the project dir; pin to the pre-created venv at
# /opt/venv so the source bind-mount onto /workspace/ does not hide it at runtime.
# Same layout as LIBERO/loco-mujoco. Without this line, runtime falls back to a
# torch reinstall via `uv pip install -e .` which has triggered torch wheel
# installation issues (missing torch/_vendor in 2.4.0+cu118).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH

# ---------------------------------------------------------
# System dependencies (uv cannot install these)
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
    tmux \
    vim \
    x11-apps \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install uv (copy static binary from official image)
# ---------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ---------------------------------------------------------
# Create Python 3.10 venv managed by uv
# ---------------------------------------------------------
RUN uv venv /opt/venv --python 3.10

# ---------------------------------------------------------
# Pre-install project dependencies (leverages Docker layer cache)
# Only pyproject.toml + uv.lock are copied so this layer is
# re-used unless dependencies change.
# ---------------------------------------------------------
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ---------------------------------------------------------
# Install robosuite v1.5.2 (editable) to /opt/third_party/
# ---------------------------------------------------------
RUN mkdir -p /opt/third_party \
 && git clone https://github.com/ARISE-Initiative/robosuite.git /opt/third_party/robosuite \
 && cd /opt/third_party/robosuite \
 && git checkout v1.5.2 \
 && uv pip install -e .

# ---------------------------------------------------------
# Entrypoint (robocasa itself is installed editable at container start)
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
