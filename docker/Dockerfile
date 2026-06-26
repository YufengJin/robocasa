# =============================================================================
# robocasa Docker image (uv-managed Python environment)
# Build from repo root:  docker build -f docker/Dockerfile .
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Pin uv project environment to /opt/venv so `uv sync` and `uv pip install` both
# target the same venv. Without this, `uv sync` creates /build/.venv which is
# separate from the /opt/venv created below — torch wheels installed via `uv pip`
# land in /opt/venv but lose sub-packages like _vendor (torch 2.4.0+cu118 bug).
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl wget tar procps bzip2 ca-certificates git build-essential cmake unzip \
    ffmpeg libgl1 libgl1-mesa-glx libglib2.0-0 \
    libx11-6 libxext6 libxrender1 libsm6 libegl1 libosmesa6 \
    lsof psmisc tmux vim x11-apps \
    && rm -rf /var/lib/apt/lists/*



# -----------------------------------------------------------------------------
# 2. uv (fast Python package manager)
# -----------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# 运行时常见把宿主机目录 bind 到 /root，会遮住镜像内 /root/.local/bin；复制 uv 到 PATH 固定位置
# （集群 apptainer run -B fakeroot:/root 时，entrypoint 的 `uv pip install -e .` 仍可用）。
RUN cp /root/.local/bin/uv /usr/local/bin/uv && chmod +x /usr/local/bin/uv
# Alternative (faster, always latest):
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# -----------------------------------------------------------------------------
# 3. Python venv via uv
# -----------------------------------------------------------------------------
RUN uv venv --python 3.10 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
# uv 把 Python 装在 /root/.local/share/uv/python/...；bind /root 时软链接失效。
# 复制到 /opt/python310 并重链，保证任何 bind 场景下 venv Python 可用。
RUN _py=$(find /root/.local/share/uv/python -maxdepth 3 -name 'python3.10' -type f 2>/dev/null | head -1) && \
    if [ -n "$_py" ]; then \
        _dir=$(dirname "$(dirname "$_py")") && \
        cp -a "$_dir" /opt/python310 && \
        ln -sf /opt/python310/bin/python3.10 /opt/venv/bin/python && \
        ln -sf /opt/python310/bin/python3.10 /opt/venv/bin/python3 && \
        ln -sf /opt/python310/bin/python3.10 /opt/venv/bin/python3.10; \
    fi



# -----------------------------------------------------------------------------
# 4. Non-Python / make-based third-party dependencies → /opt
# -----------------------------------------------------------------------------
# 
# Example:
# WORKDIR /opt/third_party
# RUN git clone https://github.com/<org>/<repo>.git && \
#     cd <repo> && git checkout <commit> && \
#     mkdir build && cd build && cmake .. && make -j$(nproc) && make install
#
# If the dep has Python bindings:
# RUN uv pip install -e /opt/third_party/<repo>

# -----------------------------------------------------------------------------
# 5. Install block — driven by InstallationPlan (<repo>/.nautilus/install_plan.json)
#    or, when no plan exists, by the legacy probe-only Option A/C heuristic.
#
# NOTE: Every `uv sync` / `uv pip install` line carries `--index-strategy
# unsafe-best-match`. uv defaults to "first-index-wins": once any extra index
# (e.g. the PyTorch CUDA wheel index) mentions a package name, uv treats that
# index as the SOLE source and refuses to consult PyPI for a better version.
# This breaks pins like `packaging==23.0` because the PyTorch index ships
# partial mirrors at different versions. `unsafe-best-match` tells uv to
# consider all indexes and pick the best match — what users intuitively
# expect. Harmless when no extra-index is configured. See:
# https://docs.astral.sh/uv/concepts/indexes/
# -----------------------------------------------------------------------------
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
RUN mkdir -p /opt/third_party && git clone https://github.com/ARISE-Initiative/robosuite.git /opt/third_party/robosuite && cd /opt/third_party/robosuite && git checkout v1.5.2 && uv pip install -e .

# 
# Slot for downstream sub-skills (benchmark / policy generators) to inject
# extra pip installs via extend_base.py. Leave the placeholder line above —
# it is the insertion anchor.

# -----------------------------------------------------------------------------
# 6. Extension point for sub-skills (benchmark / policy / robot env-generator)
# -----------------------------------------------------------------------------
# Sub-skills insert RUN lines ABOVE this marker via their extend_base.py script.
# Do NOT remove the marker line — it is how downstream tooling locates the insertion point.
# <<<EXTENSION_INSERT_ABOVE>>>

# -----------------------------------------------------------------------------
# 7. Bake robocasa source + kitchen assets into the image (self-contained)
# -----------------------------------------------------------------------------
# 与上游「源码运行时 bind」不同：为了把 ~8GB 厨房素材(inode 大户)封进 .sif（squashfs
# 单文件 = 1 inode），这里把源码连同 robocasa/models/assets 一并 COPY 进镜像并就地
# editable 安装。集群 apptainer run 时 **不要** bind /workspace/robocasa（否则会遮住烤入的
# assets）。改 robocasa 代码需重 build。本地 docker-compose 若仍 bind 源码则退回运行时下载（兼容）。
COPY . /workspace/robocasa
RUN cd /workspace/robocasa && uv pip install -e . --index-strategy unsafe-best-match
RUN cd /workspace/robocasa && python scripts/setup_macros.py 2>/dev/null || true
# assets：构建上下文已含(host 下载过)则原样烤入；否则 build 时下载（reproducible 兜底）。
RUN _assets=/workspace/robocasa/robocasa/models/assets && \
    if [ -d "$_assets/textures" ] && [ -n "$(ls -A "$_assets/textures" 2>/dev/null)" ]; then \
        echo ">> kitchen assets present in build context, baked as-is"; \
    else \
        echo ">> downloading kitchen assets at build time (~5GB)..." && \
        yes | python /workspace/robocasa/robocasa/scripts/download_kitchen_assets.py; \
    fi

# -----------------------------------------------------------------------------
# 8. Workspace + entrypoint
# -----------------------------------------------------------------------------
WORKDIR /workspace/robocasa
# headless EGL 离屏渲染(robosuite/mujoco)，run_eval 仿真必需。
ENV MUJOCO_GL=egl
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
