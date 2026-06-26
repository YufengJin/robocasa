#!/usr/bin/env bash
# 构建 robocasa.sif：先 docker build，再从本地 docker daemon 转 squashfs sif。
# 与 docker/Dockerfile 对齐（源码+~8GB 厨房素材烤入镜像；/root 软链修复；MUJOCO_GL=egl）。
# 用法（在 benchmarks/robocasa 下）：bash apptainer/build.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-vla/robocasa:latest}"
SIF_OUT="${SIF_OUT:-/mnt/ssd2T/yjin/sif/robocasa.sif}"
# apptainer 构建缓存/临时放大盘，避免撑爆 $HOME（集群同理：指向 $WORK）
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/ssd2T/yjin/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/mnt/ssd2T/yjin/.apptainer_tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$SIF_OUT")"

cd "$REPO_ROOT"
echo ">> docker build $IMAGE_TAG (含 8GB 厨房素材，首次较久)"
docker build -f docker/Dockerfile -t "$IMAGE_TAG" .
echo ">> apptainer build $SIF_OUT  <-  docker-daemon://$IMAGE_TAG"
apptainer build --force "$SIF_OUT" "docker-daemon://$IMAGE_TAG"
echo ">> done: $SIF_OUT"
ls -lh "$SIF_OUT"
