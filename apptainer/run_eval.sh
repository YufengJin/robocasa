#!/usr/bin/env bash
# 在 robocasa.sif 内跑 run_eval（websocket sim client，连 policy server）。
# 解耦评测：策略 server 在 droid.sif 里跑（见 apptainer/run_policy_server.sh），
# 本脚本只跑仿真 + 通过 localhost 向 server 取 action。
#
# 已验证的关键 flag（本机 RTX4090 + apptainer 1.5.2 实测）：
#   --writable-tmpfs  : sif 只读，robocasa 运行时要把处理后的物体 XML 写回 assets 目录 → 需可写覆盖层。
#   MUJOCO_GL=osmesa  : 用 CPU 软件渲染（libosmesa6 已装），避开 apptainer --nv 注入宿主 GL 库
#                       与容器旧 glibc 的冲突（host Ubuntu24.04 glibc2.39 GL 库 vs 容器 ubuntu20.04 glibc2.31）。
#                       sim client 不需要 GPU（策略推理在 server 端），故不加 --nv。
#   --log_dir /tmp/...: sif 只读，run_eval 默认在 cwd 建 eval_logs 会失败 → 指到可写的 /tmp。
# 源码+素材已烤入镜像，无需 bind 源码。
set -euo pipefail
SIF="${ROBOCASA_SIF:-/mnt/ssd2T/yjin/sif/robocasa.sif}"
POLICY_ADDR="${POLICY_SERVER_ADDR:-localhost:8765}"
TASK="${TASK_NAME:-PnPCounterToCab}"
NTRIALS="${NUM_TRIALS:-1}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/ssd2T/yjin/.apptainer_cache}"

# 用 apptainer run（触发 entrypoint：editable 安装 + 对齐 policy_websocket 到与 server 一致的版本）。
exec apptainer run --writable-tmpfs --env MUJOCO_GL=osmesa "$SIF" \
  bash -lc "cd /workspace/robocasa && python scripts/run_eval.py \
    --task_name '${TASK}' --policy_server_addr '${POLICY_ADDR}' \
    --num_trials '${NTRIALS}' --no_save_video --log_dir /tmp/robocasa_eval $*"
