# RoboCasa Apptainer (.sif) — 集群部署

把 robocasa 仿真环境打成单文件 `.sif`（squashfs = 1 个 inode），解 FAU NHR HPC3 的文件数配额。
源码 + ~8GB 厨房素材全部烤进镜像，集群上**不需要 bind 源码**。

## 构建（本机：docker → sif）

本机有 docker、无需在本机 root 跑：
```bash
bash apptainer/build.sh         # docker build vla/robocasa:latest → apptainer build robocasa.sif
```
产物默认 `/mnt/ssd2T/yjin/sif/robocasa.sif`（~8.5GB）。`APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` 已指向大盘，避免撑爆 `$HOME`。

## 运行（解耦评测的 sim client）

```bash
# 1) 先在 droid.sif 起策略 server（另一个终端/节点）
CKPT=/workspace/droid_policy_learning/outputs/<run>/<ts>/models/model_epoch_N.pth \
  bash ../../droid_policy_learning/apptainer/run_policy_server.sh
# 2) 跑 robocasa sim client，连 server
TASK_NAME=PnPCounterToCab NUM_TRIALS=1 bash apptainer/run_eval.sh
```
apptainer 默认共享宿主网络，server/client 走 `localhost:8765`。

## 本机实测验证（RTX4090 + apptainer 1.5.2）

完整跨容器评测跑通：`droid.sif`(server, GPU) + `robocasa.sif`(client, osmesa) → obs↔action 往返、500 步 episode 跑完、exit 0。

## 关键坑（已在 run_*.sh 里固化为 flag）

| 现象 | 原因 | 处理 |
|---|---|---|
| `--nv` 下 `import cv2`/mujoco-egl 报 `GLIBC_2.38 not found` | `--nv` 注入宿主(Ubuntu24.04 glibc2.39)GL 库，与容器(ubuntu20.04 glibc2.31)不兼容 | server 端 `--env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu`（容器库优先，torch 仍走宿主 libcuda）；client 用 osmesa 不加 `--nv` |
| `Read-only file system: ./eval_logs` | sif 是只读 squashfs | `--log_dir /tmp/...` |
| `Read-only file system: .../objects/.../*.xml` | robocasa 运行时把处理后的物体 XML 写回 assets | `--writable-tmpfs`（可写覆盖层） |
| server 报 `__command__ / 无法识别 benchmark` | droid.sif 与 robocasa.sif build 时 pin 的 `policy_websocket` 版本不同 | 两端都 `--writable-tmpfs`，靠 entrypoint 在线更到 git HEAD 对齐。**集群无外网应改为 build 时 pin 同一 commit** |
| server `Permission denied: ~/.cache/huggingface` | apptainer 默认挂宿主 `$HOME`，其 hf 目录可能 root 属主 | server `--env HF_HOME=<可写盘>` + bind |

## 集群（FAU NHR HPC3）注意
- sif、HF cache、`APPTAINER_CACHEDIR`/`TMPDIR` 都放 `$WORK`（atuin），不放 `$HOME`。
- `policy_websocket` 版本对齐别依赖在线更新（计算节点常无外网）→ 在 docker/Dockerfile 里 pin 同一 commit 后重建。
- GPU 渲染若要用 EGL（比 osmesa 快）：取决于集群 host glibc 与容器是否兼容；不兼容就继续用 osmesa。
