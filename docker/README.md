# RoboCasa Docker 使用指南

本文档介绍如何使用 Docker 构建和运行 RoboCasa 容器。

## 前提条件

- Docker 和 Docker Compose 已安装
- NVIDIA Docker 运行时已配置（用于 GPU 支持）
- 对于 X11 模式：X11 服务器正在运行

## 构建镜像

从项目根目录运行以下命令构建 Docker 镜像：

```bash
cd docker
docker-compose -f docker-compose.x11.yaml build
# 或者
docker-compose -f docker-compose.headless.yaml build
```

或者使用自定义镜像名称：

```bash
IMAGE=robocasa:custom docker-compose -f docker-compose.x11.yaml build
```

## 启动容器

### X11 模式（支持 GUI 显示）

适用于需要可视化界面的场景：

```bash
cd docker
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up -d
```

或者前台运行：

```bash
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up
```

### 无头模式（Headless）

适用于不需要 GUI 的场景（如训练、数据处理等）：

```bash
cd docker
docker-compose -f docker-compose.headless.yaml up -d
```

或者前台运行：

```bash
docker-compose -f docker-compose.headless.yaml up
```

## 进入容器

容器启动后，可以使用以下命令进入容器：

```bash
docker exec -it robocasa_container bash
```

## 停止容器

```bash
cd docker
docker-compose -f docker-compose.x11.yaml down
# 或者
docker-compose -f docker-compose.headless.yaml down
```

## 查看容器日志

```bash
docker logs robocasa_container
# 或者实时查看
docker logs -f robocasa_container
```

## 容器配置说明

### 容器名称
- 固定容器名称：`robocasa_container`

### GPU 配置
- 自动检测并使用所有可用的 NVIDIA GPU
- 使用 `GPU` 环境变量可以控制 GPU 使用（默认为 `all`）

### 工作目录
- 容器内工作目录：`/workspace`

### 网络模式
- 使用 `host` 网络模式，容器与主机共享网络

### 环境变量
- **DISPLAY**（仅 X11 模式）：X11 显示设置
- **GPU**：GPU 配置（默认为 `all`）

## 常见问题

### X11 权限错误

如果遇到 X11 权限问题，运行：

```bash
xhost +local:docker
```

### GPU 不可用

确保已安装并配置 NVIDIA Docker 运行时：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### 容器已存在

如果容器名称冲突，先停止并删除现有容器：

```bash
docker stop robocasa_container
docker rm robocasa_container
```

## 示例使用

进入容器后，可以运行 RoboCasa 的各种功能：

```bash
# 激活环境（会自动激活）
micromamba activate robocasa

# 运行演示
python -m robocasa.demos.demo_kitchen_scenes
python -m robocasa.demos.demo_tasks
python -m robocasa.demos.demo_objects
python -m robocasa.demos.demo_teleop

# 下载资源（如果未在构建时下载）
python robocasa/scripts/download_kitchen_assets.py

# 下载数据集
python robocasa/scripts/download_datasets.py --ds_types human_im
```

## 注意事项

- X11 模式下，确保 X11 服务器正在运行且允许 Docker 连接
- 首次构建可能需要较长时间（需要下载依赖和资源）
- 容器内的 `/workspace` 目录包含 robocasa 和 robosuite 的源代码
- 如需持久化数据，可以添加卷映射到 docker-compose 文件
