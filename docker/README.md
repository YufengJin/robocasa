# RoboCasa Docker Guide

This document describes how to build and run RoboCasa containers.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime configured (for GPU support)
- For X11 mode: X11 server running on the host

## Build Image

From the project root:

```bash
cd docker
docker-compose -f docker-compose.x11.yaml build
# or
docker-compose -f docker-compose.headless.yaml build
```

With custom image name:

```bash
IMAGE=robocasa:custom docker-compose -f docker-compose.x11.yaml build
```

## Start Container

### X11 mode (GUI display)

For visualization:

```bash
cd docker
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up -d
```

Or foreground:

```bash
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up
```

### Headless mode

For training, batch evaluation, or other non-GUI use:

```bash
cd docker
docker-compose -f docker-compose.headless.yaml up -d
```

Or foreground:

```bash
docker-compose -f docker-compose.headless.yaml up
```

## Attach to Container

```bash
docker exec -it robocasa_container bash
```

## Stop Container

```bash
cd docker
docker-compose -f docker-compose.x11.yaml down
# or
docker-compose -f docker-compose.headless.yaml down
```

## View Logs

```bash
docker logs robocasa_container
# or follow
docker logs -f robocasa_container
```

## Configuration

### Container name
- Fixed name: `robocasa_container`

### GPU
- Uses all available NVIDIA GPUs by default
- Set `GPU` env var to override (default: `all`)

### Working directory
- Container workdir: `/workspace`

### Network
- Uses `host` network mode

### Environment variables
- **DISPLAY** (X11 only): X11 display
- **GPU**: GPU selection (default: `all`)

## Troubleshooting

### X11 permission denied

```bash
xhost +local:docker
```

### GPU not detected

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Container name conflict

```bash
docker stop robocasa_container
docker rm robocasa_container
```

## Example Usage

Inside the container:

```bash
micromamba activate robocasa

# Demos
python -m robocasa.demos.demo_kitchen_scenes
python -m robocasa.demos.demo_tasks
python -m robocasa.demos.demo_objects
python -m robocasa.demos.demo_teleop

# Download assets (if not done at build)
python robocasa/scripts/download_kitchen_assets.py

# Download datasets
python robocasa/scripts/download_datasets.py --ds_types human_im

# Run eval (start policy server first in another terminal)
python tests/test_random_policy_server.py --port 8000
python scripts/run_eval.py --task_name PnPCounterToCab --policy_server_addr localhost:8000
```

## Notes

- X11 mode requires a running X server and `xhost` access for Docker
- First build can take a while (dependencies and assets)
- `/workspace` contains robocasa and robosuite source
- Add volume mounts in docker-compose for persistent data
