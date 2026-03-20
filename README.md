# RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/robocasa-banner.jpg" width="100%" />

This is the official codebase of RoboCasa, a large-scale simulation framework for training generally capable robots to perform everyday tasks. This guide contains information about installation and setup. Please refer to the following resources for additional information:

[**[Home page]**](https://robocasa.ai) &ensp; [**[Documentation]**](https://robocasa.ai/docs/introduction/overview.html) &ensp; [**[Paper]**](https://robocasa.ai/assets/robocasa_rss24.pdf)

-------
## Policy WebSocket & installation

This fork integrates **[policy-websocket](https://github.com/YufengJin/policy_websocket)** so you can run RoboCasa with **remote** policies: observations go to a policy server over WebSocket; actions return from another host/GPU. Use it to **benchmark OpenPI vs OpenVLA-OFT** (or other compatible servers) without loading models inside the sim.

| Policy stack | Repo | `scripts/run_demo.py` flags |
| ------------ | ---- | ----------------------------- |
| **OpenPI** (π₀ / π₀.₅ DROID) | [YufengJin/openpi](https://github.com/YufengJin/openpi) | `--arm_controller joint_vel` |
| **OpenVLA-OFT** | [YufengJin/openvla](https://github.com/YufengJin/openvla) | `--arm_controller cartesian_pose` |

**Workflow:** (1) install RoboCasa (**Docker recommended** below), (2) start **openpi** or **openvla** policy server, (3) run demo with `--policy_server_addr HOST:PORT` and matching `--arm_controller`. Full stack: [role-ros2](https://github.com/YufengJin/role-ros2).

```bash
# Inside environment / container, from repo root:
micromamba activate robocasa   # Docker only; skip if using conda below
python scripts/run_demo.py --arm_controller joint_vel --policy_server_addr HOST:PORT --task_name PnPCounterToCab      # OpenPI
python scripts/run_demo.py --arm_controller cartesian_pose --policy_server_addr HOST:PORT --task_name PnPCounterToCab  # OpenVLA-OFT
```

### Installation (Docker, recommended)

Requires Docker, Docker Compose, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/YufengJin/robocasa.git
cd robocasa
docker compose -f docker/docker-compose.headless.yaml up --build -d
docker exec -it robocasa_container bash
# In container: micromamba activate robocasa  (already in .bashrc)
# Entrypoint installs editable robocasa + policy-websocket deps and kitchen assets on first start.
```

GUI / MuJoCo windows: `xhost +local:docker` then `docker compose -f docker/docker-compose.x11.yaml up --build -d`. Full options: **[docker/README.md](docker/README.md)**.

### Installation (conda, optional)

Host-native setup without Docker:

1. `conda create -c conda-forge -n robocasa python=3.10` → `conda activate robocasa`
2. Clone [robosuite](https://github.com/ARISE-Initiative/robosuite) (master), `pip install -e .`
3. Clone this repo ([YufengJin/robocasa](https://github.com/YufengJin/robocasa) fork; upstream [robocasa/robocasa](https://github.com/robocasa/robocasa)), `cd robocasa`, `pip install -e .` (includes `policy-websocket` via `setup.py`)
4. `python robocasa/scripts/download_kitchen_assets.py` (~5GB), `python robocasa/scripts/setup_macros.py`
5. Optional: `pip install pre-commit; pre-commit install`

-------
## Latest updates
* [10/31/2024] **v0.2**: using RoboSuite `v1.5` as the backend, with improved support for custom robot composition, composite controllers, more teleoperation devices, photo-realistic rendering.

-------
## Quick start
**(Mac users: for these scripts, prepend the "python" command with "mj": `mjpython ...`)**

### Explore kitchen layouts and styles
Explore kitchen layouts (G-shaped, U-shaped, etc) and kitchen styles (mediterranean, industrial, etc):
```
python -m robocasa.demos.demo_kitchen_scenes
```

### Play back sample demonstrations of tasks
Select a task and play back a sample demonstration for the selected task:
```
python -m robocasa.demos.demo_tasks
```

### Explore library of 2500+ objects
View and interact with both human-designed and AI-generated objects:
```
python -m robocasa.demos.demo_objects
```
Note: by default this demo shows objaverse objects. To view AI-generated objects, add the flag `--obj_types aigen`.

### Teleoperate the robot
Control the robot directly, either through a keyboard controller or spacemouse. This script renders the robot semi-translucent in order to minimize occlusions and enable better visibility.
```
python -m robocasa.demos.demo_teleop
```
Note: If using spacemouse: you may need to modify the product ID to your appropriate model, setting `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`.

-------
## Tasks, datasets, policy learning, and additional use cases
Please refer to the [documentation page](https://robocasa.ai/docs/introduction/overview.html) for information about tasks and assets, downloading datasets, policy learning, API docs, and more.

-------
## License
Code: [MIT License](https://opensource.org/license/mit)

Assets and Datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

-------
## Citation
```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems},
  year={2024}
}
```
