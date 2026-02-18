#!/usr/bin/env python3
"""
run_demo.py — 在仿真中运行 policy 并显示 GUI（不做 eval）。

与 run_eval.py 类似，通过 gRPC 连接 Policy Server 获取动作，但仅用于演示：
- 默认开启 GUI（has_renderer=True）
- 默认重置场景 10 次（--num_resets 10）
- 不写 eval 日志、不统计 success rate，不保存视频（除非指定）

用法:
    # 先启动 policy server，例如:
    python tests/test_random_policy_server.py --port 50051

    # 再运行 demo（GUI + 默认 10 次 reset）:
    python scripts/run_demo.py \
        --task_name PnPCounterToCab \
        --policy_server_addr localhost:50051

    # 指定重置次数:
    python scripts/run_demo.py --task_name PnPCounterToCab --num_resets 5
"""

import argparse
import ast
import atexit
import os
import random
import signal
import sys
import time

import cv2
import grpc
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from policy_bridge.grpc.robocasa import policy_service_pb2, policy_service_pb2_grpc  
from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)

TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}


def jpeg_encode(image: np.ndarray, quality: int = 95) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def create_robocasa_env(args, seed=None, episode_idx=None, use_gui: bool = True):
    """创建 RoboCasa 环境；use_gui=True 时开启屏幕渲染并显示 GUI。"""
    layout_and_style_ids = None
    if args.layout_and_style_ids:
        all_layout_style_ids = ast.literal_eval(args.layout_and_style_ids)
        if episode_idx is not None:
            scene_index = (episode_idx // 10) % len(all_layout_style_ids)
            layout_and_style_ids = (all_layout_style_ids[scene_index],)
        else:
            layout_and_style_ids = all_layout_style_ids

    controller_configs = load_composite_controller_config(
        controller=None,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    env_kwargs = dict(
        env_name=args.task_name,
        robots=args.robots,
        controller_configs=controller_configs,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=args.img_res,
        camera_heights=args.img_res,
        has_renderer=use_gui,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=args.obj_instance_split,
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    if use_gui:
        # # render_camera=None 使用 mjviewer 默认自由相机，支持鼠标拖拽旋转/平移视角；
        # # 若设为固定相机（如 "robot0_agentview_center"）则无法拖拽。
        # env_kwargs["render_camera"] = None
        # env_kwargs["renderer"] = "mjviewer"
        env_kwargs["render_camera"] = "robot0_agentview_left"
    env = robosuite.make(**env_kwargs)
    return env


def prepare_observation(obs, flip_images: bool = True):
    primary_img = obs.get("robot0_agentview_left_image")
    secondary_img = obs.get("robot0_agentview_right_image")
    wrist_img = obs.get("robot0_eye_in_hand_image")
    if flip_images:
        if primary_img is not None:
            primary_img = np.flipud(primary_img).copy()
        if secondary_img is not None:
            secondary_img = np.flipud(secondary_img).copy()
        if wrist_img is not None:
            wrist_img = np.flipud(wrist_img).copy()
    proprio = np.concatenate((
        obs["robot0_gripper_qpos"],
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
    ))
    return {
        "primary_image": primary_img,
        "secondary_image": secondary_img,
        "wrist_image": wrist_img,
        "proprio": proprio,
    }


def grpc_reset(stub, task_name, task_description, env):
    action_low, action_high = env.action_spec
    request = policy_service_pb2.ResetRequest(
        task_name=task_name,
        task_description=task_description,
        action_dim=action_low.shape[0],
        action_low=action_low.tolist(),
        action_high=action_high.tolist(),
    )
    response = stub.Reset(request)
    return response.success


def grpc_get_action(stub, observation, task_description, img_res):
    request = policy_service_pb2.ObservationRequest(
        primary_image=jpeg_encode(observation["primary_image"]),
        secondary_image=jpeg_encode(observation["secondary_image"]),
        wrist_image=jpeg_encode(observation["wrist_image"]),
        proprio=observation["proprio"].tolist(),
        task_description=task_description,
        image_height=img_res,
        image_width=img_res,
    )
    response = stub.GetAction(request)
    return np.array(response.action, dtype=np.float64)


def run_episode(args, env, task_description, stub, episode_idx, use_gui: bool):
    """跑一局：等物体稳定后按 max_steps 步循环取 obs -> policy -> step，并可选渲染。"""
    NUM_WAIT_STEPS = 10
    for _ in range(NUM_WAIT_STEPS):
        dummy = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy)

    max_steps = TASK_MAX_STEPS.get(args.task_name, 500)
    success = False
    episode_length = 0

    for t in range(max_steps):
        observation = prepare_observation(obs, flip_images=args.flip_images)

        start = time.time()
        action = grpc_get_action(stub, observation, task_description, args.img_res)
        if t % 50 == 0:
            print(f"  t={t}: action query {time.time() - start:.3f}s")

        # Extend 7D policy output to env.action_dim (e.g. 11 or 12 for PandaMobile)
        if action.shape[-1] == 7 and env.action_dim > 7:
            pad_dim = env.action_dim - 7
            mobile_base = np.zeros(pad_dim, dtype=np.float64)
            mobile_base[-1] = -1.0  # last dim -1 often means "hold" for mobile base
            action = np.concatenate([action, mobile_base])

        obs, reward, done, info = env.step(action)
        episode_length += 1

        if use_gui and env.sim is not None:
            env.render()

        if env._check_success():
            success = True
            print(f"  Success at t={t}!")
            break

    print(f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} (length={episode_length})")
    return success, episode_length


def parse_args():
    all_tasks = list({**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}.keys())
    parser = argparse.ArgumentParser(
        description="RoboCasa demo: 在仿真中跑 policy 并显示 GUI，不做 eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_server_addr", type=str, default="localhost:50051",
                        help="Policy gRPC 服务地址")
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy 名称（仅用于打印）")
    parser.add_argument("--task_name", type=str, default="PnPCounterToCab",
                        choices=all_tasks, help="任务名")
    parser.add_argument("--num_resets", type=int, default=10,
                        help="重置场景次数（默认 10 次）")
    parser.add_argument("--robots", type=str, default="PandaMobile")
    parser.add_argument("--img_res", type=int, default=224)
    parser.add_argument("--obj_instance_split", type=str, default="B")
    parser.add_argument("--layout_and_style_ids", type=str,
                        default="((1,1),(2,2),(4,4),(6,9),(7,10))")
    parser.add_argument("--flip_images", action="store_true", default=True)
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--no_gui", action="store_true",
                        help="关闭 GUI，仅仿真（无窗口）")
    return parser.parse_args()


def set_seed_everywhere(seed: int, deterministic: bool = True):
    if deterministic:
        os.environ["DETERMINISTIC"] = "True"
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if args.task_name not in all_tasks:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {list(all_tasks.keys())}")

    set_seed_everywhere(args.seed, deterministic=args.deterministic)
    use_gui = not args.no_gui

    print("=" * 60)
    print("RoboCasa Demo (run policy in sim, no eval)")
    print("=" * 60)
    print(f"  task_name:    {args.task_name}")
    print(f"  num_resets:   {args.num_resets}")
    print(f"  policy:       {args.policy}")
    print(f"  policy_server: {args.policy_server_addr}")
    print(f"  GUI:          {'on' if use_gui else 'off (--no_gui)'}")
    print("=" * 60)

    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(args.policy_server_addr, options=options)
    stub = policy_service_pb2_grpc.PolicyServiceStub(channel)

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        channel.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(_cleanup)

    print("Waiting for policy server ...")
    try:
        grpc.channel_ready_future(channel).result(timeout=30)
    except grpc.FutureTimeoutError:
        print("ERROR: policy server not ready within 30s")
        sys.exit(1)
    print("Policy server ready.\n")

    for ep_idx in range(args.num_resets):
        print(f"--- Reset {ep_idx + 1}/{args.num_resets} ---")
        seed = args.seed * ep_idx * 256 if args.deterministic else None
        env = create_robocasa_env(args, seed=seed, episode_idx=ep_idx, use_gui=use_gui)
        env.reset()
        task_description = env.get_ep_meta()["lang"]
        print(f"Task: {task_description}")

        ok = grpc_reset(stub, args.task_name, task_description, env)
        if not ok:
            print("  WARNING: policy server Reset returned failure")

        run_episode(args, env, task_description, stub, ep_idx, use_gui)
        env.close()

    channel.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
