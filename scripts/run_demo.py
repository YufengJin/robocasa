#!/usr/bin/env python3
"""
run_demo.py — 在仿真中运行 policy 并显示 GUI（不做 eval）。

与 run_eval.py 类似，通过 WebSocket 连接 Policy Server 获取动作，但仅用于演示：
- 默认开启 GUI（has_renderer=True）
- 默认重置场景 10 次（--num_resets 10）
- 不写 eval 日志、不统计 success rate，不保存视频（除非指定）

用法:
    # 先启动 policy server，例如:
    python tests/test_random_policy_server.py --port 8000

    # 再运行 demo（GUI + 默认 10 次 reset）:
    python scripts/run_demo.py \
        --task_name PnPCounterToCab \
        --policy_server_addr localhost:8000

    # 指定重置次数:
    python scripts/run_demo.py --task_name PnPCounterToCab --num_resets 5

    # 关节空间控制器（cartesian_pose | joint_pos | joint_vel）:
    python scripts/run_demo.py --task_name PnPCounterToCab --arm_controller joint_pos

    # 无 GUI 模式:
    python scripts/run_demo.py --task_name PnPCounterToCab --no_gui

    # DROID 格式（OpenPI DROID policy, joint_vel）:
    python scripts/run_demo.py --droid --policy_server_addr localhost:8000 --task_name PnPCounterToCab
"""

import argparse
import atexit
import os
import signal
import sys
import time

import numpy as np

from policy_websocket import WebsocketClientPolicy
from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)
from robocasa.utils.run_utils import (
    ARM_CONTROLLER_MAP,
    create_robocasa_env,
    enable_joint_pos_observable,
    get_task_max_steps,
    pad_action_for_env,
    prepare_observation,
    prepare_observation_droid,
    set_seed_everywhere,
)


def _create_env(args, seed=None, episode_idx=None, use_gui: bool = True):
    """Create a RoboCasa environment for demo (optionally with GUI)."""
    return create_robocasa_env(
        task_name=args.task_name,
        robots=args.robots,
        arm_controller=args.arm_controller,
        img_res=args.img_res,
        obj_instance_split=args.obj_instance_split,
        layout_and_style_ids=args.layout_and_style_ids,
        seed=seed,
        episode_idx=episode_idx,
        use_gui=use_gui,
    )


def run_episode(args, env, task_description, policy, episode_idx, use_gui: bool):
    """跑一局：等物体稳定后按 max_steps 步循环取 obs -> policy -> step，并可选渲染。"""
    droid = getattr(args, "droid", False)

    NUM_WAIT_STEPS = 10
    for _ in range(NUM_WAIT_STEPS):
        dummy = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy)

    max_steps = get_task_max_steps(args.task_name, default_horizon=500)
    success = False
    episode_length = 0

    for t in range(max_steps):
        if droid:
            observation = prepare_observation_droid(
                obs, task_description, flip_images=args.flip_images, img_size=args.img_res
            )
        else:
            observation = prepare_observation(obs, flip_images=args.flip_images)
            observation["task_description"] = task_description

        start = time.time()
        result = policy.infer(observation)
        action = result["actions"]
        if hasattr(action, "ndim") and action.ndim > 1:
            action = action[0]
        if t % 50 == 0:
            print(f"  t={t}: infer {time.time() - start:.3f}s")

        action = pad_action_for_env(action, args.arm_controller, env.action_dim)

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
    parser.add_argument("--policy_server_addr", type=str, default="localhost:8000",
                        help="WebSocket policy 服务地址 host:port")
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy 名称（仅用于打印）")
    parser.add_argument("--task_name", type=str, default="PnPCounterToCab",
                        choices=all_tasks, help="任务名")
    parser.add_argument("--num_resets", type=int, default=10,
                        help="重置场景次数（默认 10 次）")
    parser.add_argument("--droid", action="store_true",
                        help="Use DROID obs format (joint_vel, OpenPI DROID policy)")
    parser.add_argument("--arm_controller", type=str, default="cartesian_pose",
                        choices=list(ARM_CONTROLLER_MAP.keys()),
                        help="Arm controller type (ignored when --droid)")
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


def main():
    args = parse_args()
    if args.droid:
        args.arm_controller = "joint_vel"

    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if args.task_name not in all_tasks:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {list(all_tasks.keys())}")

    set_seed_everywhere(args.seed, deterministic=args.deterministic)
    use_gui = not args.no_gui

    addr = args.policy_server_addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        port = int(port)
    else:
        host, port = addr, 8000

    print("=" * 60)
    print("RoboCasa Demo (run policy in sim, no eval)")
    print("=" * 60)
    print(f"  task_name:    {args.task_name}")
    print(f"  num_resets:   {args.num_resets}")
    print(f"  policy:       {args.policy}")
    print(f"  arm_controller: {args.arm_controller} ({ARM_CONTROLLER_MAP[args.arm_controller]})")
    print(f"  droid:         {args.droid}")
    print(f"  policy_server: ws://{host}:{port}")
    print(f"  GUI:          {'on' if use_gui else 'off (--no_gui)'}")
    print("=" * 60)

    policy = WebsocketClientPolicy(host=host, port=port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    env = None

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        policy.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(policy.close)

    try:
        for ep_idx in range(args.num_resets):
            print(f"\n--- Reset {ep_idx + 1}/{args.num_resets} ---")
            seed = args.seed * ep_idx * 256 if args.deterministic else None
            env = _create_env(args, seed=seed, episode_idx=ep_idx, use_gui=use_gui)
            env.reset()
            if args.droid:
                enable_joint_pos_observable(env)
            task_description = env.get_ep_meta()["lang"]
            print(f"Task: {task_description}")

            policy.reset()
            action_low, action_high = env.action_spec
            if not args.droid:
                init_obs = {
                    "action_dim": action_low.shape[0],
                    "action_low": action_low,
                    "action_high": action_high,
                    "task_name": args.task_name,
                    "task_description": task_description,
                }
                policy.infer(init_obs)

            run_episode(args, env, task_description, policy, ep_idx, use_gui)
            env.close()
            env = None
    finally:
        policy.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
