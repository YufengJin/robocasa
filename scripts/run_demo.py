#!/usr/bin/env python3
"""
run_demo.py — Run policy in simulation (no eval).

Connects to a Policy Server over WebSocket for action inference.
Client sends raw robosuite obs; policy server handles all remapping.
For demo only: no eval logs or success-rate tracking. Default: headless, saves videos to demo_log/.

Usage:
    python scripts/run_demo.py --task_name PnPCounterToCab --policy_server_addr localhost:8000
    python scripts/run_demo.py --gui --task_name PnPCounterToCab --policy_server_addr localhost:8000
    # OpenVLA (7D): --arm_controller cartesian_pose
    # OpenPI (8D):  --arm_controller joint_vel
"""

import argparse
import atexit
import os
import signal
import sys
import time
from datetime import datetime

import imageio
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
    get_expected_policy_action_dim,
    get_task_max_steps,
    pad_action_for_env,
    set_seed_everywhere,
)


def _create_env(args, seed=None, episode_idx=None, use_gui: bool = False):
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


def save_rollout_video(
    primary_images,
    secondary_images,
    wrist_images,
    episode_idx,
    success,
    task_description,
    output_dir,
):
    """Save a concatenated MP4 of primary | secondary | wrist camera views."""
    os.makedirs(output_dir, exist_ok=True)
    tag = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:40]
    filename = f"episode={episode_idx}--success={success}--task={tag}.mp4"
    mp4_path = os.path.join(output_dir, filename)
    writer = imageio.get_writer(mp4_path, fps=30, format="FFMPEG", codec="libx264")
    for p, s, w in zip(primary_images, secondary_images, wrist_images):
        frame = np.concatenate([p, s, w], axis=1)
        writer.append_data(frame)
    writer.close()
    print(f"Saved rollout video: {mp4_path}")
    return mp4_path


def run_episode(args, env, task_description, policy, episode_idx, use_gui: bool,
                save_video: bool = False):
    """Run one episode: obs -> policy -> step loop with optional rendering."""
    NUM_WAIT_STEPS = 10
    for _ in range(NUM_WAIT_STEPS):
        dummy = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy)

    max_steps = get_task_max_steps(args.task_name, default_horizon=500)
    success = False
    episode_length = 0
    replay_primary, replay_secondary, replay_wrist = [], [], []

    for t in range(max_steps):
        observation = {**obs, "task_description": task_description}
        if save_video:
            p = obs["robot0_agentview_left_image"]
            s = obs["robot0_agentview_right_image"]
            w = obs["robot0_eye_in_hand_image"]
            replay_primary.append(p.copy() if hasattr(p, "copy") else p)
            replay_secondary.append(s.copy() if hasattr(s, "copy") else s)
            replay_wrist.append(w.copy() if hasattr(w, "copy") else w)

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
    return success, episode_length, replay_primary, replay_secondary, replay_wrist


def parse_args():
    all_tasks = list({**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}.keys())
    parser = argparse.ArgumentParser(
        description="RoboCasa demo: run policy in sim via WebSocket (no eval)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_server_addr", type=str, default="localhost:8000",
                        help="WebSocket policy server address host:port")
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy name (for display)")
    parser.add_argument("--task_name", type=str, default="PnPCounterToCab",
                        choices=all_tasks, help="Task name")
    parser.add_argument("--num_resets", type=int, default=10,
                        help="Number of scene resets")
    parser.add_argument("--arm_controller", type=str, default="cartesian_pose",
                        choices=list(ARM_CONTROLLER_MAP.keys()),
                        help="cartesian_pose (7D/OpenVLA) or joint_vel (8D/OpenPI)")
    parser.add_argument("--robots", type=str, default="PandaMobile")
    parser.add_argument("--img_res", type=int, default=224)
    parser.add_argument("--obj_instance_split", type=str, default="B")
    parser.add_argument("--layout_and_style_ids", type=str,
                        default="((1,1),(2,2),(4,4),(6,9),(7,10))")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--gui", action="store_true",
                        help="Enable interactive GUI rendering (default: headless no_gui)")
    parser.add_argument("--demo_log_dir", type=str, default="./demo_log",
                        help="Directory for saved videos in no_gui mode")
    return parser.parse_args()


def main():
    args = parse_args()
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if args.task_name not in all_tasks:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {list(all_tasks.keys())}")

    set_seed_everywhere(args.seed, deterministic=args.deterministic)
    use_gui = args.gui
    save_video = not use_gui

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
    print(f"  policy_server: ws://{host}:{port}")
    print(f"  GUI:          {'on (--gui)' if use_gui else 'off (no_gui, videos saved)'}")
    if not use_gui:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.demo_log_dir, f"{args.task_name}--{date_str}")
        os.makedirs(run_dir, exist_ok=True)
        args._run_dir = run_dir
        print(f"  demo_log_dir:  {run_dir}")
    print("=" * 60)

    policy = WebsocketClientPolicy(host=host, port=port)
    metadata = policy.get_server_metadata()
    print(f"Server metadata: {metadata}")
    policy_dim = metadata.get("action_dim") or metadata.get("action_dims")
    expected_dim = get_expected_policy_action_dim(args.arm_controller)
    if policy_dim is not None and int(policy_dim) != expected_dim:
        raise ValueError(
            f"arm_controller={args.arm_controller} expects policy action_dim={expected_dim}, "
            f"but server returns {policy_dim}. Use cartesian_pose for OpenVLA (7D) or joint_vel for OpenPI (8D)."
        )

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
            enable_joint_pos_observable(env)
            task_description = env.get_ep_meta()["lang"]
            print(f"Task: {task_description}")

            policy.reset()
            action_low, action_high = env.action_spec
            init_obs = {
                "action_dim": action_low.shape[0],
                "action_low": action_low,
                "action_high": action_high,
                "task_name": args.task_name,
                "task_description": task_description,
            }
            policy.infer(init_obs)

            success, ep_len, rep_p, rep_s, rep_w = run_episode(
                args, env, task_description, policy, ep_idx, use_gui, save_video=save_video,
            )
            if save_video and rep_p and rep_s and rep_w:
                save_rollout_video(
                    rep_p, rep_s, rep_w, ep_idx, success, task_description,
                    output_dir=getattr(args, "_run_dir", args.demo_log_dir),
                )
            env.close()
            env = None
    finally:
        policy.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
