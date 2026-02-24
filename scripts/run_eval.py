#!/usr/bin/env python3
"""
run_eval.py — RoboCasa evaluation client (WebSocket).

Runs the RoboCasa simulation loop and delegates action inference to a remote
Policy Server over WebSocket. Client sends raw robosuite obs; policy server handles all remapping.
Logs and videos are written to: <log_dir>/<task_name>--<YYYYMMDD_HHMMSS>/

Usage:
    python scripts/run_eval.py --task_name PnPCounterToCab --policy_server_addr localhost:8000
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


def log(msg: str, log_file=None):
    """Print a message and optionally write it to a log file."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _create_env(args, seed=None, episode_idx=None):
    """Create a RoboCasa environment for eval (no GUI)."""
    return create_robocasa_env(
        task_name=args.task_name,
        robots=args.robots,
        arm_controller=args.arm_controller,
        img_res=args.img_res,
        obj_instance_split=args.obj_instance_split,
        layout_and_style_ids=args.layout_and_style_ids,
        seed=seed,
        episode_idx=episode_idx,
        use_gui=False,
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


def run_episode(args, env, task_description, policy, episode_idx, log_file=None):
    """Run a single evaluation episode via WebSocket policy.infer()."""
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
        query_time = time.time() - start

        if t % 50 == 0:
            log(f"  t={t}: infer {query_time:.3f}s, action[:4]={action[:4]}", log_file)

        action = pad_action_for_env(action, args.arm_controller, env.action_dim)

        obs, reward, done, info = env.step(action)
        episode_length += 1

        if env._check_success():
            success = True
            log(f"  Success at t={t}!", log_file)
            break

    log(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} "
        f"(length={episode_length})",
        log_file,
    )
    return success, episode_length, replay_primary, replay_secondary, replay_wrist


def run_task(args, policy, log_file=None):
    """Evaluate a task over multiple episodes and report success rate."""
    log(f"\nEvaluating task: {args.task_name}", log_file)

    successes = []
    lengths = []

    for ep_idx in range(args.num_trials):
        log(f"\n--- Episode {ep_idx + 1}/{args.num_trials} ---", log_file)

        seed = args.seed * ep_idx * 256 if args.deterministic else None
        env = _create_env(args, seed=seed, episode_idx=ep_idx)
        env.reset()
        enable_joint_pos_observable(env)

        task_description = env.get_ep_meta()["lang"]
        log(f"Task description: {task_description}", log_file)

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

        success, length, rep_p, rep_s, rep_w = run_episode(
            args, env, task_description, policy, ep_idx, log_file
        )
        successes.append(success)
        lengths.append(length)
        if args.save_video:
            save_rollout_video(
                rep_p, rep_s, rep_w,
                ep_idx, success, task_description,
                output_dir=args.log_dir,
            )

        env.close()

        sr = sum(successes) / len(successes) * 100
        log(f"Running success rate: {sum(successes)}/{len(successes)} ({sr:.1f}%)", log_file)

    success_rate = np.mean(successes)
    avg_length = np.mean(lengths)
    log("\n" + "=" * 60, log_file)
    log("FINAL RESULTS", log_file)
    log("=" * 60, log_file)
    log(f"Policy:           {args.policy}", log_file)
    log(f"Task:             {args.task_name}", log_file)
    log(f"Success rate:     {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log(f"Avg ep length:    {avg_length:.1f}", log_file)
    log(f"Total episodes:   {len(successes)}", log_file)
    log(f"Total successes:  {sum(successes)}", log_file)
    log("=" * 60, log_file)
    return success_rate


def parse_args():
    all_tasks = list({**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}.keys())

    parser = argparse.ArgumentParser(
        description="RoboCasa WebSocket evaluation client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_server_addr", type=str, default="localhost:8000",
                        help="Address of the WebSocket policy server (host:port)")
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy name for logging (e.g. randomPolicy, cosmos)")
    parser.add_argument("--task_name", type=str, default="PnPCounterToCab",
                        choices=all_tasks,
                        help="RoboCasa task name")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Number of evaluation episodes per task")
    parser.add_argument("--arm_controller", type=str, default="cartesian_pose",
                        choices=list(ARM_CONTROLLER_MAP.keys()),
                        help="cartesian_pose (7D/OpenVLA) or joint_vel (8D/OpenPI)")
    parser.add_argument("--robots", type=str, default="PandaMobile",
                        help="Robot type")
    parser.add_argument("--img_res", type=int, default=224,
                        help="Camera image resolution (square)")
    parser.add_argument("--obj_instance_split", type=str, default="B",
                        help="Object instance split (B = held-out test)")
    parser.add_argument("--layout_and_style_ids", type=str,
                        default="((1,1),(2,2),(4,4),(6,9),(7,10))",
                        help="Layout and style IDs for scene selection")
    parser.add_argument("--seed", type=int, default=195,
                        help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic seeding")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--log_dir", type=str, default="./eval_logs",
                        help="Directory for logs and rollout videos")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save rollout videos")
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate task
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if args.task_name not in all_tasks:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {list(all_tasks.keys())}")

    set_seed_everywhere(args.seed, deterministic=args.deterministic)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, f"{args.task_name}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)
    args.log_dir = run_dir

    log_path = os.path.join(run_dir, "eval.log")
    log_file = open(log_path, "w")
    log("=" * 60, log_file)
    log("RoboCasa WebSocket Eval Run", log_file)
    log("=" * 60, log_file)
    log(f"  policy:           {args.policy}", log_file)
    log(f"  task_name:         {args.task_name}", log_file)
    log(f"  num_trials:        {args.num_trials}", log_file)
    log(f"  seed:              {args.seed}", log_file)
    log(f"  deterministic:     {args.deterministic}", log_file)
    log(f"  policy_server:     {args.policy_server_addr}", log_file)
    log(f"  log_dir (run_dir): {run_dir}", log_file)
    log(f"  img_res:           {args.img_res}", log_file)
    log(f"  robots:            {args.robots}", log_file)
    log(f"  arm_controller:    {args.arm_controller} ({ARM_CONTROLLER_MAP[args.arm_controller]})", log_file)
    log(f"  obj_instance_split: {args.obj_instance_split}", log_file)
    log(f"  layout_and_style_ids: {args.layout_and_style_ids}", log_file)
    log("=" * 60, log_file)
    log("", log_file)

    addr = args.policy_server_addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        port = int(port)
    else:
        host, port = addr, 8000

    log(f"Connecting to policy server at ws://{host}:{port} ...", log_file)
    policy = WebsocketClientPolicy(host=host, port=port)
    metadata = policy.get_server_metadata()
    log(f"Server metadata: {metadata}", log_file)
    policy_dim = metadata.get("action_dim") or metadata.get("action_dims")
    expected_dim = get_expected_policy_action_dim(args.arm_controller)
    if policy_dim is not None and int(policy_dim) != expected_dim:
        raise ValueError(
            f"arm_controller={args.arm_controller} expects policy action_dim={expected_dim}, "
            f"but server returns {policy_dim}. Use cartesian_pose for OpenVLA (7D) or joint_vel for OpenPI (8D)."
        )

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        policy.close()
        if not log_file.closed:
            log_file.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(policy.close)

    try:
        success_rate = run_task(args, policy, log_file)
        log(f"\nLog saved to: {log_path}", log_file)
        print(f"\nLog saved to: {log_path}")
        print(f"Run directory (logs + videos): {run_dir}")
        return success_rate
    finally:
        policy.close()
        if not log_file.closed:
            log_file.close()


if __name__ == "__main__":
    main()
