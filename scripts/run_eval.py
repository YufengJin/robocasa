#!/usr/bin/env python3
"""
run_eval.py — RoboCasa evaluation client (gRPC).

Runs the RoboCasa simulation loop and delegates action inference to a remote
Policy Server over gRPC.  This cleanly decouples environment execution from
policy inference so they can live in separate containers / processes.

Usage:
    # Start the policy server first (e.g. the random-action test server):
    python tests/test_random_policy_server.py --port 50051

    # Then run the evaluation client:
    python scripts/run_eval.py \
        --task_name PnPCounterToCab \
        --num_trials 5 \
        --policy randomPolicy \
        --seed 195 \
        --policy_server_addr localhost:50051

    Logs and videos are written to: <log_dir>/<task_name>--<YYYYMMDD_HHMMSS>/
"""

import argparse
import ast
import atexit
import os
import random
import signal
import sys
import time
from datetime import datetime

import cv2
import grpc
import imageio
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

# ---------------------------------------------------------------------------
# Make sure the workspace root is on sys.path so that the generated gRPC stubs
# under robocasa/grpc/ can be imported.
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from robocasa.grpc import policy_service_pb2, policy_service_pb2_grpc  # noqa: E402
from robocasa.utils.dataset_registry import (  # noqa: E402
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)

# ---------------------------------------------------------------------------
# Task-specific max steps (from cosmos-policy reference)
# ---------------------------------------------------------------------------
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


# ── helpers ─────────────────────────────────────────────────────────────────

def log(msg: str, log_file=None):
    """Print a message and optionally write it to a log file."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def jpeg_encode(image: np.ndarray, quality: int = 95) -> bytes:
    """Encode a uint8 HWC image to JPEG bytes."""
    # OpenCV expects BGR; our images are RGB.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# ── environment ─────────────────────────────────────────────────────────────

def create_robocasa_env(args, seed=None, episode_idx=None):
    """Create a RoboCasa environment.

    Mirrors cosmos-policy's ``create_robocasa_env`` but uses
    ``load_composite_controller_config`` instead of a pickled controller blob.
    """
    # Parse layout_and_style_ids
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
        has_renderer=False,
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
    env = robosuite.make(**env_kwargs)
    return env


def prepare_observation(obs, flip_images: bool = True):
    """Extract images and proprio from a raw environment observation.

    Returns a dict with keys:
        primary_image, secondary_image, wrist_image, proprio
    """
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


# ── video saving ────────────────────────────────────────────────────────────

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


# ── gRPC helpers ────────────────────────────────────────────────────────────

def grpc_reset(stub, task_name, task_description, env):
    """Send a Reset RPC to the policy server."""
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
    """Send an observation and receive an action from the policy server."""
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


# ── episode / task runners ──────────────────────────────────────────────────

def run_episode(args, env, task_description, stub, episode_idx, log_file=None):
    """Run a single evaluation episode.

    The environment loop mirrors cosmos-policy's ``run_episode`` but replaces
    local model inference with a ``GetAction`` gRPC call.
    """
    # Wait for objects to settle
    NUM_WAIT_STEPS = 10
    for _ in range(NUM_WAIT_STEPS):
        dummy = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy)

    max_steps = TASK_MAX_STEPS.get(args.task_name, 500)
    success = False
    episode_length = 0

    # Containers for replay video
    replay_primary, replay_secondary, replay_wrist = [], [], []

    for t in range(max_steps):
        observation = prepare_observation(obs, flip_images=args.flip_images)

        replay_primary.append(observation["primary_image"])
        replay_secondary.append(observation["secondary_image"])
        replay_wrist.append(observation["wrist_image"])

        # Query the policy server
        start = time.time()
        action = grpc_get_action(stub, observation, task_description, args.img_res)
        query_time = time.time() - start

        if t % 50 == 0:
            log(f"  t={t}: action query {query_time:.3f}s, action[:4]={action[:4]}", log_file)

        # Pad 7-dim policy action to 12-dim env action if needed
        if action.shape[-1] == 7 and env.action_dim == 12:
            mobile_base = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
            action = np.concatenate([action, mobile_base])

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


def run_task(args, stub, log_file=None):
    """Evaluate a task over multiple episodes and report success rate."""
    log(f"\nEvaluating task: {args.task_name}", log_file)

    successes = []
    lengths = []

    for ep_idx in range(args.num_trials):
        log(f"\n--- Episode {ep_idx + 1}/{args.num_trials} ---", log_file)

        seed = args.seed * ep_idx * 256 if args.deterministic else None
        env = create_robocasa_env(args, seed=seed, episode_idx=ep_idx)
        env.reset()

        task_description = env.get_ep_meta()["lang"]
        log(f"Task description: {task_description}", log_file)

        # Notify policy server of new episode
        ok = grpc_reset(stub, args.task_name, task_description, env)
        if not ok:
            log("  WARNING: policy server Reset returned failure", log_file)

        success, length, rep_p, rep_s, rep_w = run_episode(
            args, env, task_description, stub, ep_idx, log_file
        )
        successes.append(success)
        lengths.append(length)

        # Save video
        if args.save_video:
            save_rollout_video(
                rep_p, rep_s, rep_w,
                ep_idx, success, task_description,
                output_dir=args.log_dir,
            )

        env.close()

        sr = sum(successes) / len(successes) * 100
        log(f"Running success rate: {sum(successes)}/{len(successes)} ({sr:.1f}%)", log_file)

    # Final summary
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


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    all_tasks = list({**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}.keys())

    parser = argparse.ArgumentParser(
        description="RoboCasa gRPC evaluation client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # gRPC connection
    parser.add_argument("--policy_server_addr", type=str, default="localhost:50051",
                        help="Address of the policy gRPC server")
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy name for logging (e.g. randomPolicy, cosmos)")
    # Task
    parser.add_argument("--task_name", type=str, default="PnPCounterToCab",
                        choices=all_tasks,
                        help="RoboCasa task name")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Number of evaluation episodes per task")
    # Environment
    parser.add_argument("--robots", type=str, default="PandaMobile",
                        help="Robot type")
    parser.add_argument("--img_res", type=int, default=224,
                        help="Camera image resolution (square)")
    parser.add_argument("--obj_instance_split", type=str, default="B",
                        help="Object instance split (B = held-out test)")
    parser.add_argument("--layout_and_style_ids", type=str,
                        default="((1,1),(2,2),(4,4),(6,9),(7,10))",
                        help="Layout and style IDs for scene selection")
    parser.add_argument("--flip_images", action="store_true", default=True,
                        help="Flip images vertically (RoboCasa renders upside-down)")
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images")
    # Reproducibility
    parser.add_argument("--seed", type=int, default=195,
                        help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic seeding")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    # Logging
    parser.add_argument("--log_dir", type=str, default="./eval_logs",
                        help="Directory for logs and rollout videos")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save rollout videos")
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")

    return parser.parse_args()


def set_seed_everywhere(seed: int, deterministic: bool = True):
    """Set global random seeds for reproducibility."""
    if deterministic:
        os.environ["DETERMINISTIC"] = "True"
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    # Validate task
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if args.task_name not in all_tasks:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {list(all_tasks.keys())}")

    # Set global seed before any other randomness
    set_seed_everywhere(args.seed, deterministic=args.deterministic)

    # Log directory: base_log_dir / task_name--YYYYMMDD_HHMMSS
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, f"{args.task_name}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)
    args.log_dir = run_dir  # use run_dir for this run's logs and videos

    log_path = os.path.join(run_dir, "eval.log")
    log_file = open(log_path, "w")

    # Comprehensive run header
    log("=" * 60, log_file)
    log("RoboCasa gRPC Eval Run", log_file)
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
    log(f"  obj_instance_split: {args.obj_instance_split}", log_file)
    log(f"  layout_and_style_ids: {args.layout_and_style_ids}", log_file)
    log("=" * 60, log_file)
    log("", log_file)

    log(f"Connecting to policy server at {args.policy_server_addr} ...", log_file)

    # Set up gRPC channel with increased message size (images can be large)
    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),    # 50 MB
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50 MB
    ]
    channel = grpc.insecure_channel(args.policy_server_addr, options=options)
    stub = policy_service_pb2_grpc.PolicyServiceStub(channel)

    # ── Graceful shutdown on kill / Ctrl+C ──────────────────────────────
    def _cleanup(signum=None, frame=None):
        print("\nCleaning up gRPC channel …")
        channel.close()
        if not log_file.closed:
            log_file.close()
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(_cleanup)

    # Wait for server to be ready
    log("Waiting for policy server to be ready ...", log_file)
    try:
        grpc.channel_ready_future(channel).result(timeout=30)
    except grpc.FutureTimeoutError:
        log("ERROR: policy server did not become ready within 30 s", log_file)
        sys.exit(1)
    log("Policy server is ready.", log_file)

    try:
        success_rate = run_task(args, stub, log_file)
        log(f"\nLog saved to: {log_path}", log_file)
        print(f"\nLog saved to: {log_path}")
        print(f"Run directory (logs + videos): {run_dir}")
        return success_rate
    finally:
        channel.close()
        if not log_file.closed:
            log_file.close()


if __name__ == "__main__":
    main()
