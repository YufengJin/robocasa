#!/usr/bin/env python3
"""
Test arm controllers for RoboCasa via dynamics prediction.

For each controller (cartesian_pose, joint_pos, joint_vel):
  1. Get mid_state after warmup
  2. Apply known action for N steps
  3. Predict expected state from the controller's kinematic model
  4. Compare predicted vs actual state (direction cosine similarity + magnitude)

Controller models (from robosuite source):
  - OSC_POSE (delta): goal_eef += scale_action(a) each step
        scale = (output_max - output_min) / (input_max - input_min)
        pos: [-1,1] -> [-0.05, 0.05] m   => scale_pos = 0.05
        ori: [-1,1] -> [-0.5, 0.5] rad    => scale_ori = 0.5
  - JOINT_POSITION (delta): goal_qpos += scale_action(a) each step
        [-1,1] -> [-0.05, 0.05] rad       => scale_jpos = 0.05
  - JOINT_VELOCITY (absolute): target_vel = scale_action(a)
        [-1,1] -> [-1, 1] rad/s           => scale_jvel = 1.0
        joint_pos integrates: dq ≈ vel * dt, dt = 1/control_freq

Run in container:
    docker exec robocasa_container /opt/conda/envs/robocasa/bin/python tests/test_arm_controller.py --num_trials 2
"""

import os
import sys

import numpy as np

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)
from robocasa.utils.run_utils import (
    ARM_CONTROLLER_ACTION_DIMS,
    ARM_CONTROLLER_MAP,
    create_robocasa_env,
    enable_joint_pos_observable,
)

TASK_NAME = "PnPCounterToCab"
DEFAULT_NUM_TRIALS = 50
STEPS_PER_TRIAL = 5
CONTROL_FREQ = 20
ACTION_SCALE = 0.15

# robosuite default scale_action parameters per controller
OSC_POS_SCALE = 0.05      # input[-1,1] -> output[-0.05,0.05] m
OSC_ORI_SCALE = 0.5       # input[-1,1] -> output[-0.5,0.5] rad
JPOS_SCALE = 0.05         # input[-1,1] -> output[-0.05,0.05] rad
JVEL_SCALE = 1.0           # input[-1,1] -> output[-1,1] rad/s

DIRECTION_COSINE_THRESHOLD = 0.5
MAGNITUDE_RATIO_BOUNDS = (0.01, 20.0)


def _get_robot_state(obs, has_joint_pos: bool):
    """Extract robot state from obs."""
    state = {
        "eef_pos": np.array(obs["robot0_eef_pos"], dtype=np.float64),
        "eef_quat": np.array(obs["robot0_eef_quat"], dtype=np.float64),
        "gripper_qpos": np.array(obs["robot0_gripper_qpos"], dtype=np.float64),
        # eef position in robot base frame (matches OSC delta reference frame)
        "base_to_eef_pos": np.array(obs["robot0_base_to_eef_pos"], dtype=np.float64),
    }
    if has_joint_pos and "robot0_joint_pos" in obs:
        state["joint_pos"] = np.array(obs["robot0_joint_pos"], dtype=np.float64)
    return state


def _make_action(arm_controller: str, action_low, action_high, trial_idx: int) -> np.ndarray:
    """Generate a deterministic non-zero action that varies per trial."""
    arm_dim = ARM_CONTROLLER_ACTION_DIMS[arm_controller]
    action_dim = action_low.shape[0]
    action = np.zeros(action_dim, dtype=np.float64)
    for i in range(arm_dim):
        action[i] = ACTION_SCALE * np.sin(trial_idx * 0.7 + i * 1.1)
    action[-1] = -1.0  # gripper closed, base dims stay 0
    action = np.clip(action, action_low, action_high)
    return action


def predict_state(mid_state: dict, action: np.ndarray, arm_controller: str,
                  n_steps: int) -> dict:
    """Predict robot state after n_steps of constant action, given mid_state.

    Uses the controller's kinematic model (scale_action + integration).
    Returns predicted state dict with same keys as mid_state.
    """
    arm_dim = ARM_CONTROLLER_ACTION_DIMS[arm_controller]
    arm_action = action[:arm_dim]
    pred = {}

    if arm_controller == "cartesian_pose":
        # OSC delta mode (input_ref_frame="base"): goal accumulates in base frame
        # Use base_to_eef_pos for comparison to avoid world-frame mismatch
        scaled_pos = arm_action[0:3] * OSC_POS_SCALE
        pred["base_to_eef_pos"] = mid_state["base_to_eef_pos"] + n_steps * scaled_pos

    elif arm_controller == "joint_pos":
        # Joint position delta mode: each step goal_qpos += arm_action * JPOS_SCALE
        scaled_delta = arm_action * JPOS_SCALE
        if "joint_pos" in mid_state:
            pred["joint_pos"] = mid_state["joint_pos"] + n_steps * scaled_delta
        pred["eef_pos"] = None  # can't easily predict eef from joint delta without FK

    elif arm_controller == "joint_vel":
        # Joint velocity: target_vel = arm_action * JVEL_SCALE
        # integration: dq = vel * dt per step, dt = 1/control_freq
        dt = 1.0 / CONTROL_FREQ
        scaled_vel = arm_action * JVEL_SCALE
        if "joint_pos" in mid_state:
            pred["joint_pos"] = mid_state["joint_pos"] + n_steps * scaled_vel * dt
        pred["eef_pos"] = None

    return pred


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_prediction(mid_state: dict, actual_state: dict, pred_state: dict,
                     arm_controller: str) -> tuple[bool, str]:
    """Compare predicted state with actual state.

    Checks:
      1. Direction: cosine similarity between predicted and actual delta vectors
      2. Magnitude: ratio of actual/predicted magnitude is within reasonable bounds

    Returns (passed, detail_msg).
    """
    if arm_controller == "cartesian_pose":
        if pred_state.get("base_to_eef_pos") is None:
            return False, "no base_to_eef_pos prediction"
        pred_delta = pred_state["base_to_eef_pos"] - mid_state["base_to_eef_pos"]
        actual_delta = actual_state["base_to_eef_pos"] - mid_state["base_to_eef_pos"]
        label = "base_to_eef_pos"

    elif arm_controller in ("joint_pos", "joint_vel"):
        if "joint_pos" not in mid_state or pred_state.get("joint_pos") is None:
            return False, "no joint_pos available"
        pred_delta = pred_state["joint_pos"] - mid_state["joint_pos"]
        actual_delta = actual_state["joint_pos"] - mid_state["joint_pos"]
        label = "joint_pos"
    else:
        return False, f"unknown controller {arm_controller}"

    pred_mag = np.linalg.norm(pred_delta)
    actual_mag = np.linalg.norm(actual_delta)

    if pred_mag < 1e-10:
        return (actual_mag < 1e-4), f"{label}: pred_mag≈0, actual_mag={actual_mag:.6f}"

    if actual_mag < 1e-10:
        return False, f"{label}: actual_mag≈0 but pred_mag={pred_mag:.6f}"

    cos = _cosine_sim(pred_delta, actual_delta)
    ratio = actual_mag / pred_mag

    direction_ok = cos >= DIRECTION_COSINE_THRESHOLD
    lo, hi = MAGNITUDE_RATIO_BOUNDS
    magnitude_ok = lo <= ratio <= hi

    detail = (
        f"{label}: cos={cos:.3f} (thresh={DIRECTION_COSINE_THRESHOLD}), "
        f"ratio={ratio:.3f} (bounds={lo}-{hi}), "
        f"pred_mag={pred_mag:.5f}, actual_mag={actual_mag:.5f}"
    )

    passed = direction_ok and magnitude_ok
    return passed, detail


def test_controller_dynamics(arm_controller: str, num_trials: int) -> tuple[int, int, list]:
    """Run num_trials dynamics-prediction tests. Returns (passed, failed, details)."""
    env = create_robocasa_env(
        task_name=TASK_NAME,
        robots="PandaMobile",
        arm_controller=arm_controller,
        img_res=64,
        seed=0,
    )
    enable_joint_pos_observable(env)

    action_low, action_high = env.action_spec
    actual_dim = action_low.shape[0]

    passed = 0
    failed = 0
    details = []

    for trial in range(num_trials):
        try:
            obs = env.reset()
            has_joint = "robot0_joint_pos" in obs

            for _ in range(3):
                obs, _, _, _ = env.step(np.zeros(actual_dim))

            mid_state = _get_robot_state(obs, has_joint)
            action = _make_action(arm_controller, action_low, action_high, trial)

            for _ in range(STEPS_PER_TRIAL):
                obs, _, _, _ = env.step(action)

            actual_state = _get_robot_state(obs, has_joint)
            pred_state = predict_state(mid_state, action, arm_controller, STEPS_PER_TRIAL)

            ok, detail = check_prediction(mid_state, actual_state, pred_state, arm_controller)
            if ok:
                passed += 1
            else:
                failed += 1
            details.append((trial, ok, detail))

        except Exception as e:
            failed += 1
            details.append((trial, False, f"ERROR: {e}"))

    env.close()
    return passed, failed, details


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test arm controller dynamics prediction")
    parser.add_argument("--num_trials", type=int, default=DEFAULT_NUM_TRIALS,
                        help=f"Trials per controller (default {DEFAULT_NUM_TRIALS})")
    parser.add_argument("--verbose", action="store_true", help="Print every trial detail")
    args = parser.parse_args()

    print("=" * 70)
    print("RoboCasa arm controller dynamics prediction test")
    print("=" * 70)
    print(f"  task:             {TASK_NAME}")
    print(f"  trials:           {args.num_trials} per controller")
    print(f"  steps/trial:      {STEPS_PER_TRIAL}")
    print(f"  cos threshold:    {DIRECTION_COSINE_THRESHOLD}")
    print(f"  magnitude bounds: {MAGNITUDE_RATIO_BOUNDS}")
    print("=" * 70)

    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if TASK_NAME not in all_tasks:
        print(f"ERROR: Task {TASK_NAME} not found")
        sys.exit(1)

    all_ok = True
    for arm_controller in ARM_CONTROLLER_MAP:
        print(f"\n--- {arm_controller} ({ARM_CONTROLLER_MAP[arm_controller]}) ---")
        passed, failed, details = test_controller_dynamics(arm_controller, args.num_trials)
        total = passed + failed
        pct = 100 * passed / total if total > 0 else 0

        if args.verbose:
            for trial, ok, detail in details:
                tag = "PASS" if ok else "FAIL"
                print(f"  [{tag}] trial {trial}: {detail}")
        else:
            fail_details = [(t, d) for t, ok, d in details if not ok]
            for t, d in fail_details[:3]:
                print(f"  [FAIL] trial {t}: {d}")
            if len(fail_details) > 3:
                print(f"  ... and {len(fail_details) - 3} more failures")

        status = "OK" if failed == 0 else "FAIL"
        print(f"  Result: {passed}/{total} passed ({pct:.1f}%) [{status}]")
        if failed > 0:
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("All controller dynamics prediction tests passed.")
    else:
        print("Some trials failed — check cosine/magnitude details above.")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
