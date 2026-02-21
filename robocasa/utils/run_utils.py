"""Shared utilities for run_eval.py and run_demo.py (WebSocket policy scripts)."""

import ast
import os
import random
from typing import Dict, Optional

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)

# ---------------------------------------------------------------------------
# Arm controller mapping:  CLI name  →  robosuite controller type
# ---------------------------------------------------------------------------
ARM_CONTROLLER_MAP: Dict[str, str] = {
    "cartesian_pose": "OSC_POSE",
    "joint_pos": "JOINT_POSITION",
    "joint_vel": "JOINT_VELOCITY",
}

ARM_CONTROLLER_ACTION_DIMS: Dict[str, int] = {
    "cartesian_pose": 6,
    "joint_pos": 7,
    "joint_vel": 7,
}

def _patch_joint_vel_controller() -> None:
    """Workaround for robosuite <= 1.5.2 bug: ``JointVelocityController.__init__``
    assigns ``self.torque_compensation = bool`` which collides with the read-only
    ``@property`` on the ``Controller`` base class.

    We patch ``__init__`` to store the flag as ``_use_torque_compensation`` and
    ``run_controller`` to read it from there.
    """
    from robosuite.controllers.parts.generic.joint_vel import JointVelocityController as JVC
    from robosuite.controllers.parts.controller import Controller

    if getattr(JVC, "_patched_tc", False):
        return

    _orig_init = JVC.__init__

    def _new_init(self, *args, **kwargs):
        _prop = Controller.__dict__["torque_compensation"]
        Controller.torque_compensation = property(_prop.fget, lambda self, v: None)
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            Controller.torque_compensation = _prop
        self._use_torque_compensation = kwargs.get("use_torque_compensation", True)

    def _new_run(self):
        if self.goal_vel is None:
            self.set_goal(np.zeros(self.joint_dim))
        self.update()
        if self.interpolator is not None and self.interpolator.order == 1:
            self.current_vel = self.interpolator.get_interpolated_goal()
        else:
            self.current_vel = np.array(self.goal_vel)
        err = self.current_vel - self.joint_vel
        derr = err - self.last_err
        self.last_err = err
        self.derr_buf.push(derr)
        if not self.saturated:
            self.summed_err += err
        if self._use_torque_compensation:
            torques = (
                self.kp * err + self.ki * self.summed_err
                + self.kd * self.derr_buf.average + self.torque_compensation
            )
        else:
            torques = self.kp * err + self.ki * self.summed_err + self.kd * self.derr_buf.average
        self.torques = self.clip_torques(torques)
        self.saturated = np.any(self.torques != torques)
        super(JVC, self).run_controller()
        return self.torques

    JVC.__init__ = _new_init
    JVC.run_controller = _new_run
    JVC._patched_tc = True


_patch_joint_vel_controller()


def build_controller_config(robot_name: str, arm_controller: str) -> dict:
    """Build composite controller config with the requested arm controller type.

    ``load_composite_controller_config`` flattens the body_parts dict so arm
    configs appear as ``body_parts["right"]``, ``body_parts["left"]``, etc.
    (not nested under an ``"arms"`` key).

    Args:
        robot_name: Robot type, e.g. "PandaMobile".
        arm_controller: One of ARM_CONTROLLER_MAP keys.
    """
    cfg = load_composite_controller_config(controller=None, robot=robot_name)
    arm_type = ARM_CONTROLLER_MAP[arm_controller]
    arm_keys = {"right", "left"}
    bp = cfg.get("body_parts", {})
    for arm_key in arm_keys & bp.keys():
        arm_cfg = bp[arm_key]
        arm_cfg["type"] = arm_type
        if arm_type in ("JOINT_POSITION", "JOINT_VELOCITY"):
            arm_cfg.pop("output_max", None)
            arm_cfg.pop("output_min", None)
            arm_cfg.pop("uncouple_pos_ori", None)
            arm_cfg.pop("position_limits", None)
            arm_cfg.pop("orientation_limits", None)
    return cfg


def get_arm_action_dim(arm_controller: str) -> int:
    """Return the action dimension of the arm (excluding gripper)."""
    return ARM_CONTROLLER_ACTION_DIMS[arm_controller]


def get_task_max_steps(task_name: str, default_horizon: int = 500) -> int:
    """Return horizon for task; matches robocasa_rollout_utils.

    Reads from SINGLE_STAGE_TASK_DATASETS and MULTI_STAGE_TASK_DATASETS.
    """
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if task_name in all_tasks and "horizon" in all_tasks[task_name]:
        return all_tasks[task_name]["horizon"]
    return default_horizon


DROID_IMG_SIZE = 224


def _ensure_uint8_hwc(img: np.ndarray, target_hw: tuple = (DROID_IMG_SIZE, DROID_IMG_SIZE)) -> np.ndarray:
    """Convert image to uint8 HWC and resize to target if needed."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    h, w = target_hw
    if img.shape[0] != h or img.shape[1] != w:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((w, h), resample=Image.BICUBIC))
    return img


def _gripper_qpos_to_droid(gripper_qpos: np.ndarray) -> np.ndarray:
    """Map Panda robot0_gripper_qpos (2,) to DROID gripper_position (1,) in [0,1]. 0=open, 1=closed."""
    q = np.asarray(gripper_qpos).flatten()
    val = (q[0] + q[1]) / 2.0 if len(q) >= 2 else float(q[0])
    # Panda typical range: 0 (open) to ~0.04 (closed) per finger
    return np.array([np.clip(val / 0.04, 0.0, 1.0)], dtype=np.float64)


def prepare_observation_droid(
    obs: dict,
    task_description: str,
    flip_images: bool = True,
    img_size: int = DROID_IMG_SIZE,
) -> dict:
    """Convert raw RoboCasa observation to OpenPI DROID format.

    Returns a dict with keys expected by DROID policy:
        observation/exterior_image_1_left, observation/wrist_image_left,
        observation/joint_position, observation/gripper_position, prompt
    """
    target_hw = (img_size, img_size)

    exterior = obs.get("robot0_agentview_left_image")
    wrist = obs.get("robot0_eye_in_hand_image")
    if flip_images:
        if exterior is not None:
            exterior = np.flipud(exterior).copy()
        if wrist is not None:
            wrist = np.flipud(wrist).copy()

    exterior = _ensure_uint8_hwc(exterior, target_hw) if exterior is not None else np.zeros((*target_hw, 3), dtype=np.uint8)
    wrist = _ensure_uint8_hwc(wrist, target_hw) if wrist is not None else np.zeros((*target_hw, 3), dtype=np.uint8)

    joint_pos = obs.get("robot0_joint_pos")
    if joint_pos is None:
        raise KeyError(
            "robot0_joint_pos not in observation. Enable it when creating env for --droid, "
            "e.g. env.modify_observable(observable_name='robot0_joint_pos', attribute='active', modifier=True)"
        )
    joint_pos = np.asarray(joint_pos, dtype=np.float64).flatten()
    if joint_pos.shape[0] != 7:
        raise ValueError(f"robot0_joint_pos must be 7D, got shape {joint_pos.shape}")

    gripper_qpos = obs["robot0_gripper_qpos"]
    gripper_pos = _gripper_qpos_to_droid(gripper_qpos)

    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": task_description,
    }


def enable_joint_pos_observable(env) -> None:
    """Enable robot0_joint_pos observable for DROID mode (required for prepare_observation_droid)."""
    for ob_name in getattr(env, "observation_names", []):
        if "joint_pos" in ob_name:
            env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)
            break


def prepare_observation(obs: dict, flip_images: bool = True) -> dict:
    """Extract images and proprio from a raw environment observation.

    Returns a dict with keys: primary_image, secondary_image, wrist_image, proprio.
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


def set_seed_everywhere(seed: int, deterministic: bool = True) -> None:
    """Set global random seeds for reproducibility."""
    if deterministic:
        os.environ["DETERMINISTIC"] = "True"
    random.seed(seed)
    np.random.seed(seed)


def pad_action_for_env(
    action: np.ndarray,
    arm_controller: str,
    env_action_dim: int,
) -> np.ndarray:
    """Pad policy output to env.action_dim (e.g. PandaMobile has extra base dims)."""
    expected_dim = get_arm_action_dim(arm_controller) + 1  # +1 gripper
    if action.shape[-1] == expected_dim and env_action_dim > expected_dim:
        pad_dim = env_action_dim - expected_dim
        mobile_base = np.zeros(pad_dim, dtype=np.float64)
        mobile_base[-1] = -1.0
        action = np.concatenate([action, mobile_base])
    return np.array(action, dtype=np.float64, copy=True)


def create_robocasa_env(
    task_name: str,
    robots: str,
    arm_controller: str,
    img_res: int = 224,
    obj_instance_split: str = "B",
    layout_and_style_ids: Optional[str] = None,
    seed: Optional[int] = None,
    episode_idx: Optional[int] = None,
    use_gui: bool = False,
) -> "robosuite.environments.base.Environment":
    """Create a RoboCasa environment for run_eval / run_demo."""
    layout_ids = None
    if layout_and_style_ids:
        all_ids = ast.literal_eval(layout_and_style_ids)
        if episode_idx is not None:
            scene_index = (episode_idx // 10) % len(all_ids)
            layout_ids = (all_ids[scene_index],)
        else:
            layout_ids = all_ids

    robot_name = robots if isinstance(robots, str) else robots[0]
    controller_configs = build_controller_config(robot_name, arm_controller)

    env_kwargs = dict(
        env_name=task_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=img_res,
        camera_heights=img_res,
        has_renderer=use_gui,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=layout_ids,
        translucent_robot=False,
    )
    if use_gui:
        env_kwargs["render_camera"] = "robot0_agentview_left"
    return robosuite.make(**env_kwargs)
