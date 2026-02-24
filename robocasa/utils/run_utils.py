"""Shared utilities for run_eval.py and run_demo.py (WebSocket policy scripts).
Client sends raw robosuite obs; policy server handles all remapping."""

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


def get_expected_policy_action_dim(arm_controller: str) -> int:
    """Return the policy output dimension expected by this arm controller.
    cartesian_pose -> 7 (6 pose + 1 gripper); joint_pos/joint_vel -> 8 (7 joints + 1 gripper).
    """
    return get_arm_action_dim(arm_controller) + 1


def get_task_max_steps(task_name: str, default_horizon: int = 500) -> int:
    """Return horizon for task; matches robocasa_rollout_utils.

    Reads from SINGLE_STAGE_TASK_DATASETS and MULTI_STAGE_TASK_DATASETS.
    """
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if task_name in all_tasks and "horizon" in all_tasks[task_name]:
        return all_tasks[task_name]["horizon"]
    return default_horizon


def enable_joint_pos_observable(env) -> None:
    """Enable robot0_joint_pos observable. Call once after env creation."""
    for ob_name in getattr(env, "observation_names", []):
        if "joint_pos" in ob_name:
            env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)
            break


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
