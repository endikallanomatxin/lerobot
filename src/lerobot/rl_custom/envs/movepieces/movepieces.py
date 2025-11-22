from __future__ import annotations

import colorsys
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import json
from typing import List

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.error import Error as GymError


@dataclass(frozen=True)
class PiecePose:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]


@dataclass(frozen=True)
class PieceSpec:
    name: str
    mesh_file: Path
    initial: PiecePose
    target: PiecePose
    color: tuple[float, float, float]
    scale: float = 1.0


class MovePiecesEnv(gym.Env):
    """
    Genesis bimanual environment that places multiple SO101 parts.

    It speaks the same action/observation dialect as the real bimanual SO101 robot:
    - Actions: dict or tensor with 12 joints (`left/right_* .pos`), arms in [-100, 100], gripper in [0, 100].
    - Observations: per-joint keys with those ranges, plus `environment_state` (float32, batch x obs_dim).
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 60,
    }

    def __init__(
        self,
        device: torch.device,
        batch_size,
        max_steps,
        show_viewer: bool = False,
        record: bool = False,
        camera_setups: dict | None = None,
        piece_layout: list[dict] | None = None,
        piece_layout_file: str | Path | None = None,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.record = record
        self.debug_env_count = min(4, batch_size)
        self.env_colors = self._generate_pair_colors(self.debug_env_count)

        assets_root = Path(__file__).resolve().parent
        robot_path = assets_root / "assets" / "SO101" / "so101_new_calib.xml"

        gs.init(backend=gs.gs_backend.gpu, performance_mode=True)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1 / 60),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.1, 0.1, 0.1),
                n_rendered_envs=min(4, batch_size),
            ),
            renderer=gs.renderers.Rasterizer(),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # Camera setups (roughly aligned with wrist/overhead vantage points).
        default_cam_setups = {
            "left_wrist": {"res": (640, 480), "pos": (-0.35, -0.05, 0.25), "lookat": (0.0, -0.10, 0.05), "fov": 70},
            "right_wrist": {"res": (640, 480), "pos": (0.35, -0.05, 0.25), "lookat": (0.0, -0.10, 0.05), "fov": 70},
            "overhead": {"res": (640, 480), "pos": (0.0, -0.10, 0.75), "lookat": (0.0, -0.10, 0.05), "fov": 60},
        }
        cam_cfg = default_cam_setups if camera_setups is None else camera_setups
        self.cameras = {
            name: self.scene.add_camera(res=tuple(cfg["res"]), pos=cfg["pos"], lookat=cfg["lookat"], fov=cfg["fov"], GUI=False)
            for name, cfg in cam_cfg.items()
        }

        # self.plane = self.scene.add_entity(gs.morphs.Plane())
        # Table spans 0.8 m along x (between robots) and sits flush with the ground (top at z=0).
        self.print_bed_size = (0.80, 1.45, 0.02)
        self.print_bed_center = (0.0, 0.0, -self.print_bed_size[2])
        self.print_bed_top_z = self.print_bed_center[2] + self.print_bed_size[2] / 2.0
        self.print_bed = self.scene.add_entity(
            gs.morphs.Box(
                pos=self.print_bed_center,
                size=self.print_bed_size,
                fixed=True,
                visualization=True,
                collision=True,
            )
        )

        self.robot_base_configs = [
            {"pos": (0.37, 0.0, 0.0), "quat": (0.0, 0.0, 0.0, 1.0)},
            {"pos": (-0.37, 0.0, 0.0), "quat": (1.0, 0.0, 0.0, 0.0)},
        ]
        self.sides = ["left", "right"]
        self.side_to_robot_idx = {"left": 0, "right": 1}
        self.robots = [
            self.scene.add_entity(
                gs.morphs.MJCF(
                    file=str(robot_path),
                    pos=config["pos"],
                    quat=config.get("quat"),
                )
            )
            for config in self.robot_base_configs
        ]
        self.num_robots = len(self.robots)

        layout_override = piece_layout
        if layout_override is None and piece_layout_file is not None:
            with open(piece_layout_file, "r") as f:
                layout_override = json.load(f)

        self.piece_specs = self._build_piece_specs(robot_path.parent, layout_override)
        self.piece_entities = [
            self.scene.add_entity(
                gs.morphs.Mesh(
                    file=str(spec.mesh_file),
                    pos=spec.initial.pos,
                    quat=spec.initial.quat,
                    scale=spec.scale,
                    fixed=False,
                    collision=True,
                    visualization=True,
                    convexify=False,
                    parse_glb_with_trimesh=True,  # ensure GLB vertex colors/materials are loaded
                )
            )
            for spec in self.piece_specs
        ]
        self.num_pieces = len(self.piece_entities)

        self.scene.build(n_envs=batch_size)

        self.jnt_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.dofs_idx = [[robot.get_joint(name).dof_idx_local for name in self.jnt_names] for robot in self.robots]
        self.dofs_per_robot = len(self.jnt_names)
        joint_limits = self._load_joint_limits(robot_path)
        lower = [joint_limits[name][0] for name in self.jnt_names]
        upper = [joint_limits[name][1] for name in self.jnt_names]
        self.joint_lower = torch.tensor(lower, device=self.device, dtype=torch.float32)
        self.joint_upper = torch.tensor(upper, device=self.device, dtype=torch.float32)
        self.gripper_link_name = "gripper_tip"
        self.forearm_link_name = "lower_arm"
        self._piece_initial_pose = torch.tensor(
            [(*spec.initial.pos, *spec.initial.quat) for spec in self.piece_specs],
            dtype=torch.float32,
            device=self.device,
        )
        self._piece_target_pose = torch.tensor(
            [(*spec.target.pos, *spec.target.quat) for spec in self.piece_specs],
            dtype=torch.float32,
            device=self.device,
        )
        self.piece_target_pos = self._piece_target_pose[:, :3]
        self.piece_target_rotvec = self._quat_to_rotvec(self._piece_target_pose[:, 3:])

        self.current_step = 0

        self.per_robot_obs_dim = 2 * self.dofs_per_robot + 12
        self.per_piece_obs_dim = 12
        self.obs_dim = self.num_robots * self.per_robot_obs_dim + self.num_pieces * self.per_piece_obs_dim
        self.act_dim = self.num_robots * self.dofs_per_robot
        self.action_keys = [f"{side}_{joint}.pos" for side in self.sides for joint in self.jnt_names]
        obs_spaces: dict[str, gym.Space] = {
            "environment_state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.batch_size, self.obs_dim),
                dtype=np.float32,
            )
        }
        for key in self.action_keys:
            if key.endswith("gripper.pos"):
                low, high = 0.0, 100.0
            else:
                low, high = -100.0, 100.0
            obs_spaces[key] = spaces.Box(
                low=low,
                high=high,
                shape=(self.batch_size,),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.batch_size, self.act_dim),
            dtype=np.float32,
        )
        self.success_threshold = 0.02
        self.num_envs = batch_size
        self.active_piece_idx = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.piece_completed_flags = torch.zeros(
            self.batch_size,
            self.num_pieces,
            dtype=torch.bool,
            device=self.device,
        )
        self.pick_threshold = 0.025
        self.place_threshold = 0.02
        self.rotation_threshold = 0.05
        self.w_pos = 4.0
        self.w_rot = 1.5
        self.w_phase1 = 1.0
        self.w_phase2 = 2.0
        self.w_phase3_pos = 3.0
        self.w_phase3_rot = 1.5
        self.completion_bonus = 5.0

        if self.record:
            self.cam.start_recording()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ctrl_pos = torch.zeros(self.dofs_per_robot)
        tiled_ctrl = torch.tile(ctrl_pos, (self.batch_size, 1))
        for robot, dof_idx in zip(self.robots, self.dofs_idx):
            robot.control_dofs_position(tiled_ctrl, dof_idx)

        self._reset_pieces()
        self.active_piece_idx.zero_()
        self.piece_completed_flags.zero_()
        self.target_pos = self.piece_target_pos[0].unsqueeze(0).repeat(self.batch_size, 1)
        self.current_step = 0

        for _ in range(10):
            self.scene.step()

        if self.record:
            self.cam.start_recording()

        gripper_pos = self._stack_link_states(self.gripper_link_name, "get_pos")
        self._update_debug_markers(gripper_pos)

        return self.get_obs(), {}

    def get_obs(self):
        dof_ang = self._stack_dof_states("get_dofs_position")
        dof_vel = self._stack_dof_states("get_dofs_velocity")
        gripper_pos = self._stack_link_states(self.gripper_link_name, "get_pos")
        gripper_quat = self._stack_link_states(self.gripper_link_name, "get_quat")
        gripper_rotvec = self._quat_to_rotvec(gripper_quat)
        gripper_vel = self._stack_link_states(self.gripper_link_name, "get_vel")
        gripper_ang = self._stack_link_states(self.gripper_link_name, "get_ang")
        piece_pos = self._stack_piece_states("get_pos")
        piece_quat = self._stack_piece_states("get_quat")
        piece_rotvec = self._quat_to_rotvec(piece_quat)
        target_pos = self.piece_target_pos.unsqueeze(1).expand(-1, self.batch_size, -1)
        target_rotvec = self.piece_target_rotvec.unsqueeze(1).expand(-1, self.batch_size, -1)

        def flatten_robot_tensor(tensor):
            return tensor.permute(1, 0, 2).reshape(self.batch_size, -1)

        def flatten_piece_tensor(tensor):
            return tensor.permute(1, 0, 2).reshape(self.batch_size, -1)

        env_state = torch.cat(
            [
                flatten_robot_tensor(dof_ang),
                flatten_robot_tensor(dof_vel),
                flatten_robot_tensor(gripper_pos),
                flatten_robot_tensor(gripper_rotvec),
                flatten_robot_tensor(gripper_vel),
                flatten_robot_tensor(gripper_ang),
                flatten_piece_tensor(piece_pos),
                flatten_piece_tensor(piece_rotvec),
                flatten_piece_tensor(target_pos),
                flatten_piece_tensor(target_rotvec),
            ],
            dim=1,
        )
        env_state = torch.nan_to_num(env_state, nan=0.0, posinf=1e4, neginf=-1e4)

        obs_dict: dict[str, np.ndarray] = {"environment_state": env_state.detach().cpu().numpy().astype(np.float32)}

        dof_ang = dof_ang.permute(1, 0, 2)
        for side, robot_idx in self.side_to_robot_idx.items():
            for j, joint in enumerate(self.jnt_names):
                ang = dof_ang[:, robot_idx, j]
                if joint == "gripper":
                    val = (ang - self.joint_lower[j]) / (self.joint_upper[j] - self.joint_lower[j] + 1e-6) * 100.0
                else:
                    mid = 0.5 * (self.joint_lower[j] + self.joint_upper[j])
                    half = 0.5 * (self.joint_upper[j] - self.joint_lower[j] + 1e-6)
                    val = torch.clamp((ang - mid) / half, -1.0, 1.0) * 100.0
                obs_dict[f"{side}_{joint}.pos"] = val.detach().cpu().numpy().astype(np.float32)

        if self.cameras:
            pixels = {}
            for name, cam in self.cameras.items():
                try:
                    img = cam.render(env_idx=0)
                except TypeError:
                    img = cam.render()
                if isinstance(img, (list, tuple)):
                    img = img[0]
                if isinstance(img, torch.Tensor):
                    img_np = img.detach().cpu().numpy()
                else:
                    img_np = np.asarray(img)
                if img_np.ndim == 4:
                    img_np = img_np[0]
                if img_np.dtype != np.uint8:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                pixels[name] = img_np
            obs_dict["pixels"] = pixels

        return obs_dict

    def compute_reward(self):
        reward_dict = {}
        batch_indices = torch.arange(self.batch_size, device=self.device)

        gripper_pos = self._stack_link_states(self.gripper_link_name, "get_pos")
        gripper_vel = self._stack_link_states(self.gripper_link_name, "get_vel")
        gripper_vel_mag = torch.linalg.vector_norm(gripper_vel, dim=-1)
        gripper_ang_vel = self._stack_link_states(self.gripper_link_name, "get_ang")
        gripper_ang_mag = torch.linalg.vector_norm(gripper_ang_vel, dim=-1)

        piece_pos = self._stack_piece_states("get_pos")
        piece_quat = self._stack_piece_states("get_quat")
        target_pos = self.piece_target_pos[:, None, :].expand(-1, self.batch_size, -1)
        target_quat = self._piece_target_pose[:, 3:].unsqueeze(1).expand(-1, self.batch_size, -1)

        pos_error = piece_pos - target_pos
        pos_error_norm = torch.linalg.vector_norm(pos_error, dim=-1)
        rot_error_vec = self._rotation_error_rotvec(piece_quat, target_quat)
        rot_error_norm = torch.linalg.vector_norm(rot_error_vec, dim=-1)

        piecewise_distance = pos_error_norm.mean(dim=0)
        piecewise_rot_distance = rot_error_norm.mean(dim=0)
        distance_reward = -self.w_pos * piecewise_distance
        rotation_reward = -self.w_rot * piecewise_rot_distance
        reward_dict["piecewise_distance"] = piecewise_distance.detach().mean().cpu()
        reward_dict["piecewise_distance_reward"] = distance_reward
        reward_dict["piecewise_rot_distance"] = piecewise_rot_distance.detach().mean().cpu()
        reward_dict["piecewise_rot_distance_reward"] = rotation_reward

        piece_pos_batch = piece_pos.permute(1, 0, 2)
        pos_error_batch = pos_error_norm.permute(1, 0)
        rot_error_batch = rot_error_norm.permute(1, 0)
        active_piece_pos = piece_pos_batch[batch_indices, self.active_piece_idx]
        active_pos_error = pos_error_batch[batch_indices, self.active_piece_idx]
        active_rot_error = rot_error_batch[batch_indices, self.active_piece_idx]

        gripper_pos_batch = gripper_pos.permute(1, 0, 2)
        gripper_to_piece = gripper_pos_batch - active_piece_pos.unsqueeze(1)
        gripper_distances = torch.linalg.vector_norm(gripper_to_piece, dim=-1)
        closest_dist, closest_robot_idx = torch.min(gripper_distances, dim=1)
        needs_pick = closest_dist > self.pick_threshold
        far_from_target = active_pos_error > self.place_threshold

        phase1_reward = torch.where(
            needs_pick,
            -self.w_phase1 * closest_dist,
            torch.zeros_like(closest_dist),
        )
        phase2_mask = (~needs_pick) & far_from_target
        phase2_reward = torch.where(
            phase2_mask,
            -self.w_phase2 * active_pos_error,
            torch.zeros_like(active_pos_error),
        )
        phase3_mask = (~needs_pick) & (~phase2_mask)
        phase3_reward = torch.where(
            phase3_mask,
            -self.w_phase3_pos * active_pos_error - self.w_phase3_rot * active_rot_error,
            torch.zeros_like(active_pos_error),
        )

        reward_dict["phase1_min_gripper_piece"] = closest_dist.detach().mean().cpu()
        reward_dict["phase1_reward"] = phase1_reward
        reward_dict["phase2_reward"] = phase2_reward
        reward_dict["phase3_reward"] = phase3_reward

        piece_completed = (pos_error_norm <= self.place_threshold) & (rot_error_norm <= self.rotation_threshold)
        piece_completed_batch = piece_completed.permute(1, 0)
        newly_completed = piece_completed_batch & (~self.piece_completed_flags)
        self.piece_completed_flags |= piece_completed_batch
        completion_bonus = newly_completed.float().sum(dim=1) * self.completion_bonus
        reward_dict["completion_bonus"] = completion_bonus

        for env_idx in range(self.batch_size):
            current_idx = int(self.active_piece_idx[env_idx].item())
            while current_idx < self.num_pieces - 1 and piece_completed[current_idx, env_idx]:
                current_idx += 1
            self.active_piece_idx[env_idx] = current_idx

        contact_force_entries = []
        for robot in self.robots:
            contact_info = robot.get_contacts()
            force_a = torch.as_tensor(contact_info["force_a"], dtype=torch.float32, device=self.device)
            force_magnitudes = torch.linalg.vector_norm(force_a, dim=-1)
            contact_force_entries.append(force_magnitudes.sum(dim=1))
        contact_force_sum = torch.stack(contact_force_entries, dim=0).sum(dim=0)
        reward_dict["contact_force_sum"] = contact_force_sum.detach().mean().cpu()
        reward_dict["contact_force_sum_reward"] = -0.1 * contact_force_sum

        links_contact_entries = []
        for robot in self.robots:
            links_contact_force = torch.as_tensor(robot.get_links_net_contact_force(), device=self.device)
            link_force_magnitudes = torch.linalg.vector_norm(links_contact_force, dim=-1)
            links_contact_entries.append(torch.mean(link_force_magnitudes, dim=1))
        links_contact_force = torch.stack(links_contact_entries, dim=0).mean(dim=0)
        reward_dict["links_force_sum"] = links_contact_force.detach().mean().cpu()
        reward_dict["links_force_sum_reward"] = -1.2 * links_contact_force

        reward_dict["gripper_velocity"] = gripper_vel_mag.detach().mean().cpu()
        reward_dict["gripper_velocity_reward"] = -0.1 * gripper_vel_mag.sum(dim=0)

        reward_dict["gripper_angular_velocity"] = gripper_ang_mag.detach().mean().cpu()
        reward_dict["gripper_angular_velocity_reward"] = -0.01 * gripper_ang_mag.sum(dim=0)

        joint_velocity = self._stack_dof_states("get_dofs_velocity")
        joint_velocity = joint_velocity.permute(1, 0, 2).reshape(self.batch_size, -1)
        joint_velocity = torch.linalg.vector_norm(joint_velocity, dim=-1)
        joint_velocity_sq = joint_velocity**2
        reward_dict["joint_velocity_sq"] = joint_velocity_sq.detach().mean().cpu()
        reward_dict["joint_velocity_sq_reward"] = -0.0009 * joint_velocity_sq

        forearm_pos = self._stack_link_states(self.forearm_link_name, "get_pos")
        forearm_pos_batch = forearm_pos.permute(1, 0, 2)
        best_forearm_pos = forearm_pos_batch[batch_indices, closest_robot_idx]
        forearm_height = best_forearm_pos[:, 2]
        reward_dict["forearm_height"] = forearm_height.detach().mean().cpu()
        reward_dict["forearm_height_reward"] = 0.02 * forearm_height

        reward = torch.zeros(self.batch_size, device=self.device)
        for key in list(reward_dict.keys()):
            if "reward" in key or key.endswith("bonus"):
                reward += reward_dict[key]
                reward_dict[key] = reward_dict[key].mean().detach().cpu().item()

        self._update_debug_markers(gripper_pos)
        return reward, reward_dict, active_pos_error.clone().detach()

    def step(self, actions, record: bool = False):
        actions = self._convert_actions(actions)
        for idx, robot in enumerate(self.robots):
            robot_actions = actions[:, idx, :]
            angles = actions_to_angles(robot_actions, self.joint_lower, self.joint_upper)
            robot.control_dofs_position(angles, self.dofs_idx[idx])
        self.scene.step()
        self.current_step += 1

        obs = self.get_obs()
        reward, reward_dict, active_distance = self.compute_reward()
        terminated = bool(torch.any(active_distance <= self.success_threshold).item())
        truncated = bool(self.current_step >= self.max_steps)

        if self.record and record:
            self.cam.render()

        info = {
            "reward_terms": reward_dict,
            "best_distance": active_distance.detach().cpu().numpy(),
            "success": terminated,
            "is_success": terminated,
        }
        obs_np = {}
        for k, v in obs.items():
            if isinstance(v, dict):
                obs_np[k] = v
            elif isinstance(v, np.ndarray):
                obs_np[k] = v
            else:
                obs_np[k] = v.detach().cpu().numpy()

        return obs_np, float(torch.mean(reward).item()), terminated, truncated, info

    def save_video(self, filename):
        if not self.record:
            gs.logger.warning("Video recording requested but record=False for this environment")
            return
        self.cam.stop_recording(save_to_filename=filename, fps=30)

    def _generate_pair_colors(self, count):
        if count <= 0:
            return []
        colors = []
        golden_ratio = 0.61803398875
        for idx in range(count):
            hue = (idx * golden_ratio) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
            colors.append(rgb)
        return colors

    def _update_debug_markers(self, gripper_pos: torch.Tensor):
        if self.scene is None:
            return
        debug_envs = min(self.debug_env_count, self.batch_size)
        if debug_envs <= 0:
            return
        self.scene.clear_debug_objects()
        for env_idx in range(debug_envs):
            for spec in self.piece_specs:
                self.scene.draw_debug_sphere(spec.initial.pos, radius=0.0075, color=spec.color)
                self.scene.draw_debug_sphere(spec.target.pos, radius=0.0095, color=spec.color)
            gripper_color = tuple(self.env_colors[env_idx % len(self.env_colors)])
            for robot_idx in range(self.num_robots):
                grip = gripper_pos[robot_idx, env_idx].detach().cpu().tolist()
                radius = 0.017 if robot_idx == 0 else 0.012
                self.scene.draw_debug_sphere(grip, radius=radius, color=gripper_color)

    def _stack_dof_states(self, attribute: str) -> torch.Tensor:
        states = []
        for robot, dof_idx in zip(self.robots, self.dofs_idx):
            method = getattr(robot, attribute)
            states.append(torch.as_tensor(method(dof_idx), device=self.device, dtype=torch.float32))
        return torch.stack(states, dim=0)

    def _stack_link_states(self, link_name: str, attribute: str) -> torch.Tensor:
        states = []
        for robot in self.robots:
            link = robot.get_link(link_name)
            method = getattr(link, attribute)
            states.append(torch.as_tensor(method(), device=self.device, dtype=torch.float32))
        return torch.stack(states, dim=0)

    def _stack_piece_states(self, attribute: str) -> torch.Tensor:
        if not self.piece_entities:
            return torch.zeros(0, self.batch_size, 3, device=self.device)
        states: List[torch.Tensor] = []
        for piece in self.piece_entities:
            method = getattr(piece, attribute)
            states.append(torch.as_tensor(method(), device=self.device, dtype=torch.float32))
        return torch.stack(states, dim=0)

    def _load_joint_limits(self, robot_path: Path):
        tree = ET.parse(robot_path)
        root = tree.getroot()
        limits = {}
        for joint in root.findall(".//joint"):
            name = joint.attrib.get("name")
            rng = joint.attrib.get("range")
            if not name or not rng:
                continue
            lo, hi = map(float, rng.split())
            limits[name] = (lo, hi)
        missing = [name for name in self.jnt_names if name not in limits]
        if missing:
            raise ValueError(f"Missing joint limits for: {missing}")
        return limits

    def _build_piece_specs(self, so101_dir: Path, layout_override: list[dict] | None = None) -> List[PieceSpec]:
        assets_dir = so101_dir / "assets"
        if layout_override:
            colors = self._generate_pair_colors(max(len(layout_override), 1))
            specs: list[PieceSpec] = []
            for idx, entry in enumerate(layout_override):
                name = entry.get("name", f"piece_{idx}")
                mesh_path = Path(entry["mesh_file"])
                if not mesh_path.is_absolute():
                    mesh_path = assets_dir / mesh_path
                init = entry["initial"]
                tgt = entry["target"]
                initial = PiecePose(pos=tuple(init["pos"]), quat=tuple(init.get("quat", (1.0, 0.0, 0.0, 0.0))))
                target = PiecePose(pos=tuple(tgt["pos"]), quat=tuple(tgt.get("quat", (1.0, 0.0, 0.0, 0.0))))
                # color = tuple(entry.get("color", colors[idx % len(colors)]))
                color = (0.98, 0.38, 0.00)
                scale = float(entry.get("scale", 1.0))
                specs.append(PieceSpec(name=name, mesh_file=mesh_path, initial=initial, target=target, color=color, scale=scale))
            return specs

        resting_z = self.print_bed_top_z + 0.03
        target_z = self.print_bed_top_z + 0.02
        target_y = 0.30
        identity = (1.0, 0.0, 0.0, 0.0)
        rot_z_90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        rot_z_180 = (0.0, 0.0, 0.0, 1.0)
        rot_y_180 = (0.0, 0.0, 1.0, 0.0)
        return [
            PieceSpec(
                name="motor_holder_so101_base_v1",
                mesh_file=assets_dir / "motor_holder_so101_base_v1.glb",
                initial=PiecePose(pos=(0, -0.1, resting_z), quat=identity),
                target=PiecePose(pos=(-0.05, target_y + 0.05, target_z), quat=rot_z_90),
                color=(0.98, 0.38, 0.00),
            ),
            PieceSpec(
                name="moving_jaw_so101_v1",
                mesh_file=assets_dir / "moving_jaw_so101_v1.glb",
                initial=PiecePose(pos=(0.08, -0.1, resting_z), quat=identity),
                target=PiecePose(pos=(0.15, target_y - 0.05, target_z), quat=rot_y_180),
                color=(0.98, 0.38, 0.00),
            ),
            PieceSpec(
                name="under_arm_so101_v1",
                mesh_file=assets_dir / "under_arm_so101_v1.glb",
                initial=PiecePose(pos=(-0.08, -0.1, resting_z), quat=identity),
                target=PiecePose(pos=(-0.25, target_y - 0.10, target_z), quat=identity),
                color=(0.98, 0.38, 0.00),
            )
        ]

    def _reset_pieces(self):
        if not self.piece_entities:
            return
        for idx, entity in enumerate(self.piece_entities):
            pose = self._piece_initial_pose[idx]
            pos = pose[:3].unsqueeze(0).repeat(self.batch_size, 1).to(device=gs.device)
            quat = pose[3:].unsqueeze(0).repeat(self.batch_size, 1).to(device=gs.device)
            entity.set_pos(pos, zero_velocity=True)
            entity.set_quat(quat, zero_velocity=True)

    def _quat_to_rotvec(self, quat: torch.Tensor) -> torch.Tensor:
        quat = torch.as_tensor(quat, device=self.device, dtype=torch.float32)
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-6)
        quat = quat / norm
        quat = torch.where(quat[..., :1] < 0, -quat, quat)
        xyz = quat[..., 1:]
        w = torch.clamp(quat[..., :1], -1.0, 1.0)
        xyz_norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(xyz_norm, w)
        axis = torch.zeros_like(xyz)
        nonzero = xyz_norm.squeeze(-1) > 1e-6
        axis[nonzero] = xyz[nonzero] / xyz_norm[nonzero]
        rotvec = torch.zeros_like(xyz)
        rotvec[nonzero] = axis[nonzero] * angle[nonzero]
        rotvec[~nonzero] = 2.0 * xyz[~nonzero]
        return rotvec

    def _rotation_error_rotvec(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = self._quat_multiply(target, self._quat_conjugate(current))
        return self._quat_to_rotvec(delta)

    @staticmethod
    def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
        conj = quat.clone()
        conj[..., 1:] = -conj[..., 1:]
        return conj

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    def _convert_actions(self, actions) -> torch.Tensor:
        """
        Convert incoming actions to (batch, num_robots, dofs_per_robot) normalized to [-1, 1].
        Accepts dict with robot-range keys or tensors/arrays.
        """
        if isinstance(actions, dict):
            per_joint = []
            for key in self.action_keys:
                if key not in actions:
                    raise KeyError(f"Missing action key '{key}' in action dict.")
                val = actions[key]
                tensor = torch.as_tensor(val, device=self.device, dtype=torch.float32)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                per_joint.append(tensor)
            actions = torch.stack(per_joint, dim=1)

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        elif not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        if actions.shape[-1] != self.act_dim:
            raise ValueError(f"Expected action dim {self.act_dim}, got {actions.shape}")

        actions = actions.reshape(-1, self.num_robots, self.dofs_per_robot)
        treat_as_normalized = torch.max(torch.abs(actions)) <= 1.05
        normalized = torch.empty_like(actions, dtype=torch.float32)
        for j, joint in enumerate(self.jnt_names):
            joint_vals = actions[:, :, j]
            if treat_as_normalized:
                norm = torch.clamp(joint_vals, -1.0, 1.0)
            elif joint == "gripper":
                norm = torch.clamp(joint_vals, 0.0, 100.0) / 50.0 - 1.0
            else:
                norm = torch.clamp(joint_vals, -100.0, 100.0) / 100.0
            normalized[:, :, j] = norm

        return normalized


def actions_to_angles(actions: torch.Tensor, joint_lower: torch.Tensor, joint_upper: torch.Tensor):
    normalized = torch.clamp(actions, -1.0, 1.0).to(dtype=torch.float32)
    lower = joint_lower.to(device=normalized.device, dtype=normalized.dtype)
    upper = joint_upper.to(device=normalized.device, dtype=normalized.dtype)
    return lower + 0.5 * (normalized + 1.0) * (upper - lower)


Environment = MovePiecesEnv

try:
    gym.register(
        id="my_environment/MovePiecesEnv-v0",
        entry_point="lerobot.rl_custom.envs.movepieces:MovePiecesEnv",
    )
except GymError:
    pass
