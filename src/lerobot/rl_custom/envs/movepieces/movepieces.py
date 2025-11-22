import colorsys
from pathlib import Path
import xml.etree.ElementTree as ET

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.error import Error as GymError
from gymnasium.utils import seeding


class MovePiecesEnv(gym.Env):
    """
    Genesis bimanual environment wrapped as a Gymnasium Env.
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
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.record = record
        self.debug_env_count = min(4, batch_size)
        self.env_colors = self._generate_pair_colors(self.debug_env_count)
        self._max_episode_steps = max_steps
        self.np_random = None

        # Assets live alongside this module in ./assets/SO101/
        assets_root = Path(__file__).resolve().parent / "assets"
        robot_path = assets_root / "SO101" / "so101_new_calib.xml"

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

        if self.record:
            self.cam = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2, 2, 1.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=False,
            )

        self.plane = self.scene.add_entity(gs.morphs.Plane())

        self.robot_base_configs = [
            {"pos": (0.25, 0.0, 0.0), "quat": (0.0, 0.0, 0.0, 1.0)},
            {"pos": (-0.25, 0.0, 0.0), "quat": (1.0, 0.0, 0.0, 0.0)},
        ]
        self.robots = [
            self.scene.add_entity(
                gs.morphs.MJCF(file=str(robot_path), pos=config["pos"], quat=config.get("quat"))
            )
            for config in self.robot_base_configs
        ]
        self.num_robots = len(self.robots)

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

        self.current_step = 0

        self.obs_dim = self.num_robots * (2 * self.dofs_per_robot + 6) + 3
        self.act_dim = self.num_robots * self.dofs_per_robot
        self.observation_space = spaces.Dict(
            {
                "environment_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.batch_size, self.obs_dim),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.batch_size, self.act_dim),
            dtype=np.float32,
        )
        self.success_threshold = 0.02
        self.num_envs = batch_size

        if self.record:
            self.cam.start_recording()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)

        ctrl_pos = torch.zeros(self.dofs_per_robot)
        tiled_ctrl = torch.tile(ctrl_pos, (self.batch_size, 1))
        for robot, dof_idx in zip(self.robots, self.dofs_idx):
            robot.control_dofs_position(tiled_ctrl, dof_idx)

        new_target_pos = torch.tensor([0.2, -0.15, 0.0]) + torch.rand(self.batch_size, 3) * torch.tensor(
            [0.2, 0.30, 0.30]
        )
        self.target_pos = new_target_pos.to(self.device)
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
        gripper_vel = self._stack_link_states(self.gripper_link_name, "get_vel")
        target_pos = torch.as_tensor(self.target_pos, device=self.device, dtype=torch.float32)

        def flatten_robot_tensor(tensor):
            return tensor.permute(1, 0, 2).reshape(self.batch_size, -1)

        obs = torch.cat(
            [
                flatten_robot_tensor(dof_ang),
                flatten_robot_tensor(dof_vel),
                flatten_robot_tensor(gripper_pos),
                flatten_robot_tensor(gripper_vel),
                target_pos,
            ],
            dim=1,
        )
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e4, neginf=-1e4)
        return {"environment_state": obs.detach().cpu().numpy().astype(np.float32)}

    def compute_reward(self):
        reward_dict = {}

        gripper_pos = self._stack_link_states(self.gripper_link_name, "get_pos")
        target_pos = torch.as_tensor(self.target_pos, device=self.device)
        distance = target_pos.unsqueeze(0) - gripper_pos
        distance_magnitude = torch.linalg.vector_norm(distance, dim=-1)
        best_distance, best_robot_idx = torch.min(distance_magnitude, dim=0)
        reward_dict["distance"] = torch.mean(best_distance).clone().detach().cpu()
        reward_dict["distance_reward"] = -best_distance

        gripper_velocity = self._stack_link_states(self.gripper_link_name, "get_vel")
        gripper_velocity_mag = torch.linalg.vector_norm(gripper_velocity, dim=-1)
        batch_indices = torch.arange(self.batch_size, device=self.device)
        distance_batch = distance.permute(1, 0, 2)
        velocity_batch = gripper_velocity.permute(1, 0, 2)
        best_distance_vec = distance_batch[batch_indices, best_robot_idx]
        best_gripper_velocity = velocity_batch[batch_indices, best_robot_idx]
        best_gripper_velocity_mag = torch.linalg.vector_norm(best_gripper_velocity, dim=-1)
        direction_similarity = torch.nn.functional.cosine_similarity(
            best_gripper_velocity, best_distance_vec, dim=-1, eps=1e-6
        )
        reward_dict["direction_similarity"] = torch.mean(direction_similarity).clone().detach().cpu()
        reward_dict["direction_similarity_reward"] = (
            1.2 * direction_similarity * (best_gripper_velocity_mag.clone().detach()) ** 2 * best_distance.clone().detach()
        )

        contact_force_entries = []
        for robot in self.robots:
            contact_info = robot.get_contacts()
            force_a = torch.as_tensor(contact_info["force_a"], dtype=torch.float32, device=self.device)
            force_magnitudes = torch.linalg.vector_norm(force_a, dim=-1)
            contact_force_entries.append(force_magnitudes.sum(dim=1))
        contact_force_sum = torch.stack(contact_force_entries, dim=0).sum(dim=0)
        reward_dict["contact_force_sum"] = torch.mean(contact_force_sum).clone().detach().cpu()
        reward_dict["contact_force_sum_reward"] = -0.1 * contact_force_sum

        links_contact_entries = []
        for robot in self.robots:
            links_contact_force = torch.as_tensor(robot.get_links_net_contact_force(), device=self.device)
            link_force_magnitudes = torch.linalg.vector_norm(links_contact_force, dim=-1)
            links_contact_entries.append(torch.mean(link_force_magnitudes, dim=1))
        links_contact_force = torch.stack(links_contact_entries, dim=0).mean(dim=0)
        reward_dict["links_force_sum"] = torch.mean(links_contact_force).clone().detach().cpu()
        reward_dict["links_force_sum_reward"] = -1.2 * links_contact_force

        reward_dict["gripper_velocity"] = torch.mean(gripper_velocity_mag).clone().detach().cpu()
        reward_dict["gripper_velocity_reward"] = -0.1 * gripper_velocity_mag.sum(dim=0)

        gripper_angular_velocity = self._stack_link_states(self.gripper_link_name, "get_ang")
        gripper_angular_velocity = torch.linalg.vector_norm(gripper_angular_velocity, dim=-1)
        reward_dict["gripper_angular_velocity"] = torch.mean(gripper_angular_velocity).clone().detach().cpu()
        reward_dict["gripper_angular_velocity_reward"] = -0.01 * gripper_angular_velocity.sum(dim=0)

        joint_velocity = self._stack_dof_states("get_dofs_velocity")
        joint_velocity = joint_velocity.permute(1, 0, 2).reshape(self.batch_size, -1)
        joint_velocity = torch.linalg.vector_norm(joint_velocity, dim=-1)
        joint_velocity_sq = joint_velocity**2
        reward_dict["joint_velocity_sq"] = torch.mean(joint_velocity_sq).clone().detach().cpu()
        reward_dict["joint_velocity_sq_reward"] = -0.0009 * joint_velocity_sq

        forearm_pos = self._stack_link_states(self.forearm_link_name, "get_pos")
        forearm_pos_batch = forearm_pos.permute(1, 0, 2)
        best_forearm_pos = forearm_pos_batch[batch_indices, best_robot_idx]
        forearm_height = best_forearm_pos[:, 2]
        reward_dict["forearm_height"] = torch.mean(forearm_height).clone().detach().cpu()
        reward_dict["forearm_height_reward"] = 0.02 * forearm_height

        reward = torch.zeros(self.batch_size, device=self.device)
        for key in reward_dict:
            if "reward" in key:
                reward += reward_dict[key].clone()
                reward_dict[key] = reward_dict[key].mean().clone().detach().cpu().item()
            elif torch.is_tensor(reward_dict[key]):
                reward_dict[key] = reward_dict[key].mean().clone().detach().cpu().item()

        self._update_debug_markers(gripper_pos)
        return reward, reward_dict, best_distance.clone().detach()

    def step(self, actions, record: bool = False):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        actions = actions.reshape(self.batch_size, self.num_robots, self.dofs_per_robot)
        for idx, robot in enumerate(self.robots):
            robot_actions = actions[:, idx, :]
            angles = actions_to_angles(robot_actions, self.joint_lower, self.joint_upper)
            robot.control_dofs_position(angles, self.dofs_idx[idx])
        self.scene.step()
        self.current_step += 1

        obs = self.get_obs()
        reward, reward_dict, best_distance = self.compute_reward()
        terminated = bool(torch.any(best_distance <= self.success_threshold).item())
        truncated = bool(self.current_step >= self.max_steps)

        if self.record and record:
            self.cam.render()

        info = {
            "reward_terms": reward_dict,
            "best_distance": best_distance.detach().cpu().numpy(),
            "success": terminated,
            "is_success": terminated,
        }
        return (
            {k: v if isinstance(v, np.ndarray) else v.detach().cpu().numpy() for k, v in obs.items()},
            float(torch.mean(reward).item()),
            terminated,
            truncated,
            info,
        )

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
        if self.scene is None or not hasattr(self, "target_pos"):
            return
        debug_envs = min(self.debug_env_count, self.batch_size)
        if debug_envs <= 0:
            return
        self.scene.clear_debug_objects()
        for env_idx in range(debug_envs):
            color = tuple(self.env_colors[env_idx % len(self.env_colors)])
            target = self.target_pos[env_idx].detach().cpu().tolist()
            self.scene.draw_debug_sphere(target, radius=0.01, color=color)
            for robot_idx in range(self.num_robots):
                grip = gripper_pos[robot_idx, env_idx].detach().cpu().tolist()
                radius = 0.017 if robot_idx == 0 else 0.012
                self.scene.draw_debug_sphere(grip, radius=radius, color=color)

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
