import gymnasium as gym
import numpy as np
import torch


class LeRobotGymEnvWrapper:
    """
    Thin adapter to expose Gymnasium envs with tensor observations/actions
    in the shape expected by LeRobot policies.

    - Input actions: torch.Tensor (batch, act_dim) in [-1, 1]
    - Output observations: dict[str, torch.Tensor]
    - Reward: float
    - Done: bool
    """

    def __init__(self, gym_env: gym.Env):
        self.env = gym_env

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return self._to_obs_dict(obs), info

    def step(self, action_tensor: torch.Tensor):
        action_np = action_tensor.detach().cpu().numpy()
        obs, rew, term, trunc, info = self.env.step(action_np)

        obs_dict = self._to_obs_dict(obs)
        rew_float = float(np.asarray(rew).mean())
        done = bool(np.asarray(term).any() or np.asarray(trunc).any())
        info["is_success"] = bool(np.asarray(info.get("is_success", term)).any())

        return obs_dict, rew_float, done, info

    @staticmethod
    def _to_obs_dict(obs):
        if isinstance(obs, dict):
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in obs.items()}
        return {"observation": torch.as_tensor(obs, dtype=torch.float32)}
