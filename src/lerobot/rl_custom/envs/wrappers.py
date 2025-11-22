import gymnasium as gym
import numpy as np
import torch
import einops


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
        if isinstance(action_tensor, dict):
            action_np = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in action_tensor.items()
            }
        else:
            action_np = action_tensor.detach().cpu().numpy() if torch.is_tensor(action_tensor) else action_tensor

        obs, rew, term, trunc, info = self.env.step(action_np)

        obs_dict = self._to_obs_dict(obs)
        rew_float = float(np.asarray(rew).mean())
        done = bool(np.asarray(term).any() or np.asarray(trunc).any())
        info["is_success"] = bool(np.asarray(info.get("is_success", term)).any())

        return obs_dict, rew_float, done, info

    @staticmethod
    def _to_obs_dict(obs):
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                if isinstance(v, dict):
                    if k == "pixels":
                        # Convert camera images to torch BCHW float32 in [0,1], with keys observation.images.*
                        for cam, img in v.items():
                            if isinstance(img, torch.Tensor):
                                img_t = img
                            else:
                                img_np = np.array(img, copy=True)
                                img_t = torch.as_tensor(img_np)
                            if img_t.ndim == 3:  # HWC
                                img_t = img_t.unsqueeze(0)
                            if img_t.shape[-1] == 4:  # drop alpha if present
                                img_t = img_t[..., :3]
                            # Convert to BCHW float in [0,1]
                            img_t = einops.rearrange(img_t, "b h w c -> b c h w").contiguous()
                            img_t = img_t.to(dtype=torch.float32) / 255.0
                            out[f"observation.images.{cam}"] = img_t
                    else:
                        out[k] = {cam: torch.as_tensor(img) if torch.is_tensor(img) else img for cam, img in v.items()}
                else:
                    out[k] = torch.as_tensor(v, dtype=torch.float32)
            if "environment_state" in out:
                out.setdefault("observation.environment_state", out["environment_state"])
            # If joint keys are present, stack them into observation.state in the env's action order.
            joint_tensors = []
            for key in obs.keys():
                if key.endswith(".pos") and not key.startswith("observation."):
                    joint_tensors.append(torch.as_tensor(obs[key], dtype=torch.float32))
            if joint_tensors:
                out["observation.state"] = torch.stack(joint_tensors, dim=1)
            return out
        return {"observation": torch.as_tensor(obs, dtype=torch.float32)}
