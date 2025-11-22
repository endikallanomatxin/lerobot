from __future__ import annotations

import torch

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
except Exception:  # pragma: no cover
    ACTPolicy = None  # type: ignore


class RandomPolicy(torch.nn.Module):
    """Uniform random policy in [-1, 1]."""

    def __init__(self, action_dim: int, device: str = "cpu"):
        super().__init__()
        self.action_dim = action_dim
        self.device = torch.device(device)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: D401
        batch = next(iter(obs_dict.values())).shape[0]
        return torch.rand((batch, self.action_dim), device=self.device) * 2.0 - 1.0


class ACTAsRLPolicy(torch.nn.Module):
    """
    Wrap a pretrained ACT policy so it can be used as an RL policy module.

    If no ACT dependency is installed, this wrapper will raise at init.
    """

    def __init__(self, policy_path: str, device: str = "cuda"):
        super().__init__()
        if ACTPolicy is None:
            raise ImportError("ACTPolicy import failed; install the ACT extras.")
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.to(device)
        self.device = device

    @torch.inference_mode()
    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: D401
        # ACT expects normalized obs keys; caller must supply correct dict.
        action = self.policy.select_action(obs_dict)
        return action
