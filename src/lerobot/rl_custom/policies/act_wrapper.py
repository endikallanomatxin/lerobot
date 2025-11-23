from __future__ import annotations

import torch

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
except Exception:  # pragma: no cover
    ACTPolicy = None  # type: ignore
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
)


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
        # Try to load matching pre/post processors for normalization.
        self.preprocessor = None
        self.postprocessor = None
        try:
            self.preprocessor = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=policy_path,
                config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            )
        except Exception:
            self.preprocessor = None
        try:
            self.postprocessor = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=policy_path,
                config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            )
        except Exception:
            self.postprocessor = None

    @torch.inference_mode()
    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: D401
        obs = obs_dict
        if self.preprocessor is not None:
            obs = self.preprocessor(obs)
        action = self.policy.select_action(obs)
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        # Clamp to env-normalized range in case postprocessor returns physical units.
        return torch.clamp(action, -1.0, 1.0)
