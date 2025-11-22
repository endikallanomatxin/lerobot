"""
Minimal training harness to roll out the custom MovePieces Genesis env.

This is a stub to verify end-to-end wiring (env -> wrapper -> policy).
It runs random actions by default; plug a pretrained ACT policy with
`--policy_path` to drive actions from BC instead. The env now speaks the same
action/observation dialect as the SO101 bimanual robot:
    - Actions: dict or tensor with 12 joints (`left/right_* .pos`), arms in [-100, 100], gripper in [0, 100].
    - Observations: per-joint keys plus `environment_state` for debugging.

Example:
python -m lerobot.rl_custom.train_genesis --device cuda --batch_size 16 --max_steps 300 --steps 400

"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
import logging

from lerobot.rl_custom.envs import LeRobotGymEnvWrapper
from lerobot.rl_custom.envs import movepieces  # registers my_environment/MovePiecesEnv-v0
from lerobot.rl_custom.policies import ACTAsRLPolicy, RandomPolicy


def parse_args():
    p = argparse.ArgumentParser(description="Roll out MovePiecesEnv with a simple policy.")
    p.add_argument("--device", default="cuda", help="Device for policy / tensors.")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for the env.")
    p.add_argument("--max_steps", type=int, default=300, help="Episode horizon.")
    p.add_argument("--steps", type=int, default=100, help="Total rollout steps to run.")
    p.add_argument("--show_viewer", action="store_true", help="Enable Genesis viewer.")
    p.add_argument("--policy_path", type=Path, default=None, help="Optional ACT policy checkpoint.")
    return p.parse_args()


def make_policy(policy_path: Path | None, action_dim: int, device: str):
    if policy_path is None:
        return RandomPolicy(action_dim=action_dim, device=device)
    return ACTAsRLPolicy(policy_path=str(policy_path), device=device)


def configure_logging():
    # Genesis mirrors logs to the Python logger; disable propagation to avoid double-printing.
    gen_logger = logging.getLogger("genesis")
    gen_logger.propagate = False
    gen_logger.handlers.clear()
    gen_logger.setLevel(logging.WARNING)


def main():
    configure_logging()
    args = parse_args()

    env = gym.make(
        "my_environment/MovePiecesEnv-v0",
        device=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        show_viewer=args.show_viewer,
        disable_env_checker=True,  # mixed obs/action dicts; we validate manually
    )
    env = LeRobotGymEnvWrapper(env)

    # Action dim comes from env.action_space
    action_dim = env.env.action_space.shape[-1]
    policy = make_policy(args.policy_path, action_dim=action_dim, device=args.device)
    policy.to(args.device)
    policy.eval()

    obs, _ = env.reset()

    successes = 0
    for step in range(args.steps):
        with torch.inference_mode():
            action = policy(obs)
        obs, reward, done, info = env.step(action)

        if info.get("is_success", False):
            successes += 1

        if done:
            obs, _ = env.reset()

        if (step + 1) % 20 == 0:
            print(
                f"[step {step+1}] reward={reward:.3f} success_rate={(successes/(step+1)):.2%} "
                f"done={done}"
            )

    print(f"Finished {args.steps} steps. Successes: {successes}/{args.steps}")


if __name__ == "__main__":
    main()
