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
import warnings
import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

from lerobot.rl_custom.envs import LeRobotGymEnvWrapper
from lerobot.rl_custom.envs import movepieces  # registers my_environment/MovePiecesEnv-v0
from lerobot.rl_custom.policies import ACTAsRLPolicy, RandomPolicy


def parse_args():
    p = argparse.ArgumentParser(description="Roll out MovePiecesEnv with a simple policy.")
    p.add_argument(
        "--device",
        default="auto",
        help="Device for policy / tensors. Use 'auto' to prefer CUDA, then MPS, then CPU.",
    )
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for the env.")
    p.add_argument("--max_steps", type=int, default=300, help="Episode horizon.")
    p.add_argument("--steps", type=int, default=100, help="Total rollout steps to run.")
    p.add_argument("--show_viewer", action="store_true", help="Enable Genesis viewer.")
    p.add_argument(
        "--show_cameras",
        action="store_true",
        help="Open pop-up windows for the three Genesis cameras (left/right wrist, overhead).",
    )
    p.add_argument("--policy_path", type=Path, default=None, help="Optional ACT policy checkpoint.")
    p.add_argument("--piece_layout", type=Path, default=None, help="Optional JSON layout file for pieces.")
    return p.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    # Suppress torchvision video deprecation warnings (not relevant for this harness).
    warnings.filterwarnings(
        "ignore",
        message="The video decoding and encoding capabilities of torchvision are deprecated",
        module="torchvision.io._video_deprecation_warning",
    )


def show_camera_popups(obs: dict, enabled: bool):
    if not enabled or not _HAS_CV2:
        return

    displayed = False
    for key, img in obs.items():
        if not key.startswith("observation.images."):
            continue
        cam_name = key.split("observation.images.", 1)[1]
        if torch.is_tensor(img):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = np.asarray(img)

        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.ndim == 3 and img_np.shape[0] in (3, 4):
            img_np = np.transpose(img_np[:3], (1, 2, 0))

        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"camera-{cam_name}", img_bgr)
        displayed = True

    if displayed:
        cv2.waitKey(1)


def main():
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    env = gym.make(
        "my_environment/MovePiecesEnv-v0",
        device=device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        show_viewer=args.show_viewer,
        show_cameras=args.show_cameras,
        disable_env_checker=True,  # mixed obs/action dicts; we validate manually
        piece_layout_file=args.piece_layout,
    )
    env = LeRobotGymEnvWrapper(env)

    # Action dim comes from env.action_space
    action_dim = env.env.action_space.shape[-1]
    policy = make_policy(args.policy_path, action_dim=action_dim, device=device)
    policy.to(device)
    policy.eval()

    if args.show_cameras and not _HAS_CV2:
        print("OpenCV not available; camera pop-ups are disabled.")

    try:
        obs, _ = env.reset()
        show_camera_popups(obs, enabled=args.show_cameras)

        successes = 0
        for step in range(args.steps):
            with torch.inference_mode():
                action = policy(obs)
            obs, reward, done, info = env.step(action)

            if info.get("is_success", False):
                successes += 1

            if done:
                obs, _ = env.reset()

            show_camera_popups(obs, enabled=args.show_cameras)

            if (step + 1) % 20 == 0:
                print(
                    f"[step {step+1}] reward={reward:.3f} success_rate={(successes/(step+1)):.2%} "
                    f"done={done}"
                )

        print(f"Finished {args.steps} steps. Successes: {successes}/{args.steps}")
    finally:
        if args.show_cameras and _HAS_CV2:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
