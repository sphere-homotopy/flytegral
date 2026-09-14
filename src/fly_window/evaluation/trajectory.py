from __future__ import annotations

from dataclasses import asdict

import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv


def rollout_deterministic(policy, env_config: EnvironmentConfig, *, seed: int) -> dict:
    policy.eval()
    env = WindowExitEnv(env_config)
    observation, state = env.reset(seed)
    frames = [
        {
            "step": 0,
            **asdict(state),
            "action": None,
            "reward": 0.0,
            "collision": False,
            "done": False,
        }
    ]
    total_reward = 0.0

    with torch.no_grad():
        while True:
            observation_tensor = torch.as_tensor(
                observation[None, :], dtype=torch.float32
            )
            output = policy(observation_tensor)
            action_tensor = policy.action_from_latent(output.mean_action)[0]
            action = action_tensor.cpu().numpy()
            result = env.step(action)
            total_reward += result.reward
            done = result.terminated or result.truncated
            frames.append(
                {
                    "step": result.state.step_count,
                    **asdict(result.state),
                    "action": [float(action[0]), float(action[1])],
                    "reward": float(result.reward),
                    "collision": bool(result.collision),
                    "done": bool(done),
                }
            )
            observation = result.observation
            if done:
                return {
                    "seed": int(seed),
                    "success": bool(result.success),
                    "steps": int(result.state.step_count),
                    "total_reward": float(total_reward),
                    "frames": frames,
                }
