from __future__ import annotations

import argparse
import json

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv
from fly_window.env.oracle import oracle_action


def run(seed: int) -> dict[str, object]:
    env = WindowExitEnv(EnvironmentConfig())
    _, state = env.reset(seed=seed)
    result = None

    for _ in range(env.config.reward.max_steps):
        result = env.step(oracle_action(state, env.config.room))
        state = result.state
        if result.terminated or result.truncated:
            break

    if result is None:
        raise RuntimeError("environment produced no steps")

    return {
        "seed": seed,
        "steps": result.state.step_count,
        "success": result.success,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "final_state": {
            "x": result.state.x,
            "y": result.state.y,
            "heading": result.state.heading,
            "speed": result.state.speed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic Fly Window oracle smoke episode.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = run(args.seed)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
