import json
from pathlib import Path

import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy
from fly_window.training.run import DEVELOPMENT_SEEDS, TrainingConfig, run_training


class TwoStepVectorEnv:
    def __init__(self, env_count: int):
        self.env_count = env_count
        self.step_count = 0
        self.reset_seeds: list[int] = []

    def reset(self, seeds):
        self.reset_seeds = [int(seed) for seed in seeds]
        assert len(self.reset_seeds) == self.env_count
        self.step_count = 0
        return np.zeros((self.env_count, 16), dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.env_count, 2)
        self.step_count += 1
        done = self.step_count >= 2
        observations = np.full(
            (self.env_count, 16), 0.1 * self.step_count, dtype=np.float32
        )
        rewards = np.ones(self.env_count, dtype=np.float32)
        dones = np.full(self.env_count, done, dtype=np.bool_)
        successes = np.full(self.env_count, done, dtype=np.bool_)
        return observations, rewards, dones, successes


def _toy_graph() -> ConnectomeGraph:
    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 1], dtype=np.int64),
        dst_idx=np.asarray([1, 2], dtype=np.int64),
        weights=np.asarray([3, 5], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_training_v1_config_pins_approved_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    config = json.loads((repo_root / "configs/training_v1.json").read_text())

    assert config == {
        "seed": 20260914,
        "parallel_envs": 16,
        "rollout_steps": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "learning_rate": 0.0003,
        "ppo_epochs": 4,
        "minibatch_size": 512,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_environment_steps": 2000000,
        "checkpoint_every_steps": 50000,
        "eval_every_steps": 50000,
    }


def test_tiny_training_run_writes_reproducible_checkpoint_and_metrics(tmp_path):
    graph = _toy_graph()
    recurrent = to_sparse_recurrent(graph)
    vector_env = TwoStepVectorEnv(env_count=2)
    config = TrainingConfig(
        seed=123,
        parallel_envs=2,
        rollout_steps=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        learning_rate=3e-4,
        ppo_epochs=1,
        minibatch_size=4,
        value_coef=0.5,
        entropy_coef=0.01,
        max_environment_steps=4,
        checkpoint_every_steps=4,
        eval_every_steps=1000,
    )

    result = run_training(
        config=config,
        graph=graph,
        recurrent=recurrent,
        vector_env=vector_env,
        run_root=tmp_path,
        graph_manifest_hash="graph-sha-123",
        git_sha="deadbeef",
        control_kind="biological",
    )

    assert result.total_environment_steps == 4
    assert vector_env.reset_seeds == [123, 124]
    assert result.run_dir.parent == tmp_path

    saved_config = json.loads((result.run_dir / "config.json").read_text())
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    metric_lines = (result.run_dir / "metrics.jsonl").read_text().strip().splitlines()
    checkpoints = sorted((result.run_dir / "checkpoints").glob("*.pt"))

    assert saved_config["seed"] == 123
    assert manifest["graph_manifest_hash"] == "graph-sha-123"
    assert manifest["git_sha"] == "deadbeef"
    assert manifest["control_kind"] == "biological"
    assert manifest["seed"] == 123
    assert len(metric_lines) >= 1
    assert json.loads(metric_lines[-1])["total_environment_steps"] == 4
    assert len(checkpoints) == 1

    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    for key in [
        "policy_state",
        "value_head_state",
        "optimizer_state",
        "total_environment_steps",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "config",
        "graph_manifest_hash",
        "git_sha",
    ]:
        assert key in checkpoint
    assert checkpoint["total_environment_steps"] == 4
    assert checkpoint["graph_manifest_hash"] == "graph-sha-123"
    assert checkpoint["git_sha"] == "deadbeef"
    assert checkpoint["optimizer_state"]["state"]

    fixed_obs = torch.full((1, 16), 0.25)
    first = ConnectomePolicy(graph, recurrent)
    first.load_state_dict(checkpoint["policy_state"])
    second = ConnectomePolicy(graph, recurrent)
    second.load_state_dict(checkpoint["policy_state"])
    with torch.no_grad():
        first_mean = first(fixed_obs).mean_action
        second_mean = second(fixed_obs).mean_action
    assert torch.equal(first_mean, second_mean)


def test_training_development_seeds_are_fixed_and_never_overlap_held_out_range():
    assert DEVELOPMENT_SEEDS == tuple(range(9000, 9032))
    assert set(DEVELOPMENT_SEEDS).isdisjoint(range(10000, 10100))
