from pathlib import Path

from fly_window.text.gate import GateConfig
from fly_window.text.pretraining import TextTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_launch_gate_uses_bounded_real_graph_probe_budget():
    config = GateConfig.from_json(REPO_ROOT / "configs" / "text_gate_v1.json")

    assert 1 <= config.replay_probe_count <= config.probe_count <= 16
    assert config.probe_max_tokens <= 96


def test_smoke_gate_is_strictly_cheaper_than_launch_gate():
    launch = GateConfig.from_json(REPO_ROOT / "configs" / "text_gate_v1.json")
    smoke = GateConfig.from_json(REPO_ROOT / "configs" / "text_gate_smoke.json")

    assert smoke.probe_count < launch.probe_count
    assert smoke.replay_probe_count <= smoke.probe_count
    assert smoke.probe_max_tokens < launch.probe_max_tokens


def test_cpu_pretraining_is_bounded_but_still_uses_cook_corpus():
    config = TextTrainingConfig.from_json(
        REPO_ROOT / "configs" / "text_training_cpu_v1.json"
    )

    assert config.curriculum_examples <= 512
    assert config.max_corpus_sequences is not None
    assert 128 <= config.max_corpus_sequences <= 512
    assert config.stage_b_epochs >= 1
    assert config.stage_c_epochs >= 1
