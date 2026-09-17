import json

import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.pretraining import (
    TextTrainingConfig,
    choose_next_input,
    load_cook_sequences,
    teacher_forcing_loss,
)
from fly_window.text.vocabulary import build_v1_vocabulary


def _toy_policy(vocab_size: int) -> FlyTextPolicy:
    graph = ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 1], dtype=np.int64),
        dst_idx=np.asarray([1, 2], dtype=np.int64),
        weights=np.asarray([3, 3], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )
    return FlyTextPolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=vocab_size,
        sensory_dim=8,
        microsteps=2,
        leak=0.5,
    )


def test_load_cook_sequences_filters_provenance_and_deduplicates(tmp_path):
    vocabulary = build_v1_vocabulary()
    corpus = tmp_path / "cook.jsonl"
    rows = [
        {
            "account": "AnalysisFact",
            "tweet_url": "https://x.com/AnalysisFact/status/101",
            "tweet_datetime": "2026-01-01T00:00:00.000Z",
            "text": "the theorem is true .",
        },
        {
            "account": "AnalysisFact",
            "tweet_url": "https://x.com/AnalysisFact/status/101?s=20",
            "tweet_datetime": "2026-01-01T00:00:00.000Z",
            "text": "the theorem is true .",
        },
        {
            "account": "AnalysisFact",
            "tweet_url": "https://x.com/JohnDCook/status/102",
            "tweet_datetime": "2026-01-01T00:00:00.000Z",
            "text": "the theorem is false .",
        },
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    sequences = load_cook_sequences(corpus, vocabulary, max_oov_fraction=0.25)

    assert len(sequences) == 1
    tokens = tuple(vocabulary.token_for(token_id) for token_id in sequences[0])
    assert tokens == ("<BOS>", "the", "theorem", "is", "true", ".", "<EOS>")


def test_teacher_forcing_loss_backpropagates_through_fly_text_interface():
    vocabulary = build_v1_vocabulary()
    policy = _toy_policy(len(vocabulary))
    sequence = (
        vocabulary.id_for("<BOS>"),
        vocabulary.id_for("theorem"),
        vocabulary.id_for("is"),
        vocabulary.id_for("true"),
        vocabulary.id_for("."),
        vocabulary.id_for("<EOS>"),
    )

    loss = teacher_forcing_loss(policy, [sequence])
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert policy.token_embedding.weight.grad is not None
    assert policy.input_gain.grad is not None
    assert policy.readout.weight.grad is not None
    assert policy.recurrent.requires_grad is False


def test_teacher_forcing_loss_rejects_sequences_without_prediction_target():
    vocabulary = build_v1_vocabulary()
    policy = _toy_policy(len(vocabulary))

    try:
        teacher_forcing_loss(policy, [(vocabulary.id_for("<BOS>"),)])
    except ValueError as error:
        assert "at least two tokens" in str(error)
    else:
        raise AssertionError("single-token sequence must be rejected")


def test_scheduled_sampling_transition_has_exact_endpoints():
    assert choose_next_input(
        teacher_token=11,
        sampled_token=22,
        teacher_probability=1.0,
        draw=0.999,
    ) == 11
    assert choose_next_input(
        teacher_token=11,
        sampled_token=22,
        teacher_probability=0.0,
        draw=0.0,
    ) == 22
    assert choose_next_input(
        teacher_token=11,
        sampled_token=22,
        teacher_probability=0.6,
        draw=0.59,
    ) == 11
    assert choose_next_input(
        teacher_token=11,
        sampled_token=22,
        teacher_probability=0.6,
        draw=0.60,
    ) == 22


def test_text_training_config_round_trips_and_validates(tmp_path):
    path = tmp_path / "training.json"
    path.write_text(
        json.dumps(
            {
                "seed": 260917,
                "curriculum_examples": 12000,
                "stage_a_epochs": 2,
                "stage_b_epochs": 3,
                "stage_c_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.0003,
                "weight_decay": 0.00001,
                "gradient_clip_norm": 1.0,
                "max_oov_fraction": 0.35,
                "stage_c_teacher_probability": 0.7,
            }
        ),
        encoding="utf-8",
    )

    config = TextTrainingConfig.from_json(path)

    assert config.seed == 260917
    assert config.curriculum_examples == 12000
    assert config.stage_b_epochs == 3
    assert config.stage_c_teacher_probability == 0.7

    path.write_text(
        json.dumps(
            {
                "seed": 1,
                "curriculum_examples": 1,
                "stage_a_epochs": 1,
                "stage_b_epochs": 1,
                "stage_c_epochs": 1,
                "batch_size": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "max_oov_fraction": 1.2,
                "stage_c_teacher_probability": 0.5,
            }
        ),
        encoding="utf-8",
    )
    try:
        TextTrainingConfig.from_json(path)
    except ValueError as error:
        assert "max_oov_fraction" in str(error)
    else:
        raise AssertionError("invalid OOV fraction must be rejected")
