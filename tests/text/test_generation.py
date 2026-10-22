import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.text.generation import GenerationConfig, generate_batch, generate_tweet
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import build_v1_vocabulary


class _CountingInitialStatePolicy(FlyTextPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state_calls = 0

    def initial_state(self, batch_size: int) -> torch.Tensor:
        self.initial_state_calls += 1
        return super().initial_state(batch_size)


def _toy_graph() -> ConnectomeGraph:
    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 1], dtype=np.int64),
        dst_idx=np.asarray([1, 2], dtype=np.int64),
        weights=np.asarray([3, 3], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )


def _policy(vocab_size: int) -> _CountingInitialStatePolicy:
    graph = _toy_graph()
    return _CountingInitialStatePolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=vocab_size,
        sensory_dim=8,
        microsteps=1,
        leak=1.0,
    )


def _config(**overrides) -> GenerationConfig:
    values = {
        "temperature": 1.0,
        "max_tokens": 12,
        "max_chars": 240,
    }
    values.update(overrides)
    return GenerationConfig(**values)


def test_generation_replays_exactly_from_seed_and_resets_state():
    vocabulary = build_v1_vocabulary()
    policy = _policy(len(vocabulary))
    with torch.no_grad():
        policy.token_embedding.weight.zero_()
        policy.input_gain.zero_()
        policy.input_bias.zero_()
        policy.readout.weight.zero_()
        policy.readout.bias.zero_()
        policy.readout.bias[vocabulary.id_for("theorem")] = 1.0
        policy.readout.bias[vocabulary.id_for("proof")] = 1.0
        policy.readout.bias[vocabulary.id_for("<EOS>")] = 0.5

    first = generate_tweet(
        policy,
        vocabulary,
        seed=12345,
        config=_config(),
        checkpoint_id="ckpt-1",
        git_sha="abc123",
        batch_id="batch-a",
        tweet_index=0,
    )
    second = generate_tweet(
        policy,
        vocabulary,
        seed=12345,
        config=_config(),
        checkpoint_id="ckpt-1",
        git_sha="abc123",
        batch_id="batch-a",
        tweet_index=0,
    )

    assert first.token_ids == second.token_ids
    assert first.token_logprobs == second.token_logprobs
    assert first.text == second.text
    assert first.termination_reason == second.termination_reason
    assert policy.initial_state_calls == 2


def test_generation_stops_on_eos_without_rendering_eos():
    vocabulary = build_v1_vocabulary()
    policy = _policy(len(vocabulary))
    with torch.no_grad():
        policy.readout.weight.zero_()
        policy.readout.bias.fill_(-100.0)
        policy.readout.bias[vocabulary.id_for("<EOS>")] = 100.0

    generated = generate_tweet(
        policy,
        vocabulary,
        seed=7,
        config=_config(),
        checkpoint_id="ckpt-eos",
        git_sha="deadbeef",
        batch_id="batch-eos",
        tweet_index=0,
    )

    assert generated.termination_reason == "eos"
    assert generated.token_ids == (vocabulary.id_for("<EOS>"),)
    assert generated.text == ""
    assert len(generated.token_logprobs) == 1


def test_generation_enforces_rendered_character_cap():
    vocabulary = build_v1_vocabulary()
    policy = _policy(len(vocabulary))
    with torch.no_grad():
        policy.readout.weight.zero_()
        policy.readout.bias.fill_(-100.0)
        policy.readout.bias[vocabulary.id_for("theorem")] = 100.0

    generated = generate_tweet(
        policy,
        vocabulary,
        seed=9,
        config=_config(max_chars=10),
        checkpoint_id="ckpt-cap",
        git_sha="cafebabe",
        batch_id="batch-cap",
        tweet_index=0,
    )

    assert len(generated.text) <= 10
    assert generated.termination_reason == "char_cap"


def test_generation_hard_token_cap_is_reported():
    vocabulary = build_v1_vocabulary()
    policy = _policy(len(vocabulary))
    with torch.no_grad():
        policy.readout.weight.zero_()
        policy.readout.bias.fill_(-100.0)
        policy.readout.bias[vocabulary.id_for("theorem")] = 100.0

    generated = generate_tweet(
        policy,
        vocabulary,
        seed=11,
        config=_config(max_tokens=3, max_chars=240),
        checkpoint_id="ckpt-token",
        git_sha="01234567",
        batch_id="batch-token",
        tweet_index=0,
    )

    assert generated.termination_reason == "token_cap"
    assert len(generated.token_ids) == 3


def test_generate_batch_emits_exactly_ten_indexed_tweets():
    vocabulary = build_v1_vocabulary()
    policy = _policy(len(vocabulary))
    with torch.no_grad():
        policy.readout.weight.zero_()
        policy.readout.bias.fill_(-100.0)
        policy.readout.bias[vocabulary.id_for("<EOS>")] = 100.0

    batch = generate_batch(
        policy,
        vocabulary,
        base_seed=1000,
        config=_config(),
        checkpoint_id="ckpt-batch",
        git_sha="feedface",
        batch_id="batch-10",
    )

    assert len(batch) == 10
    assert [tweet.tweet_index for tweet in batch] == list(range(10))
    assert [tweet.seed for tweet in batch] == list(range(1000, 1010))
    assert all(tweet.batch_id == "batch-10" for tweet in batch)
