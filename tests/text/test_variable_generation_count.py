import numpy as np

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.text.generation import GenerationConfig, generate_batch
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import build_v1_vocabulary


def _policy(vocab_size: int) -> FlyTextPolicy:
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
        microsteps=1,
        leak=1.0,
    )


def test_generate_batch_accepts_policy_selected_count():
    vocabulary = build_v1_vocabulary()
    policy = _policy(len(vocabulary))

    batch = generate_batch(
        policy,
        vocabulary,
        base_seed=700,
        config=GenerationConfig(max_tokens=1),
        checkpoint_id="ckpt-variable",
        git_sha="abc123",
        batch_id="batch-variable",
        count=3,
    )

    assert len(batch) == 3
    assert [tweet.tweet_index for tweet in batch] == [0, 1, 2]
    assert [tweet.seed for tweet in batch] == [700, 701, 702]
