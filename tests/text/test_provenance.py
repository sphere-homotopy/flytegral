import json

from fly_window.text.generation import GeneratedTweet
from fly_window.text.provenance import generated_tweet_record, write_generation_jsonl


def _tweet() -> GeneratedTweet:
    return GeneratedTweet(
        text="bzz theorem",
        token_ids=(17, 211, 1),
        token_logprobs=(-0.1, -0.2, -0.3),
        seed=260917,
        termination_reason="eos",
        checkpoint_id="text-pretrain-1",
        git_sha="abcdef123456",
        batch_id="2026-09-17-a",
        tweet_index=4,
    )


def test_generated_tweet_record_is_json_and_sheet_safe():
    record = generated_tweet_record(_tweet())

    assert record == {
        "batch_id": "2026-09-17-a",
        "tweet_index": 4,
        "text": "bzz theorem",
        "checkpoint_id": "text-pretrain-1",
        "git_sha": "abcdef123456",
        "rng_seed": 260917,
        "token_ids": "[17,211,1]",
        "token_logprobs": "[-0.1,-0.2,-0.3]",
        "termination_reason": "eos",
    }
    json.dumps(record)


def test_generation_jsonl_round_trips_without_losing_trajectory(tmp_path):
    path = tmp_path / "generation.jsonl"

    write_generation_jsonl(path, [_tweet(), _tweet()])

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert json.loads(rows[0]["token_ids"]) == [17, 211, 1]
    assert json.loads(rows[0]["token_logprobs"]) == [-0.1, -0.2, -0.3]
    assert rows[0]["termination_reason"] == "eos"
