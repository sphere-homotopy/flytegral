import json
from datetime import UTC, datetime

from fly_window.publishing.rows import FLY_TWEET_COLUMNS, generated_tweet_row
from fly_window.text.generation import GeneratedTweet


def _tweet() -> GeneratedTweet:
    return GeneratedTweet(
        text="bzz theorem",
        token_ids=(17, 211, 1),
        token_logprobs=(-0.1, -0.2, -0.3),
        seed=260917,
        termination_reason="eos",
        checkpoint_id="ckpt-1",
        git_sha="abcdef",
        batch_id="batch-a",
        tweet_index=3,
    )


def test_generated_row_matches_append_only_sheet_contract():
    generated_at = datetime(2026, 9, 17, 1, 15, tzinfo=UTC)

    row = generated_tweet_row(_tweet(), generated_at=generated_at)

    assert tuple(row) == FLY_TWEET_COLUMNS
    assert row["batch_id"] == "batch-a"
    assert row["tweet_index"] == 3
    assert row["idempotency_key"] == "batch-a:3"
    assert row["generation_time"] == "2026-09-17T01:15:00+00:00"
    assert json.loads(row["token_ids"]) == [17, 211, 1]
    assert json.loads(row["token_logprobs"]) == [-0.1, -0.2, -0.3]
    assert row["status"] == "generated"
    assert row["scheduled_at"] == ""
    assert row["tweet_url"] == ""
    assert row["reward"] == ""


def test_idempotency_key_is_stable_from_batch_and_index():
    first = generated_tweet_row(
        _tweet(), generated_at=datetime(2026, 9, 17, tzinfo=UTC)
    )
    second = generated_tweet_row(
        _tweet(), generated_at=datetime(2026, 9, 18, tzinfo=UTC)
    )

    assert first["idempotency_key"] == second["idempotency_key"] == "batch-a:3"
