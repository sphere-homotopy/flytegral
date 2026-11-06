from datetime import UTC, datetime

import pytest

from fly_window.publishing.rows import generated_tweet_row
from fly_window.text.generation import GeneratedTweet


def _tweet() -> GeneratedTweet:
    return GeneratedTweet(
        text="bzz theorem",
        token_ids=(1, 2),
        token_logprobs=(-0.1, -0.2),
        seed=17,
        termination_reason="eos",
        checkpoint_id="ckpt-a",
        git_sha="abc123",
        batch_id="batch-a",
        tweet_index=0,
    )


def test_generated_row_preserves_fly_selected_publish_time():
    publish_at = datetime(2026, 9, 18, 11, 37, tzinfo=UTC)
    row = generated_tweet_row(
        _tweet(),
        generated_at=datetime(2026, 9, 17, 15, 0, tzinfo=UTC),
        scheduled_at=publish_at,
    )

    assert row["scheduled_at"] == "2026-09-18T11:37:00+00:00"


def test_scheduled_publish_time_must_be_timezone_aware():
    with pytest.raises(ValueError, match="scheduled_at must be timezone-aware"):
        generated_tweet_row(
            _tweet(),
            generated_at=datetime(2026, 9, 17, 15, 0, tzinfo=UTC),
            scheduled_at=datetime(2026, 9, 18, 11, 37),
        )
