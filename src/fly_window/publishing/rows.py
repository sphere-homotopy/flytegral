from __future__ import annotations

from datetime import datetime

from fly_window.text.generation import GeneratedTweet
from fly_window.text.provenance import generated_tweet_record


FLY_TWEET_COLUMNS = (
    "batch_id",
    "tweet_index",
    "idempotency_key",
    "text",
    "generation_time",
    "checkpoint_id",
    "git_sha",
    "rng_seed",
    "token_ids",
    "token_logprobs",
    "termination_reason",
    "status",
    "scheduled_at",
    "buffer_post_id",
    "tweet_url",
    "published_at",
    "views",
    "likes",
    "reposts",
    "replies",
    "bookmarks",
    "metrics_collected_at",
    "reward",
    "training_consumed_at",
    "last_error",
)


def generated_tweet_row(
    tweet: GeneratedTweet,
    *,
    generated_at: datetime,
    scheduled_at: datetime | None = None,
) -> dict[str, object]:
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    if scheduled_at is not None and (
        scheduled_at.tzinfo is None or scheduled_at.utcoffset() is None
    ):
        raise ValueError("scheduled_at must be timezone-aware")

    provenance = generated_tweet_record(tweet)
    values: dict[str, object] = {
        "batch_id": tweet.batch_id,
        "tweet_index": tweet.tweet_index,
        "idempotency_key": f"{tweet.batch_id}:{tweet.tweet_index}",
        "text": tweet.text,
        "generation_time": generated_at.isoformat(),
        "checkpoint_id": tweet.checkpoint_id,
        "git_sha": tweet.git_sha,
        "rng_seed": tweet.seed,
        "token_ids": provenance["token_ids"],
        "token_logprobs": provenance["token_logprobs"],
        "termination_reason": tweet.termination_reason,
        "status": "generated",
        "scheduled_at": "" if scheduled_at is None else scheduled_at.isoformat(),
        "buffer_post_id": "",
        "tweet_url": "",
        "published_at": "",
        "views": "",
        "likes": "",
        "reposts": "",
        "replies": "",
        "bookmarks": "",
        "metrics_collected_at": "",
        "reward": "",
        "training_consumed_at": "",
        "last_error": "",
    }
    return {column: values[column] for column in FLY_TWEET_COLUMNS}
