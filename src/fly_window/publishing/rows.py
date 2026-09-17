from __future__ import annotations

import json
from datetime import datetime
from collections.abc import Sequence

from fly_window.text.generation import GeneratedTweet
from fly_window.text.provenance import generated_tweet_record


FLY_TWEET_COLUMNS = (
    "batch_id",
    "batch_size",
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
    "cadence_context",
    "cadence_action",
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
    batch_size: int | None = None,
    cadence_context: Sequence[float] | None = None,
    cadence_action: int | None = None,
) -> dict[str, object]:
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    if scheduled_at is not None and (
        scheduled_at.tzinfo is None or scheduled_at.utcoffset() is None
    ):
        raise ValueError("scheduled_at must be timezone-aware")
    if (cadence_context is None) != (cadence_action is None):
        raise ValueError("cadence_context and cadence_action must be supplied together")
    cadence_context_json = ""
    cadence_action_value: int | str = ""
    if cadence_context is not None and cadence_action is not None:
        values_context = [float(value) for value in cadence_context]
        if len(values_context) != 4:
            raise ValueError("cadence_context must contain exactly four values")
        if int(cadence_action) <= 0:
            raise ValueError("cadence_action must be a positive wait action")
        cadence_context_json = json.dumps(values_context, separators=(",", ":"))
        cadence_action_value = int(cadence_action)

    provenance = generated_tweet_record(tweet)
    values: dict[str, object] = {
        "batch_id": tweet.batch_id,
        "batch_size": "" if batch_size is None else batch_size,
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
        "cadence_context": cadence_context_json,
        "cadence_action": cadence_action_value,
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
