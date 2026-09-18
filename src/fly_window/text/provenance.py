from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from fly_window.text.generation import GeneratedTweet


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generated_tweet_record(tweet: GeneratedTweet) -> dict[str, object]:
    return {
        "batch_id": tweet.batch_id,
        "tweet_index": tweet.tweet_index,
        "text": tweet.text,
        "checkpoint_id": tweet.checkpoint_id,
        "git_sha": tweet.git_sha,
        "rng_seed": tweet.seed,
        "token_ids": _compact_json(list(tweet.token_ids)),
        "token_logprobs": _compact_json(list(tweet.token_logprobs)),
        "termination_reason": tweet.termination_reason,
    }


def write_generation_jsonl(path: Path, tweets: Sequence[GeneratedTweet]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(generated_tweet_record(tweet), ensure_ascii=False, sort_keys=True)
        for tweet in tweets
    )
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")
