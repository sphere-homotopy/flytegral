from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch.nn import functional as F

from fly_window.text.corpus import CorpusTweet, prepare_corpus_tweet
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import FlyVocabulary


def _canonical_status_url(tweet_url: str) -> str | None:
    try:
        parsed = urlparse(tweet_url)
    except ValueError:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[1].lower() != "status" or not parts[2].isdigit():
        return None
    return f"https://x.com/{parts[0]}/status/{parts[2]}"


def load_cook_sequences(
    path: Path,
    vocabulary: FlyVocabulary,
    *,
    max_oov_fraction: float = 0.35,
) -> list[tuple[int, ...]]:
    """Load a scraped Cook JSONL corpus as deduplicated teacher-forcing sequences."""
    sequences: list[tuple[int, ...]] = []
    seen_statuses: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error.msg}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"invalid JSONL at line {line_number}: expected object")

            try:
                row = CorpusTweet(
                    account=str(payload["account"]),
                    tweet_url=str(payload["tweet_url"]),
                    tweet_datetime=str(payload.get("tweet_datetime", "")),
                    text=str(payload["text"]),
                    is_reply=bool(payload.get("is_reply", False)),
                    is_repost=bool(payload.get("is_repost", False)),
                )
            except KeyError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: missing {error.args[0]}"
                ) from error

            canonical_url = _canonical_status_url(row.tweet_url)
            if canonical_url is None or canonical_url in seen_statuses:
                continue

            prepared = prepare_corpus_tweet(
                row,
                vocabulary,
                max_oov_fraction=max_oov_fraction,
            )
            if prepared is None:
                continue

            seen_statuses.add(canonical_url)
            sequences.append(prepared.token_ids)

    return sequences


def teacher_forcing_loss(
    policy: FlyTextPolicy,
    sequences: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Mean next-token cross-entropy while preserving the fly state through each sequence."""
    if not sequences:
        raise ValueError("sequences must not be empty")

    losses: list[torch.Tensor] = []
    device = policy.input_gain.device

    for sequence in sequences:
        if len(sequence) < 2:
            raise ValueError("each training sequence must contain at least two tokens")

        state = policy.initial_state(batch_size=1)
        for input_token, target_token in zip(sequence[:-1], sequence[1:], strict=True):
            token_ids = torch.tensor([int(input_token)], dtype=torch.long, device=device)
            target = torch.tensor([int(target_token)], dtype=torch.long, device=device)
            output = policy.step(token_ids, state)
            losses.append(F.cross_entropy(output.logits, target))
            state = output.state

    if not losses:
        raise ValueError("sequences did not contain any prediction targets")
    return torch.stack(losses).mean()
