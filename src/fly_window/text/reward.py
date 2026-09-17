from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RewardConfig:
    min_age_hours: float
    view_weight: float
    like_rate_weight: float
    repost_rate_weight: float
    reply_rate_weight: float
    bookmark_rate_weight: float
    rate_scale: float
    smoothing_views: float
    smoothing_events: float

    def __post_init__(self) -> None:
        if self.min_age_hours < 0.0:
            raise ValueError("min_age_hours must be non-negative")
        if self.rate_scale <= 0.0:
            raise ValueError("rate_scale must be positive")
        if self.smoothing_views <= 0.0:
            raise ValueError("smoothing_views must be positive")
        if self.smoothing_events < 0.0:
            raise ValueError("smoothing_events must be non-negative")
        weights = (
            self.view_weight,
            self.like_rate_weight,
            self.repost_rate_weight,
            self.reply_rate_weight,
            self.bookmark_rate_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("reward weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RewardedTweet:
    idempotency_key: str
    raw_score: float
    reward: float
    age_hours: float
    views: int
    likes: int
    reposts: int
    replies: int
    bookmarks: int


def _parse_datetime(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("published_at is required")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("published_at must be timezone-aware")
    return parsed.astimezone(UTC)


def _metric(row: Mapping[str, object], key: str) -> int:
    value = row.get(key, 0)
    if value in (None, ""):
        value = 0
    number = int(value)
    if number < 0:
        raise ValueError(f"{key} must be non-negative")
    return number


def _raw_score(row: Mapping[str, object], config: RewardConfig) -> tuple[float, tuple[int, int, int, int, int]]:
    views = _metric(row, "views")
    likes = _metric(row, "likes")
    reposts = _metric(row, "reposts")
    replies = _metric(row, "replies")
    bookmarks = _metric(row, "bookmarks")

    denominator = views + config.smoothing_views
    event_smoothing = config.smoothing_events
    engagement = (
        config.like_rate_weight * ((likes + event_smoothing) / denominator)
        + config.repost_rate_weight * ((reposts + event_smoothing) / denominator)
        + config.reply_rate_weight * ((replies + event_smoothing) / denominator)
        + config.bookmark_rate_weight * ((bookmarks + event_smoothing) / denominator)
    )
    # Engagement rates can be very large at low exposure. Compress their contribution
    # so the configured log-exposure term remains monotone when event counts are held
    # fixed, while still preserving ordering by engagement at equal exposure.
    engagement_score = math.log1p(config.rate_scale * engagement)
    score = config.view_weight * math.log1p(views) + engagement_score
    return score, (views, likes, reposts, replies, bookmarks)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        value = values[order[cursor]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def compute_daily_rewards(
    rows: list[Mapping[str, object]],
    *,
    config: RewardConfig,
    now: datetime,
) -> list[RewardedTweet]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(UTC)

    eligible: list[tuple[str, float, float, tuple[int, int, int, int, int]]] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = str(row.get("idempotency_key", "")).strip()
        if not key:
            raise ValueError("idempotency_key is required")
        if key in seen_keys:
            raise ValueError(f"duplicate idempotency_key: {key}")
        seen_keys.add(key)

        published_at = _parse_datetime(row.get("published_at"))
        age_hours = (now_utc - published_at).total_seconds() / 3600.0
        if age_hours < config.min_age_hours:
            continue
        score, metrics = _raw_score(row, config)
        eligible.append((key, score, age_hours, metrics))

    if not eligible:
        return []

    raw_scores = [item[1] for item in eligible]
    ranks = _average_ranks(raw_scores)
    midpoint = (len(eligible) - 1) / 2.0
    scale = midpoint if midpoint > 0.0 else 1.0

    result: list[RewardedTweet] = []
    for (key, raw_score, age_hours, metrics), rank in zip(eligible, ranks, strict=True):
        views, likes, reposts, replies, bookmarks = metrics
        result.append(
            RewardedTweet(
                idempotency_key=key,
                raw_score=raw_score,
                reward=(rank - midpoint) / scale if len(eligible) > 1 else 0.0,
                age_hours=age_hours,
                views=views,
                likes=likes,
                reposts=reposts,
                replies=replies,
                bookmarks=bookmarks,
            )
        )
    return result
