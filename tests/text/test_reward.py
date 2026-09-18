from datetime import UTC, datetime, timedelta
from pathlib import Path

from fly_window.text.reward import RewardConfig, compute_daily_rewards


NOW = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _row(
    key: str,
    *,
    age_hours: float = 30,
    views: int = 1000,
    likes: int = 10,
    reposts: int = 2,
    replies: int = 1,
    bookmarks: int = 1,
):
    return {
        "idempotency_key": key,
        "published_at": (NOW - timedelta(hours=age_hours)).isoformat(),
        "views": views,
        "likes": likes,
        "reposts": reposts,
        "replies": replies,
        "bookmarks": bookmarks,
    }


def _config() -> RewardConfig:
    return RewardConfig(
        min_age_hours=24.0,
        view_weight=1.0,
        like_rate_weight=2.0,
        repost_rate_weight=3.0,
        reply_rate_weight=2.5,
        bookmark_rate_weight=2.0,
        rate_scale=100.0,
        smoothing_views=100.0,
        smoothing_events=0.5,
    )


def test_versioned_reward_config_loads_and_prioritizes_stronger_engagement():
    config = RewardConfig.from_json(REPO_ROOT / "configs" / "daily_reward_v1.json")

    assert config.min_age_hours == 24.0
    assert config.repost_rate_weight > config.like_rate_weight
    assert config.reply_rate_weight > config.like_rate_weight
    assert config.bookmark_rate_weight > config.like_rate_weight


def test_reward_excludes_tweets_younger_than_minimum_age():
    rewards = compute_daily_rewards(
        [_row("old", age_hours=25), _row("young", age_hours=23.9)],
        config=_config(),
        now=NOW,
    )

    assert [item.idempotency_key for item in rewards] == ["old"]


def test_more_exposure_increases_raw_score_when_engagement_counts_match():
    rewards = compute_daily_rewards(
        [
            _row("low", views=100, likes=0, reposts=0, replies=0, bookmarks=0),
            _row("high", views=10_000, likes=0, reposts=0, replies=0, bookmarks=0),
        ],
        config=_config(),
        now=NOW,
    )
    by_key = {item.idempotency_key: item for item in rewards}

    assert by_key["high"].raw_score > by_key["low"].raw_score


def test_more_engagement_increases_raw_score_at_same_views():
    rewards = compute_daily_rewards(
        [
            _row("quiet", views=1000, likes=1, reposts=0, replies=0, bookmarks=0),
            _row("engaged", views=1000, likes=30, reposts=5, replies=4, bookmarks=3),
        ],
        config=_config(),
        now=NOW,
    )
    by_key = {item.idempotency_key: item for item in rewards}

    assert by_key["engaged"].raw_score > by_key["quiet"].raw_score


def test_rank_rewards_are_centered_bounded_and_order_preserving():
    rewards = compute_daily_rewards(
        [
            _row("a", views=100),
            _row("b", views=1000),
            _row("c", views=10_000),
        ],
        config=_config(),
        now=NOW,
    )
    by_key = {item.idempotency_key: item for item in rewards}

    assert abs(sum(item.reward for item in rewards)) < 1e-12
    assert all(-1.0 <= item.reward <= 1.0 for item in rewards)
    assert by_key["a"].reward < by_key["b"].reward < by_key["c"].reward


def test_tied_raw_scores_receive_identical_rank_reward():
    rewards = compute_daily_rewards(
        [_row("a"), _row("b"), _row("c", views=5000)],
        config=_config(),
        now=NOW,
    )
    by_key = {item.idempotency_key: item for item in rewards}

    assert by_key["a"].raw_score == by_key["b"].raw_score
    assert by_key["a"].reward == by_key["b"].reward
