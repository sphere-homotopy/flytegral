from datetime import UTC, datetime, timedelta

from fly_window.publishing.runtime import (
    RuntimeConfig,
    RuntimeState,
    generation_window,
    health_alerts,
)


def _now() -> datetime:
    return datetime(2026, 9, 17, 15, 0, tzinfo=UTC)


def test_bootstrap_generation_window_builds_five_day_reserve():
    now = _now()
    state = RuntimeState(current_checkpoint="pretrain-a")
    config = RuntimeConfig()

    start, end = generation_window(state, now=now, bootstrap=True, config=config)

    assert start == now
    assert end == now + timedelta(hours=120)


def test_daily_generation_extends_existing_queue_by_one_day():
    now = _now()
    horizon = now + timedelta(hours=83)
    state = RuntimeState(queue_horizon_at=horizon, current_checkpoint="daily-a")

    start, end = generation_window(state, now=now, bootstrap=False, config=RuntimeConfig())

    assert start == horizon
    assert end == horizon + timedelta(hours=24)


def test_health_alerts_cover_stale_training_and_low_reserve_independently():
    now = _now()
    state = RuntimeState(
        last_training_at=now - timedelta(days=8),
        last_generation_at=now - timedelta(hours=3),
        queue_horizon_at=now + timedelta(hours=10),
        current_checkpoint="daily-old",
    )

    alerts = health_alerts(state, now=now, config=RuntimeConfig())

    assert {alert.code for alert in alerts} == {"training_stale", "queue_low"}
    assert next(alert for alert in alerts if alert.code == "training_stale").urgent is False
    assert next(alert for alert in alerts if alert.code == "queue_low").urgent is True


def test_missing_recent_stats_does_not_create_a_blocking_health_state():
    now = _now()
    state = RuntimeState(
        last_stats_at=None,
        last_training_at=now - timedelta(days=1),
        last_generation_at=now,
        queue_horizon_at=now + timedelta(days=3),
        current_checkpoint="daily-current",
    )

    alerts = health_alerts(state, now=now, config=RuntimeConfig())

    assert alerts == ()
