from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    initial_reserve_hours: float = 120.0
    daily_refill_hours: float = 24.0
    stale_training_hours: float = 168.0
    low_queue_hours: float = 24.0
    max_posts_per_24h: int = 30
    min_gap_minutes: float = 20.0

    def __post_init__(self) -> None:
        positive = {
            "initial_reserve_hours": self.initial_reserve_hours,
            "daily_refill_hours": self.daily_refill_hours,
            "stale_training_hours": self.stale_training_hours,
            "low_queue_hours": self.low_queue_hours,
            "min_gap_minutes": self.min_gap_minutes,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_posts_per_24h <= 0:
            raise ValueError("max_posts_per_24h must be positive")

    @classmethod
    def from_json(cls, path: Path) -> "RuntimeConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("runtime config must be a JSON object")
        return cls(
            initial_reserve_hours=float(payload.get("initial_reserve_hours", 120.0)),
            daily_refill_hours=float(payload.get("daily_refill_hours", 24.0)),
            stale_training_hours=float(payload.get("stale_training_hours", 168.0)),
            low_queue_hours=float(payload.get("low_queue_hours", 24.0)),
            max_posts_per_24h=int(payload.get("max_posts_per_24h", 30)),
            min_gap_minutes=float(payload.get("min_gap_minutes", 20.0)),
        )


@dataclass(frozen=True, slots=True)
class RuntimeState:
    started_at: datetime | None = None
    last_stats_at: datetime | None = None
    last_training_at: datetime | None = None
    last_generation_at: datetime | None = None
    queue_horizon_at: datetime | None = None
    current_checkpoint: str = ""
    last_alert_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "started_at",
            "last_stats_at",
            "last_training_at",
            "last_generation_at",
            "queue_horizon_at",
            "last_alert_at",
        ):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class HealthAlert:
    code: str
    message: str
    urgent: bool


def _require_aware(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")


def generation_window(
    state: RuntimeState,
    *,
    now: datetime,
    bootstrap: bool,
    config: RuntimeConfig,
) -> tuple[datetime, datetime]:
    """Return the horizon slice that the fly should fill next.

    Existing scheduled posts are immutable. Bootstrap fills roughly five days from
    the present. Normal daily operation appends roughly one day beyond the durable
    queue horizon. If the queue has already drained, refill starts from `now`.
    """
    _require_aware(now)
    if bootstrap:
        start = now
        hours = config.initial_reserve_hours
    else:
        horizon = state.queue_horizon_at
        start = horizon if horizon is not None and horizon > now else now
        hours = config.daily_refill_hours
    return start, start + timedelta(hours=hours)


def health_alerts(
    state: RuntimeState,
    *,
    now: datetime,
    config: RuntimeConfig,
) -> tuple[HealthAlert, ...]:
    """Describe operator-visible failures without blocking publication/generation."""
    _require_aware(now)
    alerts: list[HealthAlert] = []

    training_baseline = state.last_training_at or state.started_at
    if training_baseline is not None:
        training_age = now - training_baseline
        if training_age >= timedelta(hours=config.stale_training_hours):
            alerts.append(
                HealthAlert(
                    code="training_stale",
                    message=(
                        "No successful Fly Tweets retraining for "
                        f"{training_age.total_seconds() / 3600:.1f} hours"
                    ),
                    urgent=False,
                )
            )

    horizon = state.queue_horizon_at
    queue_remaining = timedelta(0) if horizon is None else horizon - now
    if queue_remaining < timedelta(hours=config.low_queue_hours):
        alerts.append(
            HealthAlert(
                code="queue_low",
                message=(
                    "Fly Tweets publishing reserve below threshold: "
                    f"{max(queue_remaining.total_seconds(), 0.0) / 3600:.1f} hours remain"
                ),
                urgent=True,
            )
        )

    return tuple(alerts)
