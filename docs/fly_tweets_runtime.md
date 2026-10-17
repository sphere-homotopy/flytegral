# Fly Tweets runtime invariants

## Ownership of cadence

The fly policy owns both content and publishing cadence. Production code must not impose a fixed tweets-per-day target or a fixed set of daily posting slots. A generated post carries an absolute `publish_at` selected by the fly policy. Runtime safety limits may reject pathological schedules, but they must not choose the normal cadence for the fly.

## Rolling reserve

The publishing queue is durable and independent from the training process.

- Bootstrap fills approximately 120 hours (5 days) of future publishing horizon.
- A healthy daily cycle extends the queue by approximately 24 hours beyond the current horizon.
- Already scheduled posts are immutable; a later model checkpoint does not rewrite them.
- If fresh engagement statistics are available, the daily cycle may apply one conservative retraining update before generating the next horizon slice.
- If no fresh settled statistics are available, retraining is skipped and generation still proceeds from the latest valid checkpoint. Missing statistics must never drain the publishing reserve by itself.

## Health state

The durable runtime state records at least:

- `last_stats_at`
- `last_training_at`
- `last_generation_at`
- `queue_horizon_at`
- `current_checkpoint`
- `last_alert_at`

## Alerts

- Send a health alert when no successful retraining has occurred for 168 hours (7 days).
- Send an urgent reserve alert when less than 24 hours of future scheduled publishing remains.
- Alerting is observational only: it must not stop generation or publication.

## Delivery

Buffer/X credentials, account IDs, Google Sheet IDs, and other deployment-specific identifiers are runtime configuration and must not be hard-coded in the repository. The repository currently defines queue contracts only; connecting the real Buffer/X account is a separate deployment step.

## Safety bounds

Safety bounds exist only to contain a broken policy. They may cap posts per rolling 24-hour period and require a minimum inter-post gap. These bounds are not the fly's desired cadence and are not training targets.
