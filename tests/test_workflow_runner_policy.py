from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
HEAVY_FLY_TWEETS_WORKFLOWS = (
    "fly-tweets-cook-corpus.yml",
    "fly-tweets-initial-pretrain.yml",
    "fly-tweets-migrate-legacy-runtime.yml",
    "fly-tweets-daily.yml",
    "pc-runner-probe.yml",
)
EXPECTED_SELF_HOSTED_RUNNER = "runs-on: [self-hosted, windows, x64, flytegral-pc]"


def test_fly_tweets_workflows_never_reference_life_supervisor():
    offenders = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        if "life-supervisor" in path.read_text(encoding="utf-8").lower():
            offenders.append(path.name)

    assert offenders == []


def test_heavy_fly_tweets_workflows_have_no_hosted_runner_fallback():
    for filename in HEAVY_FLY_TWEETS_WORKFLOWS:
        text = (WORKFLOW_DIR / filename).read_text(encoding="utf-8")
        assert EXPECTED_SELF_HOSTED_RUNNER in text, filename
        assert "ubuntu-latest" not in text, filename
        assert "windows-latest" not in text, filename
        assert "macos-latest" not in text, filename


def test_native_bootstrap_chain_dispatches_each_next_stage_idempotently():
    probe = (WORKFLOW_DIR / "pc-runner-probe.yml").read_text(encoding="utf-8")
    migration = (WORKFLOW_DIR / "fly-tweets-migrate-legacy-runtime.yml").read_text(
        encoding="utf-8"
    )
    pretrain = (WORKFLOW_DIR / "fly-tweets-initial-pretrain.yml").read_text(
        encoding="utf-8"
    )

    assert "gh run list" in probe
    assert "fly-tweets-migrate-legacy-runtime.yml" in probe
    assert "fly-tweets-initial-pretrain.yml" in probe
    assert "PRETRAIN_RECOVERY_DISPATCHED=true" in probe
    assert "gh workflow run $pretrainWorkflow" in probe

    assert "gh run list" in migration
    assert "fly-tweets-initial-pretrain.yml" in migration
    assert "gh workflow run $workflow" in migration

    assert "gh run list" in pretrain
    assert "fly-tweets-daily.yml" in pretrain
    assert "gh workflow run $workflow" in pretrain


def test_initial_pretrain_is_bounded_and_daily_bootstrap_is_credential_gated():
    text = (WORKFLOW_DIR / "fly-tweets-initial-pretrain.yml").read_text(encoding="utf-8")

    assert "--gate-config configs/text_gate_smoke.json" in text
    assert "--config configs/text_training_cpu_v1.json" in text
    assert "--gate-config configs/text_gate_v1.json" in text
    assert "BUFFER_API_KEY: ${{ secrets.BUFFER_API_KEY }}" in text
    assert "BUFFER_CHANNEL_ID: ${{ secrets.BUFFER_CHANNEL_ID }}" in text
    assert "BUFFER_ORGANIZATION_ID: ${{ secrets.BUFFER_ORGANIZATION_ID }}" in text
    assert "DAILY_BOOTSTRAP_BLOCKED_BUFFER_CREDENTIALS=true" in text


def test_daily_worker_is_native_headless_and_cadence_owned():
    text = (WORKFLOW_DIR / "fly-tweets-daily.yml").read_text(encoding="utf-8")

    assert "schedule:" in text
    assert EXPECTED_SELF_HOSTED_RUNNER in text
    assert "-NoProfile -NonInteractive" in text
    assert "run_daily_text_training.py" in text
    assert "--cadence-checkpoint" in text
    assert "--runtime-state" in text
    assert "--runtime-config" in text
    assert "--generation-count" not in text
    assert "collect-fly-tweet-stats.js" in text
    assert "continue-on-error: true" in text


def test_daily_worker_reconciles_and_schedules_buffer_delivery():
    text = (WORKFLOW_DIR / "fly-tweets-daily.yml").read_text(encoding="utf-8")

    assert text.count("publish_fly_tweets_buffer.py") >= 2
    assert "BUFFER_API_KEY: ${{ secrets.BUFFER_API_KEY }}" in text
    assert "BUFFER_CHANNEL_ID: ${{ secrets.BUFFER_CHANNEL_ID }}" in text
    assert "BUFFER_ORGANIZATION_ID: ${{ secrets.BUFFER_ORGANIZATION_ID }}" in text
    assert "Buffer credentials are not configured" in text


def test_daily_worker_exports_non_sensitive_health_heartbeat():
    text = (WORKFLOW_DIR / "fly-tweets-daily.yml").read_text(encoding="utf-8")

    assert "fly-tweets-health-heartbeat" in text
    assert "last_training_at" in text
    assert "queue_horizon_at" in text
    assert "last_stats_at" in text
    assert "health-heartbeat.json" in text
    assert "actions/upload-artifact@v4" in text


def test_hosted_watchdog_survives_native_runner_failure_and_emails_after_week_without_training():
    text = (WORKFLOW_DIR / "fly-tweets-watchdog.yml").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-latest" in text
    assert "fly-tweets-daily.yml" in text
    assert "fly-tweets-health-heartbeat" in text
    assert "7 * 24 * 60 * 60" in text
    assert "48 * 60 * 60" in text
    assert "FLYTWEETS_SMTP_SERVER" in text
    assert "FLYTWEETS_SMTP_PASSWORD" in text
    assert "WATCHDOG_DISABLED_CREDENTIALS=true" in text
