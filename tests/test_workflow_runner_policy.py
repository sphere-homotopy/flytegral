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
    assert "gh workflow run $workflow" in probe

    assert "gh run list" in migration
    assert "fly-tweets-initial-pretrain.yml" in migration
    assert "gh workflow run $workflow" in migration

    assert "gh run list" in pretrain
    assert "fly-tweets-daily.yml" in pretrain
    assert "gh workflow run $workflow" in pretrain


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
