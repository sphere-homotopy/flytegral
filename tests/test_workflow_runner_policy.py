from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
HEAVY_FLY_TWEETS_WORKFLOWS = (
    "fly-tweets-cook-corpus.yml",
    "fly-tweets-initial-pretrain.yml",
    "fly-tweets-migrate-legacy-runtime.yml",
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
