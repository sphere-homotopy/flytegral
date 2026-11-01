import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_candidate_math_corpus import filter_external_rows


def test_only_original_math_content_from_allowed_accounts():
    rows = [
        {"account": "mouse_math", "tweet_url": "https://x.com/mouse_math/status/123",
         "text": "Topology is a mystery", "is_repost": False, "is_reply": False, "is_quote": False},
        {"account": "alcuintweets", "tweet_url": "https://x.com/other/status/124",
         "text": "Proof: this is stolen"},
        {"account": "mouse_math", "tweet_url": "https://x.com/mouse_math/status/125",
         "text": "Good afternoon"},
        {"account": "mouse_math", "tweet_url": "https://x.com/mouse_math/status/126",
         "text": "Mathematics is cool", "is_reply": True},
    ]
    result = filter_external_rows(rows, handles={"mouse_math", "alcuintweets"})
    assert len(result) == 1
    assert result[0]["url"].endswith("123")
    assert result[0]["training_approved"] is False
    assert result[0]["handle"] == "mouse_math"
