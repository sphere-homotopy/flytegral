import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_fly_v2_review_queue import choose_review_rows, write_review_csv


def test_review_queue_never_autopromotes_and_prefers_editor_drafts(tmp_path: Path):
    inputs = [
        {"tweet_id": "1", "language": "ru", "text_original": "A", "text_en": "Valid English draft",
         "translation_status": "machine_draft", "url": "https://x.com/m/status/1"},
        {"tweet_id": "2", "language": "ru", "text_original": "B", "text_en": "Proof by banana",
         "translation_status": "editor_draft", "url": "https://x.com/m/status/2"},
        {"tweet_id": "3", "language": "ru", "text_original": "C", "text_en": None,
         "translation_status": "pending", "url": "https://x.com/m/status/3"},
    ]
    result = choose_review_rows(inputs, max_chars=280, count=2)
    assert [r["tweet_id"] for r in result] == ["2", "1"]
    assert all(r["review_decision"] == "" for r in result)
    p = tmp_path / "review.csv"
    write_review_csv(p, result)
    with p.open(encoding="utf-8-sig", newline="") as stream:
        parsed = list(csv.DictReader(stream))
    assert parsed == result


def test_deduplicates_and_rejects_too_long():
    base = {"language": "en", "text_original": "Proof", "text_en": "Proof",
            "translation_status": "original_en"}
    rows = [{**base, "tweet_id": "1"}, {**base, "tweet_id": "2"},
            {**base, "tweet_id": "3", "text_en": "x" * 400}]
    assert len(choose_review_rows(rows, max_chars=280, count=10)) == 1
