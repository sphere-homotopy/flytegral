import json
import zipfile
from pathlib import Path

from scripts.prepare_owner_math_corpus import parse_archive_records, select_math_tweets


def test_archive_reads_only_tweets_and_filters_retweets_replies(tmp_path: Path):
    records = [
        {"tweet": {"id_str": "123", "full_text": "Люблю топологию, но что она мне сделала", "created_at": "now"}},
        {"tweet": {"id_str": "124", "full_text": "RT @other: math is nice"}},
        {"tweet": {"id_str": "125", "full_text": "topology is not a geometry", "in_reply_to_status_id_str": "99"}},
        {"tweet": {"id_str": "126", "full_text": "cat got food"}},
        {"tweet": {"id_str": "127", "full_text": "Theorem: I am a fly"}},
    ]
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("data/tweets.js", "window.YTD.tweets.part0 = " + json.dumps(records))
        out.writestr("data/direct-messages.js", "SENSITIVE")
    source = parse_archive_records(archive)
    result = select_math_tweets(source, handle="@sphere_homotopy")
    assert [r["tweet_id"] for r in result] == ["123", "127"]
    assert result[0]["language"] == "ru"
    assert result[0]["text_en"] is None
    assert result[1]["text_en"] == "Theorem: I am a fly"
    assert all("SENSITIVE" not in str(x) for x in result)


def test_math_corpus_rejects_duplicate_text():
    items = [{"tweet": {"id_str": str(i), "full_text": "A proof of the theorem"}} for i in (1, 2)]
    result = select_math_tweets(items, handle="sphere_homotopy")
    assert len(result) == 1
