import json

from scripts.translate_owner_math_corpus import translate_rows, save_jsonl_atomic, load_jsonl


def test_drafts_keep_original_and_attribution_and_allow_resume(tmp_path):
    original = [{
        "tweet_id": "7", "language": "ru", "text_original": "Доказательство",
        "url": "https://x.com/example/status/7", "translation_status": "pending",
    }, {
        "tweet_id": "8", "language": "en", "text_original": "A proof",
        "url": "https://x.com/example/status/8", "translation_status": "original_en",
    }]
    rows = translate_rows(original, lambda text: "Proof")
    assert rows[0]["text_original"] == "Доказательство"
    assert rows[0]["text_en"] == "Proof"
    assert rows[0]["url"] == original[0]["url"]
    assert rows[0]["translation_status"] == "machine_draft"
    assert rows[1]["text_en"] == "A proof"
    rows2 = translate_rows(original, lambda _: (_ for _ in ()).throw(AssertionError("retranslated")), existing={"7": rows[0]})
    assert rows2 == rows
    path = tmp_path / "translated.jsonl"
    save_jsonl_atomic(path, rows)
    assert load_jsonl(path) == rows


def test_invalid_translation_remains_pending_review():
    row = {"tweet_id": "1", "language": "ru", "text_original": "Топология"}
    result = translate_rows([row], lambda _: "Топология")
    assert result[0]["translation_status"] == "needs_retranslation"
    assert result[0]["text_en"] is None
