import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from apply_editor_translations import apply_edits


def test_editor_translation_retains_ru_and_url_and_is_not_approved():
    original = [{"tweet_id": "42", "language": "ru", "text_original": "Топология",
                 "url": "https://x.com/test/status/42", "translation_status": "pending"}]
    after = apply_edits(original, [{"tweet_id": "42", "text_en": "Topology"}])
    assert after[0]["text_original"] == "Топология"
    assert after[0]["url"] == original[0]["url"]
    assert after[0]["translation_status"] == "editor_draft"
    assert after[0]["training_approved"] is False


def test_editor_translation_does_not_accept_unknown_id():
    import pytest
    with pytest.raises(ValueError, match="unknown"):
        apply_edits([], [{"tweet_id": "not-owned", "text_en": "x"}])
