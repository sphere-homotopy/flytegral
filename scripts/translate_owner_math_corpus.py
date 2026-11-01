"""Produce auditable RU -> EN translation drafts using local Argos Translate.

All inference runs on the user's machine. Model download is optional and explicit;
outputs stay in .private/ by default. No publish/train integration.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def install_ru_en_package() -> None:
    import argostranslate.package as package

    package.update_package_index()
    matches = [
        item for item in package.get_available_packages()
        if item.from_code == "ru" and item.to_code == "en"
    ]
    if len(matches) != 1:
        raise RuntimeError("Could not uniquely locate the ru-en Argos model")
    package.install_from_path(matches[0].download())


def translate_rows(
    rows: list[dict],
    translator: Callable[[str], str],
    *,
    existing: dict[str, dict] | None = None,
    limit: int | None = None,
) -> list[dict]:
    previous = existing or {}
    result: list[dict] = []
    attempted = 0
    for original in rows:
        row = dict(original)
        key = str(row["tweet_id"])
        saved = previous.get(key)
        if saved and saved.get("translation_status") in ("machine_draft", "editor_draft", "original_en", "reviewed"):
            result.append(saved)
            continue
        if row["language"] == "en":
            row["text_en"] = row["text_original"]
            row["translation_status"] = "original_en"
        elif limit is None or attempted < limit:
            attempted += 1
            text = translator(str(row["text_original"]))
            if not text.strip() or CYRILLIC.search(text):
                row["text_en"] = None
                row["translation_status"] = "needs_retranslation"
            else:
                row["text_en"] = text.strip()
                row["translation_status"] = "machine_draft"
        result.append(row)
    return result


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--install-model", action="store_true", help="Download Argos RU-EN model explicitly")
    p.add_argument("--model-file", type=Path, help="Install RU-EN .argosmodel from an already downloaded local file")
    p.add_argument("--limit", type=int, default=None, help="Translate up to N pending RU tweets per call")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.model_file is not None:
        import argostranslate.package as package
        if not args.model_file.is_file():
            raise FileNotFoundError(args.model_file)
        package.install_from_path(args.model_file)
    elif args.install_model:
        install_ru_en_package()

    import argostranslate.translate as argos
    installed = {lang.code for lang in argos.get_installed_languages()}
    if not {"ru", "en"} <= installed:
        raise RuntimeError("Argos RU-EN model not installed; rerun with --install-model")
    raw = load_jsonl(args.input)
    prev = {str(row["tweet_id"]): row for row in load_jsonl(args.output)}
    results = translate_rows(raw, lambda text: argos.translate(text, "ru", "en"), existing=prev, limit=args.limit)
    save_jsonl_atomic(args.output, results)
    totals: dict[str, int] = {}
    for row in results:
        status = str(row["translation_status"])
        totals[status] = totals.get(status, 0) + 1
    print(json.dumps({"selected": len(results), "translation_status": totals, "output": str(args.output)}))


if __name__ == "__main__":
    main()
