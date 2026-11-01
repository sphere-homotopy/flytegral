"""Preview Fly-controlled semantic candidates from durable Fly outbox; never publish."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fly_window.text.meme_engine import preview_candidates  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outbox", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=24)
    args = p.parse_args()
    source = []
    for line in args.outbox.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            source.append(json.loads(line))
    result = preview_candidates(source, count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in result), encoding="utf-8")
    print(json.dumps({"source": len(source), "candidates": len(result), "output": str(args.output)}))


if __name__ == "__main__":
    main()
