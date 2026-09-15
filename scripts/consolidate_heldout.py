from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fly_window.evaluation.consolidate import consolidate_heldout_results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate biological, shuffled, and untrained 100-seed held-out evaluations."
    )
    parser.add_argument("--biological", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--untrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="Write a failed gate for diagnostics instead of exiting non-zero.",
    )
    args = parser.parse_args()

    biological = _read(args.biological)
    shuffled = _read(args.shuffled)
    untrained = _read(args.untrained)
    result = consolidate_heldout_results(
        biological=biological,
        shuffled=shuffled,
        untrained=untrained,
    )
    result["sources"] = {
        "biological": {
            "path": str(args.biological),
            "sha256": _sha256(args.biological),
            "elapsed_seconds": biological.get("elapsed_seconds"),
            "batch_size": biological.get("batch_size"),
        },
        "shuffled": {
            "path": str(args.shuffled),
            "sha256": _sha256(args.shuffled),
            "elapsed_seconds": shuffled.get("elapsed_seconds"),
            "batch_size": shuffled.get("batch_size"),
        },
        "untrained": {
            "path": str(args.untrained),
            "sha256": _sha256(args.untrained),
            "elapsed_seconds": untrained.get("elapsed_seconds"),
            "batch_size": untrained.get("batch_size"),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    if not result["release_gate"]["passed"] and not args.allow_failed_gate:
        raise SystemExit("release gate failed")


if __name__ == "__main__":
    main()
