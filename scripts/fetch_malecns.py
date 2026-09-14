from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from fly_window.data.download import download_source
from fly_window.data.urls import (
    ANNOTATIONS_SOURCE,
    CONNECTIVITY_SOURCE,
    MALECNS_DATASET,
    NEUROTRANSMITTER_SOURCE,
)


SOURCES = (ANNOTATIONS_SOURCE, NEUROTRANSMITTER_SOURCE, CONNECTIVITY_SOURCE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned MaleCNS v1.0 source files.")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = [download_source(source, args.output) for source in SOURCES]
    manifest = {
        "dataset": MALECNS_DATASET,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [record.as_dict() for record in records],
    }
    manifest_path = args.output / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
