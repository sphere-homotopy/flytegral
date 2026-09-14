from __future__ import annotations

import argparse
import json
from pathlib import Path

from fly_window.data.export import export_subgraph
from fly_window.data.load import load_malecns_tables
from fly_window.data.select import SubgraphConfig, select_task_subgraph
from fly_window.data.urls import ANNOTATIONS_SOURCE, CONNECTIVITY_SOURCE, NEUROTRANSMITTER_SOURCE


def _load_config(path: Path) -> SubgraphConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["visual_type_prefixes"] = tuple(raw["visual_type_prefixes"])
    return SubgraphConfig(**raw)


def _summary(graph, config: SubgraphConfig) -> dict[str, int]:
    input_count = int(
        graph.nodes["type"].fillna("").astype(str).str.startswith(config.visual_type_prefixes).sum()
    )
    output_count = int(
        graph.nodes["superclass"].fillna("").astype(str).eq(config.output_superclass).sum()
    )
    return {
        "input_seed_count": input_count,
        "output_seed_count": output_count,
        "node_count": int(len(graph.nodes)),
        "edge_count": int(len(graph.edges)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pinned Fly Window MaleCNS task subgraph.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/subgraph_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/graphs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    manifest_path = args.raw_dir / "source-manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tables = load_malecns_tables(
        args.raw_dir / ANNOTATIONS_SOURCE.name,
        args.raw_dir / NEUROTRANSMITTER_SOURCE.name,
        args.raw_dir / CONNECTIVITY_SOURCE.name,
    )
    selected = select_task_subgraph(tables, config)
    summary = _summary(selected, config)

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    exported = export_subgraph(selected, config, source_manifest, args.output_dir)
    print(json.dumps({**summary, "artifact_sha256": exported["artifact_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
