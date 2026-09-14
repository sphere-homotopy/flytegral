# Fly Window Data & Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible MaleCNS v1.0 acquisition and subgraph-extraction pipeline that produces one compact, auditable graph artifact for the Fly Window simulator.

**Architecture:** Use the official public MaleCNS v1.0 bulk Feather files as the primary source so the core pipeline does not depend on a neuPrint token. Read curated neuron annotations and the full segment-to-segment connection-strength table, select a biologically constrained visual-to-descending task subgraph with deterministic graph rules, and emit NPZ/Parquet artifacts plus a machine-readable provenance manifest.

**Tech Stack:** Python 3.12, uv, pandas, pyarrow, numpy, scipy, httpx, pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-09-14-fly-window-design.md`

## Global Constraints

- Source dataset is `MaleCNS v1.0`.
- Real connectome topology must be preserved in the trained model; no decorative/arbitrary network may substitute for it.
- Dataset files are never committed to git; only manifests, configs, and small derived summaries are committed.
- Every source URL, retrieval date, SHA256, selection rule, node count, edge count, and pruning threshold must be recorded.
- Subgraph extraction must be deterministic for a given config and source-file hashes.
- The first release uses a task-focused MaleCNS-derived subgraph rather than the whole CNS.
- All major completed checkpoints must be committed and pushed before starting the next major phase.

---

## File map

- Create `pyproject.toml` — package metadata, runtime/test dependencies, pytest config.
- Create `.gitignore` — exclude source datasets, run artifacts, virtualenvs, caches, and secrets.
- Create `src/fly_window/__init__.py` — package root.
- Create `src/fly_window/data/urls.py` — canonical MaleCNS v1.0 source URLs and filenames.
- Create `src/fly_window/data/download.py` — streamed HTTP download with SHA256 manifesting.
- Create `src/fly_window/data/schema.py` — typed records/configs for graph nodes, edges, source files, and subgraph selection.
- Create `src/fly_window/data/load.py` — load/normalize annotation and connectivity Feather tables.
- Create `src/fly_window/data/select.py` — deterministic visual-to-descending subgraph extraction.
- Create `src/fly_window/data/export.py` — compact Parquet/NPZ graph artifact and provenance JSON writer.
- Create `scripts/fetch_malecns.py` — CLI for source acquisition.
- Create `scripts/build_subgraph.py` — CLI for extraction/export.
- Create `configs/subgraph_v1.json` — exact extraction thresholds.
- Create `PROVENANCE.md` — human-readable methodology and source attribution.
- Create `tests/data/test_download.py`, `test_load.py`, `test_select.py`, `test_export.py`.

### Task 1: Bootstrap the Python project and immutable MaleCNS source catalog

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/fly_window/__init__.py`
- Create: `src/fly_window/data/__init__.py`
- Create: `src/fly_window/data/urls.py`
- Test: `tests/data/test_urls.py`

**Interfaces:**
- Produces: `MALECNS_DATASET = "male-cns:v1.0"`
- Produces: `SourceFile(name: str, url: str)` constants `ANNOTATIONS_SOURCE`, `NEUROTRANSMITTER_SOURCE`, and `CONNECTIVITY_SOURCE`.

- [ ] **Step 1: Write the failing source-catalog test**

```python
from fly_window.data.urls import ANNOTATIONS_SOURCE, CONNECTIVITY_SOURCE, NEUROTRANSMITTER_SOURCE, MALECNS_DATASET


def test_malecns_v1_sources_are_pinned():
    assert MALECNS_DATASET == "male-cns:v1.0"
    assert ANNOTATIONS_SOURCE.name == "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    assert NEUROTRANSMITTER_SOURCE.name == "body-neurotransmitters-male-cns-v1.0.feather"
    assert CONNECTIVITY_SOURCE.name == "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    assert ANNOTATIONS_SOURCE.url.startswith("https://storage.googleapis.com/flyem-male-cns/v1.0/")
    assert CONNECTIVITY_SOURCE.url.startswith("https://storage.googleapis.com/flyem-male-cns/v1.0/")
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `uv run pytest tests/data/test_urls.py -v`
Expected: FAIL because `fly_window.data.urls` does not exist.

- [ ] **Step 3: Add package config and pinned URLs**

`pyproject.toml` must require Python `>=3.12` and include: `numpy`, `scipy`, `pandas`, `pyarrow`, `httpx`, `pydantic`; dev dependency `pytest`.

`src/fly_window/data/urls.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str


MALECNS_DATASET = "male-cns:v1.0"
_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
ANNOTATIONS_SOURCE = SourceFile(
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    f"{_BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
)
NEUROTRANSMITTER_SOURCE = SourceFile(
    "body-neurotransmitters-male-cns-v1.0.feather",
    f"{_BASE}/body-neurotransmitters-male-cns-v1.0.feather",
)
CONNECTIVITY_SOURCE = SourceFile(
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    f"{_BASE}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)
```

`.gitignore` must include `.venv/`, `__pycache__/`, `.pytest_cache/`, `data/raw/`, `artifacts/`, `.env`.

- [ ] **Step 4: Run the test**

Run: `uv sync && uv run pytest tests/data/test_urls.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push checkpoint**

```bash
git add pyproject.toml .gitignore src tests/data/test_urls.py
git commit -m "build: bootstrap fly-window python project"
git push -u origin <implementation-branch>
```

### Task 2: Stream-download official source files and record hashes

**Files:**
- Create: `src/fly_window/data/download.py`
- Create: `scripts/fetch_malecns.py`
- Test: `tests/data/test_download.py`

**Interfaces:**
- Produces: `DownloadRecord(source_name: str, url: str, path: str, size_bytes: int, sha256: str)`
- Produces: `download_source(source: SourceFile, destination_dir: Path, client: httpx.Client | None = None) -> DownloadRecord`
- CLI: `uv run python scripts/fetch_malecns.py --output data/raw`

- [ ] **Step 1: Write tests for hashing, atomic writes, and reuse**

Use `httpx.MockTransport` to return a known byte string. Assert the function writes to `*.part` then atomically renames, returns the expected SHA256, and on a second call re-hashes/reuses the completed file without a second HTTP request.

```python
def test_download_source_is_atomic_and_hashes(tmp_path):
    payload = b"male-cns-test"
    calls = 0
    # MockTransport handler increments calls and returns payload.
    record = download_source(TEST_SOURCE, tmp_path, client=client)
    assert record.size_bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / TEST_SOURCE.name).read_bytes() == payload
    assert not (tmp_path / f"{TEST_SOURCE.name}.part").exists()
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/data/test_download.py -v`
Expected: FAIL because `download_source` is undefined.

- [ ] **Step 3: Implement streamed download and manifest output**

Use `httpx.Client(follow_redirects=True, timeout=None)` and `response.iter_bytes(1024 * 1024)`. Compute SHA256 while streaming, `fsync()` the temporary file, then `Path.replace()` it. If final file exists, hash it and return without downloading.

The CLI downloads all three canonical files and writes `data/raw/source-manifest.json` containing dataset id, UTC retrieval time, and all three `DownloadRecord`s.

- [ ] **Step 4: Run unit tests and CLI help**

Run: `uv run pytest tests/data/test_download.py -v && uv run python scripts/fetch_malecns.py --help`
Expected: PASS and CLI usage text.

- [ ] **Step 5: Commit and push checkpoint**

```bash
git add src/fly_window/data/download.py scripts/fetch_malecns.py tests/data/test_download.py
git commit -m "feat: add reproducible MaleCNS downloader"
git push
```

### Task 3: Normalize MaleCNS annotations and weighted edges

**Files:**
- Create: `src/fly_window/data/schema.py`
- Create: `src/fly_window/data/load.py`
- Test: `tests/data/test_load.py`

**Interfaces:**
- Produces: `NormalizedTables(nodes: pd.DataFrame, edges: pd.DataFrame)`
- Produces: `load_malecns_tables(annotation_path: Path, neurotransmitter_path: Path, connectivity_path: Path) -> NormalizedTables`
- Node columns after normalization: `body_id:int64`, `type:str|None`, `superclass:str|None`, `cell_class:str|None`, `side:str|None`, `nt:str|None`.
- Edge columns after normalization: `src:int64`, `dst:int64`, `weight:int64`.

- [ ] **Step 1: Create tiny synthetic Feather fixtures in the test**

Write annotation, neurotransmitter, and connectivity DataFrames with the actual expected semantic fields but intentionally varied integer dtypes and nullable strings. Test both categorical neurotransmitter input and probability-column argmax (including a tie). Assert output columns/names/dtypes and that zero/negative weights are rejected.

- [ ] **Step 2: Run the test and verify failure**

Run: `uv run pytest tests/data/test_load.py -v`
Expected: FAIL because `load_malecns_tables` does not exist.

- [ ] **Step 3: Implement column resolution and validation**

Implement a private `_resolve_column(frame, candidates)` helper so the loader accepts source naming variants such as `bodyId`/`body` for neuron ids and `body_pre`/`bodyPre` plus `body_post`/`bodyPost` for edges. Raise `ValueError` with the full available-column list if a required semantic column cannot be resolved.

Resolve the neuron id from the neurotransmitter Feather table. For the neurotransmitter label, first accept a categorical column named one of `predicted_nt`, `neurotransmitter`, or `nt`; if none exists, resolve probability columns for the known labels `acetylcholine`, `gaba`, `glutamate`, `dopamine`, `serotonin`, and `octopamine` and choose the row-wise argmax, breaking exact ties lexicographically. Left-join the resulting label onto annotations by body id and expose it as `nt`. Keep only the normalized columns above, drop self-loops, require `weight > 0`, and merge duplicate `(src, dst)` rows by summing weights.

- [ ] **Step 4: Run the test suite**

Run: `uv run pytest tests/data/test_load.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push checkpoint**

```bash
git add src/fly_window/data/schema.py src/fly_window/data/load.py tests/data/test_load.py
git commit -m "feat: normalize MaleCNS graph tables"
git push
```

### Task 4: Deterministically select the visual-to-descending task subgraph

**Files:**
- Create: `configs/subgraph_v1.json`
- Create: `src/fly_window/data/select.py`
- Test: `tests/data/test_select.py`

**Interfaces:**
- Produces: `SubgraphConfig(min_edge_weight=3, max_hops=4, max_nodes=6000, output_superclass="descending_neuron", visual_type_prefixes=("LC", "LPLC", "LT", "MC", "Mi", "Tm", "T4", "T5"))`
- Produces: `select_task_subgraph(tables: NormalizedTables, config: SubgraphConfig) -> NormalizedTables`
- Selection rule:
  1. output nodes are all annotated `superclass == "descending_neuron"`;
  2. input seed nodes are typed neurons whose type begins with one of the configured visual prefixes;
  3. remove edges below `min_edge_weight`;
  4. forward BFS from visual seeds and reverse BFS from outputs up to `max_hops`;
  5. keep the intersection plus input/output seeds;
  6. if above `max_nodes`, rank intermediates deterministically by `(forward_depth + reverse_depth, -incident_retained_weight, body_id)` and keep the best-ranked nodes; never drop seeds;
  7. keep all surviving edges whose endpoints survive.

- [ ] **Step 1: Write a graph-shaped fixture that proves path intersection**

Create visual seed `LC01`, descending output `DNa01`, one valid 3-hop path, one dead-end visual branch, one unrelated high-weight branch, and one edge under threshold. Assert only the valid path plus seeds survives.

- [ ] **Step 2: Run the test and verify failure**

Run: `uv run pytest tests/data/test_select.py -v`
Expected: FAIL because selector is undefined.

- [ ] **Step 3: Implement the deterministic graph selection**

Build a SciPy CSR adjacency for forward traversal and CSC adjacency for reverse traversal after sorting nodes by `body_id` and edges by `(src,dst)`. Perform bounded frontier expansion for exactly `max_hops`, storing minimum forward/reverse depth per node. Compute `incident_retained_weight` from thresholded edges. Validate that at least one visual seed and one descending output exist; otherwise raise `ValueError` including counts and configured prefixes.

- [ ] **Step 4: Add determinism and cap tests**

Call `select_task_subgraph` twice on shuffled row orders and assert byte-for-byte equal sorted outputs. Add a fixture exceeding `max_nodes` and assert seeds survive while the deterministic score cap is honored.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/data/test_select.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and push checkpoint**

```bash
git add configs/subgraph_v1.json src/fly_window/data/select.py tests/data/test_select.py
git commit -m "feat: extract visual-to-descending MaleCNS subgraph"
git push
```

### Task 5: Export compact artifacts and auditable provenance

**Files:**
- Create: `src/fly_window/data/export.py`
- Create: `scripts/build_subgraph.py`
- Create: `PROVENANCE.md`
- Test: `tests/data/test_export.py`

**Interfaces:**
- Produces: `export_subgraph(graph: NormalizedTables, config: SubgraphConfig, source_manifest: dict, output_dir: Path) -> dict`
- Outputs:
  - `artifacts/graphs/subgraph_v1_nodes.parquet`
  - `artifacts/graphs/subgraph_v1_edges.parquet`
  - `artifacts/graphs/subgraph_v1.npz` with arrays `body_ids`, `src_idx`, `dst_idx`, `weights`, `input_mask`, `output_mask`
  - `artifacts/graphs/subgraph_v1_manifest.json`

- [ ] **Step 1: Write export round-trip test**

Build a tiny `NormalizedTables`, export it, reload all files, and assert body-id/index consistency, masks, edge weights, source hashes, config, node count, edge count, and deterministic semantic `artifact_sha256` across two exports of identical inputs.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/data/test_export.py -v`
Expected: FAIL because exporter is undefined.

- [ ] **Step 3: Implement export and manifest hashing**

Sort nodes by `body_id`; map edges to dense indices; define `input_mask` from configured visual prefixes and `output_mask` from descending superclass. Record ordinary file SHA256 values for Parquet/NPZ integrity, but compute the deterministic semantic `artifact_sha256` independently of container serialization: hash canonical JSON metadata plus each logical array name, dtype, shape, and C-order raw bytes in fixed order (`body_ids`, `src_idx`, `dst_idx`, `weights`, `input_mask`, `output_mask`). This avoids ZIP/Parquet metadata timestamps changing the semantic graph hash.

`PROVENANCE.md` must state the exact official dataset id, the three official GCS-derived HTTPS URLs, MaleCNS CC-BY licensing/attribution, the selection rule, and that visual prefixes are an engineering proxy rather than a claim that every selected cell is a photoreceptor.

- [ ] **Step 4: Add CLI and dry-run summary**

`build_subgraph.py` must accept `--raw-dir`, `--config`, `--output-dir`, and `--dry-run`. Dry-run prints input/output seed counts plus projected subgraph node/edge counts without writing artifacts.

- [ ] **Step 5: Run unit suite and, when raw data is present, the dry run**

Run: `uv run pytest tests/data -v`
Expected: PASS.

If source files have been downloaded, run:
`uv run python scripts/build_subgraph.py --raw-dir data/raw --config configs/subgraph_v1.json --output-dir artifacts/graphs --dry-run`
Expected: nonzero visual seeds, nonzero descending outputs, nonzero subgraph.

- [ ] **Step 6: Commit and push Checkpoint A code**

```bash
git add src/fly_window/data scripts configs/subgraph_v1.json PROVENANCE.md tests/data
git commit -m "feat: export auditable MaleCNS task graph"
git push
```

## Plan acceptance gate

Checkpoint A is complete only when all `tests/data` pass, source acquisition succeeds or an explicit network failure is reported, the real MaleCNS files produce a nonempty task subgraph, and the generated manifest contains exact source hashes and graph counts. Do not begin environment implementation before this checkpoint is pushed.
