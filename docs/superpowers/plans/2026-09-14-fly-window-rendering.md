# Fly Window Rendering & X Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn frozen evaluation trajectories, neural activity, and verified metrics into a 25–40 second X-ready video, a hero still, and machine-checkable render metadata without changing or cherry-picking the underlying experiment.

**Architecture:** Evaluation writes a render-neutral episode record containing states, observations, neural activities, actions, rewards, and provenance. Rendering consumes those immutable records offline. The left panel shows the room and trajectory; the right panel shows a real MaleCNS task-subgraph projection with activity intensity; ffmpeg assembles deterministic H.264 output and metadata links every displayed metric back to the evaluation summary.

**Tech Stack:** Python 3.12, numpy, pandas, matplotlib, pillow, httpx, ffmpeg CLI, pytest

**Spec:** `docs/superpowers/specs/2026-09-14-fly-window-design.md`

## Global Constraints

- Final clips come only from frozen evaluation checkpoints.
- The video must visibly distinguish real MaleCNS structure from simplified dynamics/vision/flight physics.
- Displayed success rate must equal the 100-seed held-out evaluation result; it may not be manually edited.
- Training montage may show predetermined checkpoints/episodes but may not cherry-pick a single lucky final result.
- Target duration is 25–40 seconds.
- Core deliverables are training montage, final evaluation demo, hero still, and metrics snapshot.
- MaleCNS source attribution and CC-BY attribution must be present in repository metadata and end-card copy.
- All major completed checkpoints must be committed and pushed before project completion.

---

## File map

- Modify `pyproject.toml` — add Matplotlib and Pillow runtime dependencies.
- Modify `src/fly_window/evaluation/run.py` — optional episode-record export for predetermined render seeds.
- Create `src/fly_window/render/schema.py` — immutable render-record schema.
- Create `src/fly_window/render/skeletons.py` — fetch/cache selected official MaleCNS SWC skeletons.
- Create `src/fly_window/render/project.py` — 3D-to-2D skeleton projection and activity geometry cache.
- Create `src/fly_window/render/frame.py` — split-screen frame renderer.
- Create `src/fly_window/render/video.py` — frame scheduling and ffmpeg assembly.
- Create `scripts/render_episode.py`, `scripts/render_x_video.py`.
- Create `ATTRIBUTION.md`.
- Create tests under `tests/render`.

### Task 1: Export immutable evaluation episode records

**Files:**
- Modify: `pyproject.toml`
- Create: `src/fly_window/render/__init__.py`
- Create: `src/fly_window/render/schema.py`
- Modify: `src/fly_window/evaluation/run.py`
- Test: `tests/render/test_schema.py`

**Interfaces:**
- Produces records under `artifacts/render-records/<model_label>/seed-<seed>.npz` plus same-stem `.json`, with arrays: `state[T,4]`, `observation[T,16]`, `latent_action[T,2]`, `env_action[T,2]`, `reward[T]`, `activity[T,N]`.
- Sidecar JSON fields: `model_label`, `checkpoint_sha256`, `graph_artifact_sha256`, `git_sha`, `seed`, `success`, `total_reward`, `steps`, `evaluation_summary_sha256`.
- Predetermined final-demo render seeds are fixed before evaluation as exactly `10000`, `10001`, and `10002`; render all three outcomes without substitution. If a seed fails, the final demo shows that failure rather than replacing it with a luckier seed.
- Training montage checkpoints: earliest checkpoint, midpoint-by-step checkpoint, and final trained checkpoint; for each checkpoint render development seed `9000` regardless of success.

- [ ] **Step 1: Write schema round-trip and deterministic-selection tests**

Assert arrays round-trip without dtype/shape changes. Given any fake evaluation table, including failures at 10000 or 10001, assert final-demo seeds are always exactly `[10000,10001,10002]` and are never substituted.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/render/test_schema.py -v`
Expected: FAIL.

- [ ] **Step 3: Add render dependencies and implement record writing and selection**

Add `matplotlib` and `pillow` to runtime dependencies. Evaluation must export records only when `--record-render-data` is set, to avoid large default artifacts. Record SHA256 after writing and add hashes to the evaluation JSON.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/render/test_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add pyproject.toml src/fly_window/render/schema.py src/fly_window/evaluation/run.py tests/render/test_schema.py
git commit -m "feat: export immutable render records"
git push
```

### Task 2: Fetch and cache real MaleCNS skeletons for displayed neurons

**Files:**
- Create: `src/fly_window/render/skeletons.py`
- Create: `ATTRIBUTION.md`
- Test: `tests/render/test_skeletons.py`

**Interfaces:**
- Public skeleton URL template: `https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc`.
- Produces: `fetch_skeleton(body_id: int, cache_dir: Path, client: httpx.Client | None = None) -> Path`.
- Produces: `parse_swc(path: Path) -> Skeleton(body_id, xyz: np.ndarray, parent_index: np.ndarray)`.
- Display set: all input/output neurons plus the 256 intermediate neurons with highest max absolute activity in the final three render episodes. This choice is computed, not manually curated.

- [ ] **Step 1: Write mocked fetch/cache and SWC parser tests**

Use a 4-node SWC string; assert parent mapping, XYZ extraction, cached second fetch, and a clear exception for malformed rows.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/render/test_skeletons.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement skeleton acquisition and attribution**

Use the same atomic-download pattern as source acquisition. `ATTRIBUTION.md` states MaleCNS v1.0 is from FlyEM/Janelia and collaborators, links the official project/download pages, records CC-BY, and states that rendered neural dynamics are the project's simplified model rather than data from the biological specimen.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/render/test_skeletons.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/fly_window/render/skeletons.py ATTRIBUTION.md tests/render/test_skeletons.py
git commit -m "feat: fetch real MaleCNS skeletons for rendering"
git push
```

### Task 3: Project skeletons into a canonical lateral activity panel

**Files:**
- Create: `src/fly_window/render/project.py`
- Test: `tests/render/test_project.py`

**Interfaces:**
- Produces: `ProjectionCache(segments_by_body: dict[int, np.ndarray], bounds: tuple[float,float,float,float])`.
- Projection uses MaleCNS SWC coordinates and lateral axes `(z, x)` as documented by the official navis example; y is omitted.
- Normalize all projected coordinates into `[0,1] x [0,1]` using global displayed-skeleton bounds with 3% padding.

- [ ] **Step 1: Write projection tests**

Use two toy skeletons, assert parent-child line segments are generated correctly, global normalization is deterministic, and all normalized coordinates lie in `[0,1]`.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/render/test_project.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement projection/cache**

Precompute line segments once before frame rendering. Store `projection_manifest.json` with body ids, skeleton file hashes, axis choice, and bounds.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/render/test_project.py -v`
Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/fly_window/render/project.py tests/render/test_project.py
git commit -m "feat: project MaleCNS skeleton activity panel"
git push
```

### Task 4: Render deterministic split-screen frames and hero still

**Files:**
- Create: `src/fly_window/render/frame.py`
- Create: `scripts/render_episode.py`
- Test: `tests/render/test_frame.py`

**Interfaces:**
- Output frame size: `1600x900`.
- Left panel: room boundary, bright opening, fly position/heading, trailing 80-step path.
- Right panel: projected real skeleton segments; base alpha `0.08`, activity alpha `0.08 + 0.92 * normalized_abs_activity`.
- Header fields come from metadata: model label, episode seed, success/failure.
- Footer fields come from evaluation summary: `MaleCNS v1.0`, `real wiring / simplified dynamics`, held-out success rate.
- Hero still is the frame immediately before the first successful boundary crossing among the fixed final-demo seeds `[10000,10001,10002]` in that order. If none succeeds, rendering fails the release gate instead of selecting a different seed.

- [ ] **Step 1: Write pixel-size and metadata-source tests**

Render one toy frame and assert PNG dimensions exactly `1600x900`. Monkeypatch metadata and assert displayed success-rate string is derived from the summary object, not accepted as a free-form function argument.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/render/test_frame.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement frame renderer**

Use Matplotlib's Agg backend. Keep visual constants in one frozen `RenderStyle` object. Do not recompute neural simulation; frame renderer receives recorded activity only.

- [ ] **Step 4: Run tests and render one real episode when records exist**

Run: `uv run pytest tests/render/test_frame.py -v`
Expected: PASS.

Then: `uv run python scripts/render_episode.py --record artifacts/render-records/trained/seed-10000.npz --metadata artifacts/render-records/trained/seed-10000.json --output artifacts/videos/episode-preview`
Expected: numbered PNG frames and one `hero.png` if episode succeeds.

- [ ] **Step 5: Commit and push**

```bash
git add src/fly_window/render/frame.py scripts/render_episode.py tests/render/test_frame.py
git commit -m "feat: render fly and connectome split screen"
git push
```

### Task 5: Assemble the 25–40 second X video with ffmpeg

**Files:**
- Create: `src/fly_window/render/video.py`
- Create: `scripts/render_x_video.py`
- Test: `tests/render/test_video.py`

**Interfaces:**
- Output: `artifacts/videos/fly-window-x.mp4`, H.264, yuv420p, 30 fps, 1600x900, `faststart`.
- Timeline target 32 seconds:
  - `0–3s`: hook card
  - `3–9s`: earliest checkpoint / seed 9000
  - `9–15s`: midpoint checkpoint / seed 9000
  - `15–21s`: final checkpoint / seed 9000
  - `21–30s`: the three fixed held-out seeds `10000`, `10001`, `10002`, 3s each, preserving success or failure
  - `30–32s`: end card with exact held-out metric and caveats
- If an episode is longer than its slot, uniformly time-sample recorded frames; never alter actions/trajectory.

- [ ] **Step 1: Write timeline construction tests**

Given synthetic 30-fps episode lengths, assert the schedule has exactly `960` output frames for 32 seconds, uses the required episode categories/seeds in order, and end-card metric is loaded from evaluation JSON.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/render/test_video.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement timeline and ffmpeg command construction**

Generate frames in a temporary directory, then invoke:

```bash
ffmpeg -y -framerate 30 -i frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart -crf 18 \
  artifacts/videos/fly-window-x.mp4
```

Raise a clear error if `ffmpeg` is absent or exits nonzero; preserve frame directory path in the error for debugging.

- [ ] **Step 4: Add video manifest**

Write `fly-window-x.manifest.json` with source run ids, checkpoint hashes, evaluation hash, selected seeds, frame ranges, renderer git SHA, output SHA256, resolution, fps, duration.

- [ ] **Step 5: Run unit tests and full render**

Run: `uv run pytest tests/render -v`
Expected: PASS.

Run: `uv run python scripts/render_x_video.py --evaluation artifacts/evaluation/evaluation.json --records-dir artifacts/render-records --output artifacts/videos/fly-window-x.mp4`
Expected: valid 25–40s MP4 plus manifest and hero still.

- [ ] **Step 6: Verify media mechanically**

Run:

```bash
ffprobe -v error -show_entries format=duration \
  -show_entries stream=width,height,r_frame_rate,pix_fmt \
  -of json artifacts/videos/fly-window-x.mp4
```

Expected: duration between 25 and 40 seconds, width 1600, height 900, frame rate 30/1, pixel format yuv420p.

- [ ] **Step 7: Commit and push Checkpoint E**

Do not commit the MP4 or PNG frame directory. Commit renderer code, tests, attribution, and the small render manifest/metrics summary containing the output SHA256; deliver the MP4 as a generated artifact outside git.

```bash
git add src/fly_window/render scripts/render_episode.py scripts/render_x_video.py tests/render ATTRIBUTION.md
git commit -m "feat: produce auditable X-ready fly-window video"
git push
```

## Plan acceptance gate

Checkpoint E is complete only when render tests pass, `ffprobe` validates the final MP4, every displayed metric matches the frozen evaluation JSON, the video manifest hashes the exact source run/checkpoints, and the final implementation commit is pushed. Final completion report must include branch name and pushed commit SHA.
