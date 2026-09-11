# Flytegral

A fruit fly looks at a function graph and estimates a definite integral by moving a numerical slider.

The visual joke is the product: **graph → visible fly → slider → numerical answer**. The repository now has two explicit agent modes:

- **baseline** — lightweight deterministic browser-only demo;
- **MaleCNS v1.0** — optional local whole-connectome runtime using a pinned DOOMFLY checkout plus a separately trained Flytegral scalar readout.

The two modes are deliberately labeled differently. Flytegral never silently substitutes the baseline when `agent=malecns` is requested.

## Baseline demo

The browser MVP is dependency-free and has no build step. Because it uses ES modules, serve the repository as static files:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/?seed=260912
```

The UI shows a generated cubic polynomial, highlighted integration interval, animated fly, answer slider, exact target, prediction and absolute error. `Record demo` saves MP4 when the browser exposes an MP4 `MediaRecorder` codec and otherwise falls back to WebM.

## MaleCNS mode

MaleCNS mode is intentionally a local sidecar rather than a fake browser-sized connectome. The pinned upstream runtime is `nftechie/doomfly` at commit:

```text
71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33
```

DOOMFLY uses the September 2026 MaleCNS v1.0 reconstruction and an approximate whole-graph LIF model. Its own documentation is explicit that the neural dynamics and sensory mappings are modeling assumptions rather than a literal living-fly emulation. Flytegral preserves that distinction.

### 1. Prepare the pinned connectome runtime

Python 3.11, Git and several GB of free RAM/disk are required. The upstream edge file alone is roughly 1.1 GB.

```bash
./scripts/bootstrap-malecns.sh
```

This creates an ignored `.vendor/doomfly` checkout, verifies the upstream MaleCNS source hashes, imports the graph, and produces:

```text
.vendor/doomfly/outputs/doom/malecns_v1/graph.npz
```

### 2. Train the Flytegral readout

The connectome topology and synaptic weights stay fixed. Training only fits a compact ridge-regression readout from deterministic hashed whole-brain spike-rate features to the numerical slider position.

```bash
.venv-malecns/bin/python -m brain_runtime.train_readout --train 128 --test 64
```

The command writes ignored local artifacts:

```text
brain_runtime/readout.npz
brain_runtime/readout.metrics.json
```

The metrics file records the deterministic train/test seeds, readout hyperparameters, midpoint baseline, test MAE, upstream commit and the explicit fact that topology/synaptic weights were not trained.

### 3. Start the local neural bridge

```bash
.venv-malecns/bin/python -m brain_runtime.server
```

It binds only to:

```text
http://127.0.0.1:8777
```

### 4. Run the browser in MaleCNS mode

Keep the static server running and open:

```text
http://localhost:8000/?seed=260912&agent=malecns
```

The browser rasterizes the visible graph into a 64×40 luminance stimulus. The local runtime samples that raster at the DOOMFLY R1–R6 retinal UV mapping, resets the whole MaleCNS model for an independent trial, propagates neural activity, and decodes the resulting spike-rate population into the answer slider.

You can override the bridge URL for local experiments:

```text
http://localhost:8000/?agent=malecns&brain=http://127.0.0.1:8777
```

## Tests and benchmark

Requires Node 20+ for the browser/core suite. Node 22 is used during development.

```bash
npm test
npm run smoke
npm run evaluate
npm run test:brain-runtime
```

The ordinary deterministic benchmark uses a 256-problem training split and 128-problem test split and reports MAE plus mean relative error where the denominator is stable. It compares `zero`, `midpoint`, the visual `baseline`, and a small `linear-graph` sanity model.

The MaleCNS readout has its own deterministic train/test split because collecting each example requires an actual whole-connectome simulation. Its test metrics are emitted by `brain_runtime.train_readout` and are not claimed until that runtime has actually been executed.

## Architecture

```text
index.html                    semantic demo layout
styles.css                    light responsive UI + fly/activity animation
src/math.js                   seeded polynomial tasks + exact integral + slider mapping
src/environment.js            observations/features + normalized reward
src/agent.js                  transparent browser baseline
src/training.js               small linear graph-sample sanity baseline
src/evaluation.js             deterministic browser benchmark
src/brain.js                  generic sensor/network/motor adapter boundary
src/malecns.js                graph raster encoder + local MaleCNS HTTP agent
src/app.js                    graph rendering, agent selection, animation, recording
brain_runtime/runtime.py      DOOMFLY loader, retinal sampling, whole-brain trial, readout
brain_runtime/training.py     JS-compatible task/raster generation + ridge fitting
brain_runtime/train_readout.py deterministic real-runtime readout training/evaluation
brain_runtime/server.py       local-only MaleCNS bridge
scripts/bootstrap-malecns.sh  pinned upstream checkout + verified MaleCNS preparation
scripts/evaluate.mjs          browser benchmark CLI
scripts/smoke-demo.mjs        dependency-free browser smoke check
tests/                        JS contracts and demo tests
```

The browser agent contract stays small:

```js
await agent.estimate(problem, options)
// -> { kind, value, sliderPosition, confidence?, trace?, telemetry? }
```

For MaleCNS the pipeline is:

```text
polynomial graph
      ↓
GraphRasterSensorEncoder (64×40 luminance)
      ↓
R1–R6 retinal UV sampling
      ↓
pinned MaleCNS v1.0 recurrent graph / DOOMFLY LIF runtime
      ↓
whole-brain spike-rate population
      ↓
trained scalar readout only
      ↓
slider position / numerical answer
```

See `THIRD_PARTY.md` for upstream attribution and `docs/superpowers/specs/2026-09-12-flytegral-mvp-design.md` for the original approved MVP design.
