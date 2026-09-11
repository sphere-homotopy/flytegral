# Flytegral

A fruit fly looks at a function graph and estimates a definite integral by moving a numerical slider.

The visual joke is the product: **graph → visible fly → slider → numerical answer**. The MVP is also structured as a deterministic benchmark so the transparent baseline can later be replaced by a MaleCNS/connectome-backed agent without rewriting the task or rendering pipeline.

## Demo

The project is dependency-free and has no build step. Because the browser code uses ES modules, serve the repository as static files:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/`.

Use a reproducible problem with a URL seed, for example:

```text
http://localhost:8000/?seed=260912
```

The UI shows:

- a generated cubic polynomial and highlighted integration interval;
- an animated fruit-fly agent with visible activity while it "thinks";
- a slider controlled by the agent;
- target integral, prediction, and absolute error;
- a deterministic seed and `New problem` control.

`Record demo` uses the browser's tab/screen capture API. It saves MP4 when the browser exposes an MP4 `MediaRecorder` codec and otherwise falls back to WebM.

## Tests and benchmark

Requires Node 20+ (Node 22 is used during development).

```bash
npm test
npm run smoke
npm run evaluate
```

The deterministic benchmark uses a 256-problem training split and 128-problem test split and reports MAE plus mean relative error (excluding targets too close to zero for a stable relative metric).

Current comparison includes:

- `zero` — always predicts zero;
- `midpoint` — one-point midpoint numerical estimate;
- `baseline` — the transparent noisy-target agent used by the visual demo;
- `linear-graph` — a small trainable regression agent using graph samples from the interval.

The linear graph agent is intentionally simple. For cubic polynomials, the learned sample weights can recover an exact quadrature rule, so it is a pipeline sanity check rather than a claim about biological computation.

## Architecture

```text
index.html              semantic demo layout
styles.css              light responsive UI + fly/activity animation
src/math.js             seeded RNG, polynomial generation, exact integral, slider mapping
src/environment.js      graph observations/features + normalized reward
src/agent.js            transparent visible baseline agent
src/training.js         trainable linear graph-sample baseline
src/evaluation.js       deterministic splits, baselines, MAE/relative metrics
src/brain.js            sensor encoder / motor decoder / connectome adapter boundary
src/app.js              graph rendering, animation, round lifecycle, recording
scripts/evaluate.mjs    benchmark CLI
scripts/smoke-demo.mjs  dependency-free smoke check
tests/                  deterministic unit + contract + demo smoke tests
```

The agent contract remains small:

```js
agent.estimate(problem, options)
// -> { kind, value, sliderPosition, confidence?, trace? }
```

For the future fly-brain path, `ConnectomeAgentAdapter` composes three independent parts:

```text
problem / graph observation
        ↓
GraphSensorEncoder
        ↓
connectome network.run(...)
        ↓
SliderMotorDecoder
        ↓
agent result / slider action
```

The environment does not know the internal topology of the agent. A MaleCNS implementation can therefore replace the `network.run` component while preserving task generation, reward, evaluation, and rendering.

## Important baseline note

The animated demo currently labels itself **baseline**. It is deliberately not presented as a connectome simulation: its estimate starts from the exact target and adds deterministic seed-dependent error, exactly as specified for the MVP interaction prototype.

See `docs/superpowers/specs/2026-09-12-flytegral-mvp-design.md` for the approved design.
