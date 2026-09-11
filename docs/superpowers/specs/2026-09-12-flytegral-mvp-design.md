# Flytegral MVP Design

## Goal

Build a browser demo in which a visible fruit fly looks at a plotted function and moves a numerical slider to estimate the definite integral over a displayed interval. The result should be immediately understandable in a short screen recording and should also be structured as a real benchmark that can later swap in a MaleCNS connectome-constrained agent.

## Scope

The MVP is a single-page static web app with no backend and no build step. It must run by opening `index.html` or serving the repository as static files.

The first agent is a transparent baseline, not a fake claim of connectome inference. The UI and benchmark are built first; a real connectome adapter is a later milestone.

## User experience

A round starts with a generated function and integration interval. The left side shows the graph on a light background with the interval visually highlighted. The center shows a visible fly with a short animated thinking state. The right side shows a slider whose motion represents the fly's answer. When the answer settles, the interface reveals target value, fly estimate, and absolute error.

A `New problem` control generates another deterministic problem. A seed is shown so examples can be reproduced for recording and testing.

## Function family

For the first benchmark, use low-degree polynomials

`f(x) = a0 + a1 x + a2 x^2 + a3 x^3`

with bounded random coefficients and a bounded interval `[a,b]`. The target integral is computed analytically from the polynomial coefficients. This keeps the benchmark exact and makes failures attributable to the agent rather than numerical quadrature.

## Architecture

- `index.html` — page structure and semantic regions.
- `styles.css` — light visual design, responsive layout, fly animation, slider/result presentation.
- `src/math.js` — seeded RNG, polynomial generation/evaluation, exact integral.
- `src/agent.js` — baseline agent interface returning an estimate and optional animation trace.
- `src/app.js` — state transitions, rendering, round lifecycle, slider animation.
- `tests/math.test.mjs` — deterministic tests for polynomial evaluation and integration.
- `tests/agent.test.mjs` — contract tests for estimate bounds and deterministic seeded behavior.

The agent boundary is intentionally small:

```js
estimate(problem, options) -> { value, confidence?, trace? }
```

A future MaleCNS implementation can satisfy the same contract without changing the UI.

## Baseline agent

The baseline should produce a plausible imperfect estimate by starting from the exact target and applying deterministic seed-dependent error. It exists only to validate the interaction and rendering pipeline. The interface must label it `baseline` so the demo cannot be mistaken for a connectome result.

## Visual direction

Use a light background, dark text, restrained accent colors, and a prominent illustrated/SVG fly. The composition should read left-to-right as:

`graph -> fly -> answer slider`.

The fly should visibly react during inference through wing/body/eye motion or an activity halo. Avoid question text or decorative educational copy; the visual should communicate the task by showing the graph, fly, slider, and measured result.

## Error handling

Generated problems must keep the true integral inside the slider range with margin. If generation fails validation, regenerate from the same RNG stream. UI code should fail visibly in the result panel rather than silently.

## Testing

Use Node's built-in test runner so the project remains dependency-free. Tests cover:

1. seeded generation reproducibility;
2. exact polynomial integration against hand-computed cases;
3. generated targets staying inside configured bounds;
4. deterministic baseline estimates for a seed;
5. agent output remaining inside the answer range.

A manual browser smoke check verifies the graph, fly animation, slider motion, new-problem flow, and mobile layout.

## Non-goals for this milestone

- downloading or simulating the full MaleCNS connectome;
- training synaptic weights;
- symbolic antiderivatives;
- backend services;
- framework or bundler setup;
- production analytics or deployment automation.

## Next milestone

Add a `ConnectomeAgent` adapter backed by a published/open MaleCNS-compatible simulation, preserving the same problem and rendering interfaces. That milestone must clearly distinguish trainable interface parameters from measured connectome topology and include an ordinary neural-network baseline for comparison.
