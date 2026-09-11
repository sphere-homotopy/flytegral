import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { baselineAgent } from '../src/agent.js';
import { generateProblem } from '../src/math.js';
import { rewardForAnswer } from '../src/environment.js';

const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const css = await readFile(new URL('../styles.css', import.meta.url), 'utf8');
const problem = generateProblem({ seed: 260912 });
const result = baselineAgent.estimate(problem, { seed: 260912 });

for (const id of ['graph-canvas', 'fly-agent', 'answer-slider', 'target-value', 'prediction-value', 'error-value']) {
  assert.match(html, new RegExp(`id=["']${id}["']`));
}
assert.match(css, /fly-agent\.thinking/);
assert.ok(Number.isFinite(problem.target));
assert.ok(Number.isFinite(result.value));
assert.ok(result.sliderPosition >= 0 && result.sliderPosition <= 1);
assert.ok(rewardForAnswer(problem, result.value) <= 0);

console.log('Flytegral smoke OK');
console.log({
  seed: problem.seed,
  interval: problem.interval,
  target: Number(problem.target.toFixed(4)),
  baseline: Number(result.value.toFixed(4)),
  absoluteError: Number(Math.abs(result.value - problem.target).toFixed(4)),
});
