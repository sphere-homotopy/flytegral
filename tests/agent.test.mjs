import test from 'node:test';
import assert from 'node:assert/strict';

import { BaselineAgent } from '../src/agent.js';
import { generateProblem } from '../src/math.js';

test('baseline agent is deterministic for a problem and seed', () => {
  const agent = new BaselineAgent();
  const problem = generateProblem({ seed: 12345 });

  const first = agent.estimate(problem, { seed: 99 });
  const second = agent.estimate(problem, { seed: 99 });

  assert.deepEqual(first, second);
  assert.equal(first.kind, 'baseline');
});

test('baseline answer remains inside the configured numerical range', () => {
  const agent = new BaselineAgent();

  for (let seed = 0; seed < 100; seed += 1) {
    const problem = generateProblem({ seed });
    const result = agent.estimate(problem, { seed: seed + 1000 });
    assert.ok(result.value >= problem.answerRange[0]);
    assert.ok(result.value <= problem.answerRange[1]);
    assert.ok(result.confidence >= 0 && result.confidence <= 1);
  }
});

test('baseline exposes a bounded slider trace ending at its estimate', () => {
  const agent = new BaselineAgent();
  const problem = generateProblem({ seed: 77 });
  const result = agent.estimate(problem, { seed: 88 });

  assert.ok(result.trace.length >= 3);
  assert.ok(result.trace.every((position) => position >= 0 && position <= 1));
  assert.equal(result.trace.at(-1), result.sliderPosition);
});
