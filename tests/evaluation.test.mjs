import test from 'node:test';
import assert from 'node:assert/strict';

import { makeSeedSplit, computeMetrics, evaluateAgent } from '../src/evaluation.js';
import { LinearGraphAgent, trainLinearGraphAgent } from '../src/training.js';
import { generateProblem } from '../src/math.js';

test('train/test split is deterministic and disjoint', () => {
  const first = makeSeedSplit({ baseSeed: 9000, trainCount: 40, testCount: 20 });
  const second = makeSeedSplit({ baseSeed: 9000, trainCount: 40, testCount: 20 });

  assert.deepEqual(first, second);
  assert.equal(first.train.length, 40);
  assert.equal(first.test.length, 20);
  assert.equal(new Set([...first.train, ...first.test]).size, 60);
});

test('metrics report MAE and skip unstable relative errors near zero', () => {
  const metrics = computeMetrics([
    { target: 2, prediction: 1 },
    { target: -4, prediction: -2 },
    { target: 0, prediction: 3 },
  ]);

  assert.equal(metrics.mae, 2);
  assert.equal(metrics.relativeCount, 2);
  assert.ok(Math.abs(metrics.meanRelativeError - 0.5) < 1e-12);
});

test('trainable graph agent learns a strong cubic quadrature baseline', () => {
  const train = Array.from({ length: 80 }, (_, i) => generateProblem({ seed: 1000 + i }));
  const testProblems = Array.from({ length: 40 }, (_, i) => generateProblem({ seed: 5000 + i }));
  const agent = trainLinearGraphAgent(train, { ridge: 1e-9 });

  assert.ok(agent instanceof LinearGraphAgent);
  const metrics = evaluateAgent(agent, testProblems);
  assert.ok(metrics.mae < 0.03, `expected MAE < 0.03, got ${metrics.mae}`);
});
