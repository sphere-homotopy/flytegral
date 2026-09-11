import test from 'node:test';
import assert from 'node:assert/strict';

import {
  createRng,
  evaluatePolynomial,
  integratePolynomial,
  generateProblem,
  answerToSlider,
  sliderToAnswer,
} from '../src/math.js';

test('evaluates a cubic polynomial', () => {
  assert.equal(evaluatePolynomial([1, 2, 3, 4], 2), 49);
});

test('integrates a cubic polynomial analytically', () => {
  // Integral of 1 + 2x + 3x^2 + 4x^3 from 0 to 2 = 30.
  assert.equal(integratePolynomial([1, 2, 3, 4], 0, 2), 30);
  // Integral of x^2 from -1 to 2 = 3.
  assert.equal(integratePolynomial([0, 0, 1, 0], -1, 2), 3);
});

test('seeded RNG is deterministic and seed-sensitive', () => {
  const first = createRng(42);
  const second = createRng(42);
  const third = createRng(43);

  const a = Array.from({ length: 6 }, () => first());
  const b = Array.from({ length: 6 }, () => second());
  const c = Array.from({ length: 6 }, () => third());

  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
});

test('problem generation is deterministic and keeps target inside answer range with margin', () => {
  const a = generateProblem({ seed: 20260912 });
  const b = generateProblem({ seed: 20260912 });

  assert.deepEqual(a, b);
  assert.equal(a.coefficients.length, 4);
  assert.ok(a.interval[0] < a.interval[1]);
  assert.ok(a.answerRange[0] < a.target);
  assert.ok(a.target < a.answerRange[1]);

  const width = a.answerRange[1] - a.answerRange[0];
  assert.ok(a.target - a.answerRange[0] >= width * 0.12);
  assert.ok(a.answerRange[1] - a.target >= width * 0.12);
});

test('slider mapping round-trips numerical answers', () => {
  const range = [-7.5, 12.5];
  for (const value of [-7.5, -2, 0, 8.25, 12.5]) {
    const slider = answerToSlider(value, range);
    const reconstructed = sliderToAnswer(slider, range);
    assert.ok(Math.abs(reconstructed - value) < 1e-12);
  }

  assert.equal(answerToSlider(-100, range), 0);
  assert.equal(answerToSlider(100, range), 1);
  assert.equal(sliderToAnswer(-1, range), -7.5);
  assert.equal(sliderToAnswer(2, range), 12.5);
});
