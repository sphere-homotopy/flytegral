import test from 'node:test';
import assert from 'node:assert/strict';

import { generateProblem } from '../src/math.js';
import { rewardForAnswer } from '../src/environment.js';
import { GraphSensorEncoder, SliderMotorDecoder, ConnectomeAgentAdapter } from '../src/brain.js';

test('environment reward is zero at target and decreases with normalized absolute error', () => {
  const problem = generateProblem({ seed: 31415 });
  const width = problem.answerRange[1] - problem.answerRange[0];

  assert.equal(rewardForAnswer(problem, problem.target), 0);
  assert.ok(Math.abs(rewardForAnswer(problem, problem.target + width * 0.1) + 0.1) < 1e-12);
});

test('sensor encoder is deterministic and separate from motor decoding', () => {
  const problem = generateProblem({ seed: 2718 });
  const sensor = new GraphSensorEncoder({ sampleCount: 9 });
  const motor = new SliderMotorDecoder();

  const first = sensor.encode(problem);
  const second = sensor.encode(problem);
  assert.deepEqual([...first], [...second]);
  assert.equal(first.length, 10);

  const decoded = motor.decode(0.75, problem);
  const expected = problem.answerRange[0] + 0.75 * (problem.answerRange[1] - problem.answerRange[0]);
  assert.ok(Math.abs(decoded.value - expected) < 1e-12);
});

test('connectome adapter accepts an arbitrary network without environment coupling', () => {
  const problem = generateProblem({ seed: 42 });
  const network = {
    run(input) {
      assert.ok(input instanceof Float32Array);
      return 0.25;
    },
  };
  const agent = new ConnectomeAgentAdapter({ network });
  const result = agent.estimate(problem);

  assert.equal(result.kind, 'connectome-adapter');
  assert.equal(result.sliderPosition, 0.25);
  assert.ok(result.value >= problem.answerRange[0] && result.value <= problem.answerRange[1]);
});
