import test from 'node:test';
import assert from 'node:assert/strict';

import { generateProblem } from '../src/math.js';
import { GraphRasterSensorEncoder, MaleCNSBridgeAgent } from '../src/malecns.js';

test('graph raster sensor is deterministic, bounded, and contains visible ink', () => {
  const problem = generateProblem({ seed: 260912 });
  const encoder = new GraphRasterSensorEncoder({ width: 48, height: 32 });
  const first = encoder.encode(problem);
  const second = encoder.encode(problem);

  assert.equal(first.width, 48);
  assert.equal(first.height, 32);
  assert.equal(first.luminance.length, 48 * 32);
  assert.deepEqual(first, second);
  assert.ok(first.luminance.every((value) => value >= 0 && value <= 1));
  assert.ok(first.luminance.some((value) => value < 0.4));
  assert.ok(first.luminance.some((value) => value > 0.9));
});

test('MaleCNS bridge posts visual stimulus and converts response into agent contract', async () => {
  const problem = generateProblem({ seed: 42 });
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options, body: JSON.parse(options.body) };
    return {
      ok: true,
      async json() {
        return {
          sliderPosition: 0.73,
          confidence: 0.61,
          trace: [0.5, 0.58, 0.73],
          telemetry: { totalSpikes: 1234, simMs: 80 },
        };
      },
    };
  };

  const agent = new MaleCNSBridgeAgent({ endpoint: 'http://127.0.0.1:8777', fetchImpl });
  const result = await agent.estimate(problem, { seed: 42 });

  assert.equal(request.url, 'http://127.0.0.1:8777/estimate');
  assert.equal(request.options.method, 'POST');
  assert.equal(request.body.seed, 42);
  assert.equal(request.body.answerRange.length, 2);
  assert.equal(request.body.stimulus.width, 64);
  assert.equal(request.body.stimulus.height, 40);
  assert.equal(request.body.stimulus.luminance.length, 64 * 40);
  assert.equal(result.kind, 'malecns-v1');
  assert.equal(result.sliderPosition, 0.73);
  assert.ok(result.value >= problem.answerRange[0] && result.value <= problem.answerRange[1]);
  assert.deepEqual(result.telemetry, { totalSpikes: 1234, simMs: 80 });
});

test('MaleCNS bridge does not silently fall back when the runtime is unavailable', async () => {
  const problem = generateProblem({ seed: 7 });
  const agent = new MaleCNSBridgeAgent({
    fetchImpl: async () => { throw new TypeError('fetch failed'); },
  });

  await assert.rejects(
    () => agent.estimate(problem),
    /MaleCNS runtime unavailable/,
  );
});
