import test from 'node:test';
import assert from 'node:assert/strict';
import { DEMO_RECORDING_PROBLEM_COUNT, recordingProblemSeeds } from '../src/recording.js';
import { readFile } from 'node:fs/promises';

test('demo recording covers ten consecutive integration problems', () => {
  assert.equal(DEMO_RECORDING_PROBLEM_COUNT, 10);
  assert.deepEqual(recordingProblemSeeds(260913), [
    260913, 260914, 260915, 260916, 260917,
    260918, 260919, 260920, 260921, 260922,
  ]);
});

test('recordDemo iterates the recording seed sequence before stopping MediaRecorder', async () => {
  const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const loopAt = app.indexOf('for (const [index, seed] of seeds.entries())');
  const stopAt = app.indexOf('recorder.stop();');
  assert.ok(loopAt >= 0, 'recording loop must exist');
  assert.ok(stopAt > loopAt, 'recorder must stop only after all recording problems');
  assert.match(app, /recordingProblemSeeds\(currentSeed \+ 1\)/);
  assert.match(app, /recording \$\{index \+ 1\}\/\$\{seeds\.length\}/);
});
