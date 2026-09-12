import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { DEMO_RECORDING_PROBLEM_COUNT, DEMO_RESULT_HOLD_MS, recordingProblemSeeds } from '../src/recording.js';

test('demo recording covers ten consecutive integration problems', () => {
  assert.equal(DEMO_RECORDING_PROBLEM_COUNT, 10);
  assert.deepEqual(recordingProblemSeeds(260913), [
    260913, 260914, 260915, 260916, 260917,
    260918, 260919, 260920, 260921, 260922,
  ]);
});

test('recording adds one extra second to the previous 700 ms result hold', async () => {
  assert.equal(DEMO_RESULT_HOLD_MS, 1700);
  const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  assert.match(app, /DEMO_RESULT_HOLD_MS/);
  assert.match(app, /setTimeout\(resolve, DEMO_RESULT_HOLD_MS\)/);
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

test('X recording chooses H.264 with AAC-LC and never generic Opus-prone MP4', async () => {
  const recording = await import('../src/recording.js');
  assert.ok(Array.isArray(recording.X_UPLOAD_MIME_CANDIDATES));
  assert.match(recording.X_UPLOAD_MIME_CANDIDATES[0], /video\/mp4/);
  assert.match(recording.X_UPLOAD_MIME_CANDIDATES[0], /avc1/);
  assert.match(recording.X_UPLOAD_MIME_CANDIDATES[0], /mp4a\.40\.2/);
  assert.doesNotMatch(recording.X_UPLOAD_MIME_CANDIDATES.join('\n'), /^video\/mp4$/m);

  const supported = new Set(['video/mp4;codecs=avc1,mp4a.40.2']);
  assert.equal(
    recording.chooseXRecordingMimeType((candidate) => supported.has(candidate)),
    'video/mp4;codecs=avc1,mp4a.40.2',
  );
});

test('MediaRecorder requests 128 kbps AAC audio for X-compatible output', async () => {
  const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  assert.match(app, /audioBitsPerSecond:\s*128_000/);
  assert.match(app, /chooseXRecordingMimeType\(MediaRecorder\.isTypeSupported\.bind\(MediaRecorder\)\)/);
});
