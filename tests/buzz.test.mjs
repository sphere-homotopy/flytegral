import test from 'node:test';
import assert from 'node:assert/strict';
import { ThinkingBuzz, BUZZ_AUDIO_URL, BUZZ_START_SECONDS } from '../src/buzz.js';

function fakeAudio() {
  return {
    loop: false,
    volume: 1,
    preload: '',
    currentTime: 0,
    playCalls: 0,
    pauseCalls: 0,
    play() { this.playCalls += 1; return Promise.resolve(); },
    pause() { this.pauseCalls += 1; },
  };
}

test('thinking buzz uses the public-domain Wikimedia recording and loops quietly', () => {
  const audio = fakeAudio();
  const buzz = new ThinkingBuzz({ audioFactory: () => audio });

  assert.match(BUZZ_AUDIO_URL, /upload\.wikimedia\.org\/.*Bombus_buzz\.ogg/);
  assert.equal(BUZZ_START_SECONDS, 1);
  assert.equal(audio.loop, true);
  assert.equal(audio.preload, 'auto');
  assert.ok(audio.volume > 0 && audio.volume <= 0.25);
  assert.equal(buzz.audio, audio);
});

test('thinking buzz skips the first second whenever thinking starts', async () => {
  const audio = fakeAudio();
  const buzz = new ThinkingBuzz({ audioFactory: () => audio });

  await buzz.start();
  assert.equal(audio.currentTime, 1);
  assert.equal(audio.playCalls, 1);

  audio.currentTime = 3;
  buzz.stop();
  assert.equal(audio.pauseCalls, 1);
  assert.equal(audio.currentTime, 1);
});
