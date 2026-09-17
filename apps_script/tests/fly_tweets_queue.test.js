const test = require('node:test');
const assert = require('node:assert/strict');

const {
  flyTweetsIdempotencyKey,
  scheduleFlyTweetSlots,
} = require('../FlyTweetsQueue');

const BASELINES = [
  '2026-09-17T07:30:00.000Z',
  '2026-09-17T09:00:00.000Z',
  '2026-09-17T10:30:00.000Z',
  '2026-09-17T12:00:00.000Z',
  '2026-09-17T13:30:00.000Z',
  '2026-09-17T15:00:00.000Z',
  '2026-09-17T16:30:00.000Z',
  '2026-09-17T18:00:00.000Z',
  '2026-09-17T19:30:00.000Z',
  '2026-09-17T21:00:00.000Z',
];


test('schedule is deterministic for a seed and returns ten sorted slots', () => {
  const first = scheduleFlyTweetSlots(BASELINES, 260917);
  const second = scheduleFlyTweetSlots(BASELINES, 260917);

  assert.deepEqual(first, second);
  assert.equal(first.length, 10);
  assert.deepEqual([...first].sort(), first);
});


test('all scheduled slots stay within the 45 minute baseline clip', () => {
  const slots = scheduleFlyTweetSlots(BASELINES, 1234, {
    sigmaMinutes: 22,
    clipMinutes: 45,
    minGapMinutes: 35,
  });

  for (let i = 0; i < slots.length; i += 1) {
    const deltaMinutes = Math.abs(
      (Date.parse(slots[i]) - Date.parse(BASELINES[i])) / 60_000,
    );
    assert.ok(deltaMinutes <= 45 + 1e-9, `slot ${i} moved ${deltaMinutes} minutes`);
  }
});


test('scheduled slots preserve at least a 35 minute gap', () => {
  const slots = scheduleFlyTweetSlots(BASELINES, 987654);

  for (let i = 1; i < slots.length; i += 1) {
    const gapMinutes = (Date.parse(slots[i]) - Date.parse(slots[i - 1])) / 60_000;
    assert.ok(gapMinutes >= 35, `gap ${i - 1}->${i} is ${gapMinutes} minutes`);
  }
});


test('different seeds change at least one slot', () => {
  const first = scheduleFlyTweetSlots(BASELINES, 1);
  const second = scheduleFlyTweetSlots(BASELINES, 2);
  assert.notDeepEqual(first, second);
});


test('idempotency key is batch plus tweet index', () => {
  assert.equal(flyTweetsIdempotencyKey('batch-a', 0), 'batch-a:0');
  assert.equal(flyTweetsIdempotencyKey('batch-a', 9), 'batch-a:9');
});


test('scheduler rejects anything other than ten baselines', () => {
  assert.throws(
    () => scheduleFlyTweetSlots(BASELINES.slice(0, 9), 1),
    /exactly 10/,
  );
});
