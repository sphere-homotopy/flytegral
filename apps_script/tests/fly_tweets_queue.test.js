const test = require('node:test');
const assert = require('node:assert/strict');

const {
  flyTweetsIdempotencyKey,
  scheduleFlyTweetSlots,
} = require('../FlyTweetsQueue');

const FLY_SELECTED_SLOTS = [
  '2026-09-17T08:17:00.000Z',
  '2026-09-17T12:41:00.000Z',
  '2026-09-17T19:03:00.000Z',
];


test('transport preserves fly-selected absolute publish times exactly', () => {
  const scheduled = scheduleFlyTweetSlots(FLY_SELECTED_SLOTS, 260917);

  assert.deepEqual(scheduled, FLY_SELECTED_SLOTS);
});


test('transport accepts a variable number of fly-selected slots', () => {
  assert.equal(scheduleFlyTweetSlots(FLY_SELECTED_SLOTS.slice(0, 1), 1).length, 1);
  assert.equal(scheduleFlyTweetSlots(FLY_SELECTED_SLOTS, 1).length, 3);
});


test('transport safety guard rejects unsorted or too-close fly-selected slots', () => {
  assert.throws(
    () => scheduleFlyTweetSlots([
      '2026-09-17T10:00:00.000Z',
      '2026-09-17T09:00:00.000Z',
    ], 1),
    /strictly increasing/,
  );
  assert.throws(
    () => scheduleFlyTweetSlots([
      '2026-09-17T10:00:00.000Z',
      '2026-09-17T10:10:00.000Z',
    ], 1, { minGapMinutes: 20 }),
    /minimum gap/,
  );
});


test('idempotency key supports variable batch sizes', () => {
  assert.equal(flyTweetsIdempotencyKey('batch-a', 0), 'batch-a:0');
  assert.equal(flyTweetsIdempotencyKey('batch-a', 42), 'batch-a:42');
  assert.throws(() => flyTweetsIdempotencyKey('batch-a', -1), /non-negative/);
});
