'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { buildBufferCreatePostPayload } = require('../FlyTweetsBuffer');


test('Buffer payload preserves fly-selected absolute time', () => {
  const payload = buildBufferCreatePostPayload({
    text: 'bzz topology',
    scheduled_at: '2026-09-18T11:37:00+00:00',
    idempotency_key: 'batch-a:0',
  }, 'channel-123');

  assert.equal(payload.variables.input.text, 'bzz topology');
  assert.equal(payload.variables.input.channelId, 'channel-123');
  assert.equal(payload.variables.input.schedulingType, 'automatic');
  assert.equal(payload.variables.input.mode, 'customScheduled');
  assert.equal(payload.variables.input.dueAt, '2026-09-18T11:37:00.000Z');
  assert.equal(payload.variables.input.source, 'flytegral:batch-a:0');
  assert.match(payload.query, /createPost/);
});


test('Buffer payload rejects missing fly schedule', () => {
  assert.throws(
    () => buildBufferCreatePostPayload({ text: 'bzz', idempotency_key: 'a:0' }, 'channel-123'),
    /scheduled_at/,
  );
});
