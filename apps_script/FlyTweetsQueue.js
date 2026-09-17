'use strict';

function flyTweetsIdempotencyKey(batchId, tweetIndex) {
  const batch = String(batchId || '').trim();
  const index = Number(tweetIndex);
  if (!batch) throw new Error('batchId is required');
  if (!Number.isInteger(index) || index < 0) {
    throw new Error('tweetIndex must be a non-negative integer');
  }
  return `${batch}:${index}`;
}

function scheduleFlyTweetSlots(publishAtIsoStrings, seed, options) {
  void seed;
  if (!Array.isArray(publishAtIsoStrings) || publishAtIsoStrings.length === 0) {
    throw new Error('at least one fly-selected publish time is required');
  }
  const config = Object.assign({
    minGapMinutes: 20,
    maxPostsPer24h: 30,
  }, options || {});

  if (!Number.isFinite(config.minGapMinutes) || config.minGapMinutes <= 0) {
    throw new Error('minGapMinutes must be a positive finite number');
  }
  if (!Number.isInteger(config.maxPostsPer24h) || config.maxPostsPer24h <= 0) {
    throw new Error('maxPostsPer24h must be a positive integer');
  }

  const millis = publishAtIsoStrings.map((value, index) => {
    const parsed = Date.parse(value);
    if (!Number.isFinite(parsed)) throw new Error(`invalid publish time at index ${index}`);
    return parsed;
  });

  const minimumGapMillis = config.minGapMinutes * 60_000;
  for (let i = 1; i < millis.length; i += 1) {
    if (millis[i] <= millis[i - 1]) {
      throw new Error('fly-selected publish times must be strictly increasing');
    }
    if (millis[i] - millis[i - 1] < minimumGapMillis) {
      throw new Error('fly-selected publish times violate the minimum gap safety bound');
    }
  }

  const dayMillis = 24 * 60 * 60 * 1000;
  let left = 0;
  for (let right = 0; right < millis.length; right += 1) {
    while (millis[right] - millis[left] >= dayMillis) left += 1;
    if (right - left + 1 > config.maxPostsPer24h) {
      throw new Error('fly-selected publish times exceed the rolling 24h safety cap');
    }
  }

  // Transport is intentionally cadence-neutral: the fly chooses absolute times.
  return publishAtIsoStrings.slice();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    flyTweetsIdempotencyKey,
    scheduleFlyTweetSlots,
  };
}
