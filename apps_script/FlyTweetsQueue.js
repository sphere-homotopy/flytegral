'use strict';

function flyTweetsMulberry32(seed) {
  let state = Number(seed) >>> 0;
  return function random() {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function flyTweetsNormal(random) {
  let u1 = random();
  const u2 = random();
  if (u1 <= Number.EPSILON) u1 = Number.EPSILON;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function flyTweetsClamp(value, lower, upper) {
  return Math.min(upper, Math.max(lower, value));
}

function flyTweetsIdempotencyKey(batchId, tweetIndex) {
  const batch = String(batchId || '').trim();
  const index = Number(tweetIndex);
  if (!batch) throw new Error('batchId is required');
  if (!Number.isInteger(index) || index < 0 || index > 9) {
    throw new Error('tweetIndex must be an integer in [0, 9]');
  }
  return `${batch}:${index}`;
}

function scheduleFlyTweetSlots(baselineIsoStrings, seed, options) {
  if (!Array.isArray(baselineIsoStrings) || baselineIsoStrings.length !== 10) {
    throw new Error('Fly Tweets v1 requires exactly 10 baseline slots');
  }
  const config = Object.assign({
    sigmaMinutes: 22,
    clipMinutes: 45,
    minGapMinutes: 35,
  }, options || {});

  for (const key of ['sigmaMinutes', 'clipMinutes', 'minGapMinutes']) {
    if (!Number.isFinite(config[key]) || config[key] < 0) {
      throw new Error(`${key} must be a non-negative finite number`);
    }
  }
  if (config.clipMinutes <= 0 || config.minGapMinutes <= 0) {
    throw new Error('clipMinutes and minGapMinutes must be positive');
  }

  const baselines = baselineIsoStrings.map((value, index) => {
    const millis = Date.parse(value);
    if (!Number.isFinite(millis)) throw new Error(`invalid baseline slot at index ${index}`);
    return millis;
  });
  for (let i = 1; i < baselines.length; i += 1) {
    if (baselines[i] <= baselines[i - 1]) {
      throw new Error('baseline slots must be strictly increasing');
    }
  }

  const minute = 60_000;
  const clip = config.clipMinutes * minute;
  const gap = config.minGapMinutes * minute;
  const random = flyTweetsMulberry32(seed);
  const lower = baselines.map((value) => value - clip);
  const upper = baselines.map((value) => value + clip);
  const targets = baselines.map((value) => {
    const jitterMinutes = flyTweetsClamp(
      flyTweetsNormal(random) * config.sigmaMinutes,
      -config.clipMinutes,
      config.clipMinutes,
    );
    return value + jitterMinutes * minute;
  });

  const latest = new Array(baselines.length);
  latest[latest.length - 1] = upper[upper.length - 1];
  for (let i = latest.length - 2; i >= 0; i -= 1) {
    latest[i] = Math.min(upper[i], latest[i + 1] - gap);
    if (latest[i] < lower[i]) {
      throw new Error('baseline windows cannot satisfy the minimum gap constraint');
    }
  }

  const scheduled = new Array(baselines.length);
  for (let i = 0; i < scheduled.length; i += 1) {
    const earliest = i === 0 ? lower[i] : Math.max(lower[i], scheduled[i - 1] + gap);
    if (earliest > latest[i]) {
      throw new Error('baseline windows cannot satisfy the minimum gap constraint');
    }
    scheduled[i] = flyTweetsClamp(targets[i], earliest, latest[i]);
  }

  return scheduled.map((millis) => new Date(millis).toISOString());
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    flyTweetsIdempotencyKey,
    scheduleFlyTweetSlots,
  };
}
