const UINT32_MAX_PLUS_ONE = 0x1_0000_0000;

function toSeed(seed) {
  if (Number.isInteger(seed)) return seed >>> 0;

  const text = String(seed);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function createRng(seed) {
  let state = toSeed(seed);
  return function random() {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / UINT32_MAX_PLUS_ONE;
  };
}

export function evaluatePolynomial(coefficients, x) {
  return coefficients.reduceRight((acc, coefficient) => acc * x + coefficient, 0);
}

export function integratePolynomial(coefficients, a, b) {
  return coefficients.reduce((total, coefficient, degree) => {
    const exponent = degree + 1;
    return total + (coefficient / exponent) * (b ** exponent - a ** exponent);
  }, 0);
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function answerToSlider(value, answerRange) {
  const [min, max] = answerRange;
  if (!(max > min)) throw new RangeError('answer range must have positive width');
  return clamp((value - min) / (max - min), 0, 1);
}

export function sliderToAnswer(position, answerRange) {
  const [min, max] = answerRange;
  if (!(max > min)) throw new RangeError('answer range must have positive width');
  return min + clamp(position, 0, 1) * (max - min);
}

function choose(rng, values) {
  return values[Math.floor(rng() * values.length)];
}

function quantizedCoefficient(rng, maxAbs = 2.25, step = 0.25) {
  const units = Math.round(maxAbs / step);
  return (Math.floor(rng() * (units * 2 + 1)) - units) * step;
}

function makeAnswerRange(target) {
  const rawRadius = Math.max(4, Math.abs(target) * 1.75 + 1.5);
  const radius = Math.ceil(rawRadius * 2) / 2;
  return [-radius, radius];
}

export function generateProblem({ seed, coefficientMaxAbs = 2.25 } = {}) {
  if (seed === undefined || seed === null) {
    throw new TypeError('generateProblem requires an explicit seed');
  }

  const rng = createRng(seed);
  const coefficients = Array.from({ length: 4 }, () => quantizedCoefficient(rng, coefficientMaxAbs));

  // Keep the visual interesting: avoid a completely flat function.
  if (coefficients.slice(1).every((coefficient) => coefficient === 0)) {
    coefficients[1 + Math.floor(rng() * 3)] = choose(rng, [-1, -0.5, 0.5, 1]);
  }

  const endpoints = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
  const startIndex = Math.floor(rng() * (endpoints.length - 2));
  const possibleEndIndexes = [];
  for (let index = startIndex + 2; index < endpoints.length; index += 1) {
    possibleEndIndexes.push(index);
  }
  const endIndex = choose(rng, possibleEndIndexes);
  const interval = [endpoints[startIndex], endpoints[endIndex]];
  const target = integratePolynomial(coefficients, interval[0], interval[1]);

  return {
    seed,
    coefficients,
    interval,
    target,
    answerRange: makeAnswerRange(target),
    graphDomain: [-2.5, 2.5],
  };
}
