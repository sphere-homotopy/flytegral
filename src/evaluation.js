import { answerToSlider, clamp, createRng, evaluatePolynomial, generateProblem } from './math.js';

export function makeSeedSplit({ baseSeed = 20260912, trainCount = 256, testCount = 128 } = {}) {
  const total = trainCount + testCount;
  const seeds = Array.from({ length: total }, (_, index) => baseSeed + index);
  const rng = createRng(`${baseSeed}:split`);

  for (let index = seeds.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    [seeds[index], seeds[swapIndex]] = [seeds[swapIndex], seeds[index]];
  }

  return {
    train: seeds.slice(0, trainCount),
    test: seeds.slice(trainCount),
  };
}

export function problemsFromSeeds(seeds) {
  return seeds.map((seed) => generateProblem({ seed }));
}

export function computeMetrics(rows, { relativeEpsilon = 0.1 } = {}) {
  if (!rows.length) return { mae: NaN, meanRelativeError: NaN, relativeCount: 0, count: 0 };

  let absoluteErrorSum = 0;
  let relativeErrorSum = 0;
  let relativeCount = 0;

  for (const { target, prediction } of rows) {
    const error = Math.abs(prediction - target);
    absoluteErrorSum += error;
    if (Math.abs(target) >= relativeEpsilon) {
      relativeErrorSum += error / Math.abs(target);
      relativeCount += 1;
    }
  }

  return {
    mae: absoluteErrorSum / rows.length,
    meanRelativeError: relativeCount ? relativeErrorSum / relativeCount : NaN,
    relativeCount,
    count: rows.length,
  };
}

export function evaluateAgent(agent, problems) {
  const rows = problems.map((problem) => ({
    target: problem.target,
    prediction: agent.estimate(problem).value,
  }));
  return { ...computeMetrics(rows), rows };
}

export class ZeroAgent {
  constructor() { this.kind = 'zero'; }
  estimate(problem) {
    const value = clamp(0, problem.answerRange[0], problem.answerRange[1]);
    return { kind: this.kind, value, sliderPosition: answerToSlider(value, problem.answerRange) };
  }
}

export class MidpointAgent {
  constructor() { this.kind = 'midpoint'; }
  estimate(problem) {
    const [a, b] = problem.interval;
    const midpoint = (a + b) / 2;
    const rawValue = evaluatePolynomial(problem.coefficients, midpoint) * (b - a);
    const value = clamp(rawValue, problem.answerRange[0], problem.answerRange[1]);
    return { kind: this.kind, value, sliderPosition: answerToSlider(value, problem.answerRange) };
  }
}
