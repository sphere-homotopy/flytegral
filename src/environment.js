import { evaluatePolynomial } from './math.js';

export function graphSampleObservation(problem, { sampleCount = 5 } = {}) {
  if (!Number.isInteger(sampleCount) || sampleCount < 2) {
    throw new RangeError('sampleCount must be an integer >= 2');
  }

  const [a, b] = problem.interval;
  const width = b - a;
  const samples = Array.from({ length: sampleCount }, (_, index) => {
    const t = index / (sampleCount - 1);
    const x = a + t * width;
    return { x, y: evaluatePolynomial(problem.coefficients, x) };
  });

  return {
    interval: [a, b],
    width,
    samples,
    answerRange: [...problem.answerRange],
  };
}

export function graphFeatureVector(problem, options = {}) {
  const observation = graphSampleObservation(problem, options);
  return [1, ...observation.samples.map(({ y }) => observation.width * y)];
}

export function rewardForAnswer(problem, value) {
  const width = problem.answerRange[1] - problem.answerRange[0];
  if (!(width > 0)) throw new RangeError('answer range must have positive width');
  const error = Math.abs(value - problem.target);
  return error === 0 ? 0 : -error / width;
}
