import { answerToSlider, clamp, createRng } from './math.js';

export class BaselineAgent {
  constructor({ errorFraction = 0.08 } = {}) {
    this.errorFraction = errorFraction;
    this.kind = 'baseline';
  }

  estimate(problem, { seed = problem.seed } = {}) {
    const rng = createRng(`${seed}:baseline`);
    const [min, max] = problem.answerRange;
    const width = max - min;

    // Triangular noise gives frequent near-misses with occasional larger errors.
    const noise = (rng() + rng() - 1) * width * this.errorFraction;
    const value = clamp(problem.target + noise, min, max);
    const sliderPosition = answerToSlider(value, problem.answerRange);
    const normalizedError = Math.abs(value - problem.target) / width;
    const confidence = clamp(1 - normalizedError * 4, 0, 1);

    const wobbleA = clamp(0.5 + (rng() - 0.5) * 0.28, 0, 1);
    const wobbleB = clamp(sliderPosition + (rng() - 0.5) * 0.18, 0, 1);

    return {
      kind: this.kind,
      value,
      sliderPosition,
      confidence,
      trace: [0.5, wobbleA, wobbleB, sliderPosition],
    };
  }
}

export const baselineAgent = new BaselineAgent();
