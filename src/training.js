import { answerToSlider, clamp } from './math.js';
import { graphFeatureVector } from './environment.js';

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]]);

  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];

    const divisor = a[column][column];
    if (Math.abs(divisor) < 1e-12) throw new Error('linear system is singular');
    for (let j = column; j <= n; j += 1) a[column][j] /= divisor;

    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = a[row][column];
      for (let j = column; j <= n; j += 1) a[row][j] -= factor * a[column][j];
    }
  }

  return a.map((row) => row[n]);
}

export class LinearGraphAgent {
  constructor(weights, { sampleCount = 5 } = {}) {
    this.weights = [...weights];
    this.sampleCount = sampleCount;
    this.kind = 'linear-graph';
  }

  estimate(problem) {
    const features = graphFeatureVector(problem, { sampleCount: this.sampleCount });
    const rawValue = features.reduce((sum, value, index) => sum + value * this.weights[index], 0);
    const value = clamp(rawValue, problem.answerRange[0], problem.answerRange[1]);
    const sliderPosition = answerToSlider(value, problem.answerRange);

    return {
      kind: this.kind,
      value,
      sliderPosition,
      confidence: 0.5,
      trace: [0.5, sliderPosition],
    };
  }
}

export function trainLinearGraphAgent(problems, { sampleCount = 5, ridge = 1e-8 } = {}) {
  if (!Array.isArray(problems) || problems.length === 0) {
    throw new TypeError('training requires at least one problem');
  }

  const rows = problems.map((problem) => graphFeatureVector(problem, { sampleCount }));
  const featureCount = rows[0].length;
  const xtx = Array.from({ length: featureCount }, () => Array(featureCount).fill(0));
  const xty = Array(featureCount).fill(0);

  rows.forEach((features, rowIndex) => {
    const target = problems[rowIndex].target;
    for (let i = 0; i < featureCount; i += 1) {
      xty[i] += features[i] * target;
      for (let j = 0; j < featureCount; j += 1) {
        xtx[i][j] += features[i] * features[j];
      }
    }
  });

  for (let i = 0; i < featureCount; i += 1) {
    xtx[i][i] += ridge;
  }

  return new LinearGraphAgent(solveLinearSystem(xtx, xty), { sampleCount });
}
