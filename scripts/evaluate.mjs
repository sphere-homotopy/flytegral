import { BaselineAgent } from '../src/agent.js';
import { makeSeedSplit, problemsFromSeeds, evaluateAgent, ZeroAgent, MidpointAgent } from '../src/evaluation.js';
import { trainLinearGraphAgent } from '../src/training.js';

const split = makeSeedSplit({ baseSeed: 20260912, trainCount: 256, testCount: 128 });
const trainProblems = problemsFromSeeds(split.train);
const testProblems = problemsFromSeeds(split.test);
const agents = [
  new ZeroAgent(),
  new MidpointAgent(),
  new BaselineAgent(),
  trainLinearGraphAgent(trainProblems),
];

const rows = agents.map((agent) => {
  const metrics = evaluateAgent(agent, testProblems);
  return {
    agent: agent.kind,
    MAE: metrics.mae.toFixed(4),
    'mean relative error': `${(metrics.meanRelativeError * 100).toFixed(2)}%`,
    'relative n': metrics.relativeCount,
  };
});

console.log(`Flytegral deterministic benchmark — ${trainProblems.length} train / ${testProblems.length} test`);
console.table(rows);
