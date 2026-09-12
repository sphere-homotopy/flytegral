export const DEMO_RECORDING_PROBLEM_COUNT = 10;
export const DEMO_RESULT_HOLD_MS = 1700;

export function recordingProblemSeeds(firstSeed, count = DEMO_RECORDING_PROBLEM_COUNT) {
  if (!Number.isInteger(firstSeed)) throw new TypeError('firstSeed must be an integer');
  if (!Number.isInteger(count) || count < 1) throw new RangeError('count must be a positive integer');
  return Array.from({ length: count }, (_, index) => firstSeed + index);
}
