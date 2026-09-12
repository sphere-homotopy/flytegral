export const DEMO_RECORDING_PROBLEM_COUNT = 10;
export const DEMO_RESULT_HOLD_MS = 1700;
export const X_UPLOAD_MIME_CANDIDATES = [
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4;codecs=avc1,mp4a.40.2',
];

export function chooseXRecordingMimeType(isTypeSupported) {
  if (typeof isTypeSupported !== 'function') throw new TypeError('isTypeSupported must be a function');
  return X_UPLOAD_MIME_CANDIDATES.find((candidate) => isTypeSupported(candidate)) ?? '';
}

export function recordingProblemSeeds(firstSeed, count = DEMO_RECORDING_PROBLEM_COUNT) {
  if (!Number.isInteger(firstSeed)) throw new TypeError('firstSeed must be an integer');
  if (!Number.isInteger(count) || count < 1) throw new RangeError('count must be a positive integer');
  return Array.from({ length: count }, (_, index) => firstSeed + index);
}
