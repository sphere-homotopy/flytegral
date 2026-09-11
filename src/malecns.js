import { evaluatePolynomial, sliderToAnswer } from './math.js';

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function graphExtents(problem) {
  const [xMin, xMax] = problem.graphDomain;
  let maxAbs = 1;
  for (let index = 0; index <= 160; index += 1) {
    const x = xMin + (index / 160) * (xMax - xMin);
    maxAbs = Math.max(maxAbs, Math.abs(evaluatePolynomial(problem.coefficients, x)));
  }
  return { xMin, xMax, yMin: -maxAbs * 1.14, yMax: maxAbs * 1.14 };
}

function drawPoint(buffer, width, height, x, y, luminance, radius = 1) {
  const px = Math.round(x);
  const py = Math.round(y);
  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      const sx = px + dx;
      const sy = py + dy;
      if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue;
      const distance = Math.hypot(dx, dy);
      if (distance > radius + 0.25) continue;
      const index = sy * width + sx;
      buffer[index] = Math.min(buffer[index], luminance + distance * 0.08);
    }
  }
}

export class GraphRasterSensorEncoder {
  constructor({ width = 64, height = 40, padding = 3 } = {}) {
    if (!Number.isInteger(width) || width < 16) throw new RangeError('width must be an integer >= 16');
    if (!Number.isInteger(height) || height < 12) throw new RangeError('height must be an integer >= 12');
    this.width = width;
    this.height = height;
    this.padding = padding;
  }

  encode(problem) {
    const { width, height, padding } = this;
    const luminance = new Array(width * height).fill(1);
    const { xMin, xMax, yMin, yMax } = graphExtents(problem);
    const innerWidth = width - padding * 2 - 1;
    const innerHeight = height - padding * 2 - 1;
    const xToPx = (x) => padding + ((x - xMin) / (xMax - xMin)) * innerWidth;
    const yToPx = (y) => padding + ((yMax - y) / (yMax - yMin)) * innerHeight;
    const zeroY = yToPx(0);
    const [a, b] = problem.interval;

    // Pale signed-area fill gives the retina the same interval cue as the browser UI.
    const left = Math.max(0, Math.ceil(xToPx(a)));
    const right = Math.min(width - 1, Math.floor(xToPx(b)));
    for (let px = left; px <= right; px += 1) {
      const x = xMin + ((px - padding) / innerWidth) * (xMax - xMin);
      const curveY = yToPx(evaluatePolynomial(problem.coefficients, x));
      const start = Math.max(0, Math.ceil(Math.min(zeroY, curveY)));
      const end = Math.min(height - 1, Math.floor(Math.max(zeroY, curveY)));
      for (let py = start; py <= end; py += 1) {
        luminance[py * width + px] = Math.min(luminance[py * width + px], 0.86);
      }
    }

    // Axes are visible but lighter than the function trace.
    const zeroRow = Math.round(zeroY);
    if (zeroRow >= 0 && zeroRow < height) {
      for (let px = padding; px < width - padding; px += 1) {
        luminance[zeroRow * width + px] = Math.min(luminance[zeroRow * width + px], 0.72);
      }
    }
    if (xMin <= 0 && xMax >= 0) {
      const zeroColumn = Math.round(xToPx(0));
      for (let py = padding; py < height - padding; py += 1) {
        luminance[py * width + zeroColumn] = Math.min(luminance[py * width + zeroColumn], 0.72);
      }
    }

    // Dense sampling prevents gaps after rasterization.
    const steps = Math.max(width * 5, 240);
    for (let index = 0; index <= steps; index += 1) {
      const x = xMin + (index / steps) * (xMax - xMin);
      drawPoint(luminance, width, height, xToPx(x), yToPx(evaluatePolynomial(problem.coefficients, x)), 0.08, 1);
    }

    // Interval boundaries are a mid-dark cue like the dashed lines in the UI.
    for (const boundary of [a, b]) {
      const px = Math.round(xToPx(boundary));
      for (let py = padding; py < height - padding; py += 2) {
        luminance[py * width + px] = Math.min(luminance[py * width + px], 0.46);
      }
    }

    return {
      width,
      height,
      luminance: luminance.map((value) => Number(clamp01(value).toFixed(4))),
    };
  }
}

export class MaleCNSBridgeAgent {
  constructor({
    endpoint = 'http://127.0.0.1:8777',
    fetchImpl = globalThis.fetch?.bind(globalThis),
    sensorEncoder = new GraphRasterSensorEncoder(),
  } = {}) {
    if (typeof fetchImpl !== 'function') throw new TypeError('MaleCNSBridgeAgent requires fetch');
    this.endpoint = endpoint.replace(/\/$/, '');
    this.fetchImpl = fetchImpl;
    this.sensorEncoder = sensorEncoder;
    this.kind = 'malecns-v1';
  }

  async estimate(problem, { seed = problem.seed } = {}) {
    const stimulus = this.sensorEncoder.encode(problem);
    let response;
    try {
      response = await this.fetchImpl(`${this.endpoint}/estimate`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          schema: 1,
          seed,
          stimulus,
          answerRange: [...problem.answerRange],
        }),
      });
    } catch (error) {
      throw new Error(`MaleCNS runtime unavailable at ${this.endpoint}: ${error.message}`);
    }

    if (!response.ok) {
      let detail = '';
      try {
        const payload = await response.json();
        detail = payload.error ? `: ${payload.error}` : '';
      } catch {
        // Keep the status-only error if the bridge did not return JSON.
      }
      throw new Error(`MaleCNS runtime returned HTTP ${response.status ?? 'error'}${detail}`);
    }

    const payload = await response.json();
    const sliderPosition = clamp01(Number(payload.sliderPosition));
    if (!Number.isFinite(sliderPosition)) throw new Error('MaleCNS runtime returned an invalid slider position');

    return {
      kind: this.kind,
      value: sliderToAnswer(sliderPosition, problem.answerRange),
      sliderPosition,
      confidence: payload.confidence == null ? null : clamp01(Number(payload.confidence)),
      trace: Array.isArray(payload.trace) && payload.trace.length
        ? payload.trace.map((value) => clamp01(Number(value)))
        : [0.5, sliderPosition],
      telemetry: payload.telemetry ?? null,
    };
  }
}
