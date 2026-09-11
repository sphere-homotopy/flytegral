import { baselineAgent } from './agent.js';
import { MaleCNSBridgeAgent } from './malecns.js';
import { evaluatePolynomial, generateProblem, sliderToAnswer } from './math.js';

const elements = {
  canvas: document.querySelector('#graph-canvas'),
  equation: document.querySelector('#equation'),
  interval: document.querySelector('#interval-chip'),
  fly: document.querySelector('#fly-agent'),
  flyStage: document.querySelector('.fly-stage'),
  thinkingLabel: document.querySelector('#thinking-label'),
  slider: document.querySelector('#answer-slider'),
  answerLive: document.querySelector('#answer-live'),
  sliderMin: document.querySelector('#slider-min'),
  sliderMax: document.querySelector('#slider-max'),
  confidence: document.querySelector('#confidence'),
  target: document.querySelector('#target-value'),
  prediction: document.querySelector('#prediction-value'),
  error: document.querySelector('#error-value'),
  results: document.querySelector('#results'),
  errorBanner: document.querySelector('#error-banner'),
  seed: document.querySelector('#seed-value'),
  newProblem: document.querySelector('#new-problem'),
  recordDemoButton: document.querySelector('#record-demo'),
  recordingStatus: document.querySelector('#recording-status'),
  agentBadgeLabel: document.querySelector('#agent-badge-label'),
  flyAgentMode: document.querySelector('#fly-agent-mode'),
  footerMode: document.querySelector('#footer-mode'),
};

let currentSeed = initialSeed();
let currentProblem = null;
let runToken = 0;

const params = new URLSearchParams(window.location.search);
const agentMode = params.get('agent') === 'malecns' ? 'malecns' : 'baseline';
const activeAgent = agentMode === 'malecns'
  ? new MaleCNSBridgeAgent({ endpoint: params.get('brain') ?? 'http://127.0.0.1:8777' })
  : baselineAgent;

function configureAgentLabels() {
  if (agentMode === 'malecns') {
    elements.agentBadgeLabel.textContent = 'MaleCNS v1.0';
    elements.flyAgentMode.textContent = 'whole CNS + trained readout';
    elements.footerMode.textContent = 'MaleCNS v1.0 • real connectome recurrence • trained scalar readout';
  } else {
    elements.agentBadgeLabel.textContent = 'baseline agent';
    elements.flyAgentMode.textContent = 'baseline';
    elements.footerMode.textContent = 'visual benchmark • exact polynomial target • transparent baseline';
  }
}

function initialSeed() {
  const params = new URLSearchParams(window.location.search);
  const requested = Number.parseInt(params.get('seed') ?? '', 10);
  if (Number.isInteger(requested)) return requested;
  return 260912;
}

function formatNumber(value, digits = 2) {
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  return normalized.toFixed(digits).replace('-', '−');
}

function formatCoefficient(value) {
  if (Number.isInteger(value)) return String(Math.abs(value));
  return String(Math.abs(value)).replace(/\.0+$/, '');
}

function formatPolynomial(coefficients) {
  const terms = [];
  const powers = ['', 'x', 'x²', 'x³'];

  coefficients.forEach((coefficient, degree) => {
    if (coefficient === 0) return;
    const sign = coefficient < 0 ? '−' : '+';
    const magnitude = Math.abs(coefficient);
    const number = degree > 0 && magnitude === 1 ? '' : formatCoefficient(coefficient);
    const body = `${number}${powers[degree]}`;

    if (terms.length === 0) {
      terms.push(coefficient < 0 ? `−${body}` : body);
    } else {
      terms.push(`${sign} ${body}`);
    }
  });

  return `f(x) = ${terms.join(' ') || '0'}`;
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

function graphExtents(problem) {
  const [xMin, xMax] = problem.graphDomain;
  let maxAbs = 1;
  for (let i = 0; i <= 180; i += 1) {
    const x = xMin + (i / 180) * (xMax - xMin);
    maxAbs = Math.max(maxAbs, Math.abs(evaluatePolynomial(problem.coefficients, x)));
  }
  return { xMin, xMax, yMin: -maxAbs * 1.14, yMax: maxAbs * 1.14 };
}

function drawRoundedLabel(ctx, text, x, y, align = 'center') {
  ctx.save();
  ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
  const paddingX = 7;
  const width = ctx.measureText(text).width + paddingX * 2;
  const left = align === 'left' ? x : x - width / 2;
  ctx.fillStyle = 'rgba(255,255,255,.92)';
  ctx.strokeStyle = '#e1e4ec';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(left, y - 12, width, 22, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#5e6474';
  ctx.textAlign = align === 'left' ? 'left' : 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, align === 'left' ? left + paddingX : x, y - 1);
  ctx.restore();
}

function drawGraph(problem) {
  const canvas = elements.canvas;
  const ctx = canvas.getContext('2d');
  const { width, height, ratio } = resizeCanvas(canvas);
  const { xMin, xMax, yMin, yMax } = graphExtents(problem);
  const pad = 40 * ratio;
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;

  const xToPx = (x) => pad + ((x - xMin) / (xMax - xMin)) * innerWidth;
  const yToPx = (y) => pad + ((yMax - y) / (yMax - yMin)) * innerHeight;
  const zeroY = yToPx(0);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#fbfcff';
  ctx.fillRect(0, 0, width, height);

  ctx.lineWidth = ratio;
  ctx.strokeStyle = '#edf0f5';
  for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x += 1) {
    const px = xToPx(x);
    ctx.beginPath();
    ctx.moveTo(px, pad);
    ctx.lineTo(px, height - pad);
    ctx.stroke();
  }
  for (let i = 1; i < 6; i += 1) {
    const py = pad + (i / 6) * innerHeight;
    ctx.beginPath();
    ctx.moveTo(pad, py);
    ctx.lineTo(width - pad, py);
    ctx.stroke();
  }

  ctx.strokeStyle = '#b6bbc8';
  ctx.lineWidth = 1.25 * ratio;
  ctx.beginPath();
  ctx.moveTo(pad, zeroY);
  ctx.lineTo(width - pad, zeroY);
  ctx.stroke();

  if (xMin <= 0 && xMax >= 0) {
    const zeroX = xToPx(0);
    ctx.beginPath();
    ctx.moveTo(zeroX, pad);
    ctx.lineTo(zeroX, height - pad);
    ctx.stroke();
  }

  const [a, b] = problem.interval;
  ctx.save();
  ctx.beginPath();
  ctx.rect(xToPx(a), pad, xToPx(b) - xToPx(a), innerHeight);
  ctx.clip();

  ctx.beginPath();
  ctx.moveTo(xToPx(a), zeroY);
  const areaSteps = 100;
  for (let i = 0; i <= areaSteps; i += 1) {
    const x = a + (i / areaSteps) * (b - a);
    ctx.lineTo(xToPx(x), yToPx(evaluatePolynomial(problem.coefficients, x)));
  }
  ctx.lineTo(xToPx(b), zeroY);
  ctx.closePath();
  const areaGradient = ctx.createLinearGradient(0, pad, 0, height - pad);
  areaGradient.addColorStop(0, 'rgba(108, 92, 231, .22)');
  areaGradient.addColorStop(1, 'rgba(40, 184, 167, .16)');
  ctx.fillStyle = areaGradient;
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = '#6c5ce7';
  ctx.lineWidth = 3 * ratio;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  const steps = 220;
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (i / steps) * (xMax - xMin);
    const px = xToPx(x);
    const py = yToPx(evaluatePolynomial(problem.coefficients, x));
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  ctx.save();
  ctx.setLineDash([5 * ratio, 5 * ratio]);
  ctx.lineWidth = 1.5 * ratio;
  ctx.strokeStyle = 'rgba(108, 92, 231, .55)';
  for (const boundary of [a, b]) {
    const px = xToPx(boundary);
    ctx.beginPath();
    ctx.moveTo(px, pad);
    ctx.lineTo(px, height - pad);
    ctx.stroke();
  }
  ctx.restore();

  ctx.save();
  ctx.scale(ratio, ratio);
  drawRoundedLabel(ctx, `a = ${formatNumber(a, 1)}`, xToPx(a) / ratio, (height - pad + 19 * ratio) / ratio);
  drawRoundedLabel(ctx, `b = ${formatNumber(b, 1)}`, xToPx(b) / ratio, (height - pad + 19 * ratio) / ratio);
  ctx.restore();
}

function setSliderPosition(position, problem) {
  const normalized = Math.min(1, Math.max(0, position));
  elements.slider.value = String(Math.round(normalized * 1000));
  elements.answerLive.textContent = formatNumber(sliderToAnswer(normalized, problem.answerRange));
}

function tween(from, to, duration, onFrame) {
  return new Promise((resolve) => {
    const started = performance.now();
    function frame(now) {
      const progress = Math.min(1, (now - started) / duration);
      const eased = 1 - (1 - progress) ** 3;
      onFrame(from + (to - from) * eased);
      if (progress < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });
}

async function animateSlider(trace, problem, token) {
  let previous = Number(elements.slider.value) / 1000;
  for (const position of trace) {
    if (token !== runToken) return;
    await tween(previous, position, 320, (value) => setSliderPosition(value, problem));
    previous = position;
  }
}

function setThinking(thinking) {
  elements.fly.classList.toggle('thinking', thinking);
  elements.flyStage.classList.toggle('thinking', thinking);
  elements.thinkingLabel.textContent = thinking ? 'estimating area…' : 'answer locked';
}

function resetResults(problem) {
  elements.results.classList.remove('revealed');
  elements.target.textContent = '—';
  elements.prediction.textContent = '—';
  elements.error.textContent = '—';
  elements.confidence.textContent = '—';
  elements.errorBanner.hidden = true;
  elements.sliderMin.textContent = formatNumber(problem.answerRange[0], 1);
  elements.sliderMax.textContent = formatNumber(problem.answerRange[1], 1);
  setSliderPosition(0.5, problem);
}

async function runProblem(seed = currentSeed) {
  const token = ++runToken;
  currentSeed = seed;
  elements.newProblem.disabled = true;
  elements.errorBanner.hidden = true;

  try {
    const problem = generateProblem({ seed });
    currentProblem = problem;
    const result = await activeAgent.estimate(problem, { seed: seed ^ 0x9e3779b9 });

    elements.seed.textContent = String(seed);
    elements.equation.textContent = formatPolynomial(problem.coefficients);
    elements.interval.textContent = `[${formatNumber(problem.interval[0], 1)}, ${formatNumber(problem.interval[1], 1)}]`;
    resetResults(problem);
    drawGraph(problem);
    setThinking(true);

    await new Promise((resolve) => setTimeout(resolve, 360));
    await animateSlider(result.trace, problem, token);
    if (token !== runToken) return;

    setThinking(false);
    elements.target.textContent = formatNumber(problem.target, 3);
    elements.prediction.textContent = formatNumber(result.value, 3);
    elements.error.textContent = formatNumber(Math.abs(result.value - problem.target), 3);
    elements.confidence.textContent = result.confidence == null
      ? (result.telemetry?.totalSpikes != null ? `${result.telemetry.totalSpikes.toLocaleString()} spikes` : 'connectome')
      : `${Math.round(result.confidence * 100)}% conf.`;
    elements.results.classList.add('revealed');
  } catch (error) {
    console.error(error);
    setThinking(false);
    elements.errorBanner.hidden = false;
    elements.errorBanner.textContent = `Demo error: ${error.message}`;
  } finally {
    if (token === runToken) elements.newProblem.disabled = false;
  }
}

function nextProblem() {
  return runProblem(currentSeed + 1);
}

function chooseRecordingMimeType() {
  const candidates = [
    'video/mp4;codecs=avc1',
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? '';
}

function downloadBlob(blob, extension) {
  const anchor = document.createElement('a');
  const url = URL.createObjectURL(blob);
  anchor.href = url;
  anchor.download = `flytegral-seed-${currentSeed}.${extension}`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function recordDemo() {
  if (!navigator.mediaDevices?.getDisplayMedia || typeof MediaRecorder === 'undefined') {
    elements.recordingStatus.textContent = 'recording is not supported by this browser';
    return;
  }

  elements.recordDemoButton.disabled = true;
  elements.recordingStatus.textContent = 'choose this tab to record';
  let stream;

  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 30 },
      audio: false,
      preferCurrentTab: true,
    });

    const mimeType = chooseRecordingMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 6_000_000 } : undefined);
    const chunks = [];
    recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size) chunks.push(event.data);
    });

    const stopped = new Promise((resolve) => recorder.addEventListener('stop', resolve, { once: true }));
    recorder.start(100);
    elements.recordingStatus.textContent = '● recording';

    await runProblem(currentSeed + 1);
    await new Promise((resolve) => setTimeout(resolve, 900));
    recorder.stop();
    await stopped;

    const type = recorder.mimeType || mimeType || 'video/webm';
    const extension = type.includes('mp4') ? 'mp4' : 'webm';
    downloadBlob(new Blob(chunks, { type }), extension);
    elements.recordingStatus.textContent = extension === 'mp4' ? 'MP4 saved' : 'WebM saved (MP4 unavailable in this browser)';
  } catch (error) {
    if (error.name !== 'NotAllowedError') console.error(error);
    elements.recordingStatus.textContent = error.name === 'NotAllowedError' ? 'recording cancelled' : `recording failed: ${error.message}`;
  } finally {
    stream?.getTracks().forEach((track) => track.stop());
    elements.recordDemoButton.disabled = false;
    window.setTimeout(() => {
      if (!elements.recordingStatus.textContent.startsWith('●')) elements.recordingStatus.textContent = '';
    }, 3500);
  }
}

elements.newProblem.addEventListener('click', nextProblem);
elements.recordDemoButton.addEventListener('click', recordDemo);
window.addEventListener('resize', () => {
  if (currentProblem) drawGraph(currentProblem);
});

configureAgentLabels();
runProblem(currentSeed);
