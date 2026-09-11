import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);

async function read(path) {
  return readFile(new URL(path, root), 'utf8');
}

test('demo page wires graph, visible fly, slider and result metrics', async () => {
  const html = await read('index.html');

  assert.match(html, /id="graph-canvas"/);
  assert.match(html, /id="fly-agent"/);
  assert.match(html, /id="answer-slider"/);
  assert.match(html, /id="target-value"/);
  assert.match(html, /id="prediction-value"/);
  assert.match(html, /id="error-value"/);
  assert.match(html, /id="agent-badge-label"/);
  assert.match(html, /id="fly-agent-mode"/);
  assert.match(html, /type="module" src="\.\/src\/app\.js"/);
});

test('demo styling contains a responsive graph-fly-slider composition and thinking animation', async () => {
  const css = await read('styles.css');

  assert.match(css, /\.flow/);
  assert.match(css, /\.fly-stage/);
  assert.match(css, /\.fly-agent\.thinking/);
  assert.match(css, /@keyframes wingBeat/);
  assert.match(css, /@media/);
});

test('app connects generated problems to canvas rendering and agent estimates', async () => {
  const app = await read('src/app.js');

  assert.match(app, /generateProblem/);
  assert.match(app, /baselineAgent/);
  assert.match(app, /MaleCNSBridgeAgent/);
  assert.match(app, /agent=malecns|params.get\('agent'\)/);
  assert.match(app, /await activeAgent\.estimate\(/);
  assert.doesNotMatch(app, /const result = baselineAgent\.estimate/);
  assert.match(app, /drawGraph/);
  assert.match(app, /animateSlider/);
  assert.match(app, /recordDemo/);
});
