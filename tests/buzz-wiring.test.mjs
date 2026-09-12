import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const read = (path) => readFile(new URL(path, root), 'utf8');

test('app starts and stops the buzz with fly thinking and requests tab audio for recordings', async () => {
  const app = await read('src/app.js');

  assert.match(app, /import \{ ThinkingBuzz \} from '\.\/buzz\.js'/);
  assert.match(app, /if \(thinking\) void thinkingBuzz\.start\(\)/);
  assert.match(app, /else thinkingBuzz\.stop\(\)/);
  assert.match(app, /audio: true/);
  assert.match(app, /enable Share tab audio/);
});
