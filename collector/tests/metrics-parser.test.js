const test = require('node:test');
const assert = require('node:assert/strict');

const {
  metricsCollectorLaunchOptions,
  parseCompactCount,
  parseMetricAriaLabel,
  xProfileHrefMatchesExpectedHandle,
} = require('../metrics-core');


test('compact metric counts support commas and K/M/B suffixes', () => {
  assert.equal(parseCompactCount('1,234'), 1234);
  assert.equal(parseCompactCount('1.2K'), 1200);
  assert.equal(parseCompactCount('2.5M'), 2500000);
  assert.equal(parseCompactCount('1B'), 1000000000);
});


test('aria label parser handles plural metrics in arbitrary order', () => {
  assert.deepEqual(
    parseMetricAriaLabel('45 replies, 1,234 likes, 12 reposts, 2 bookmarks, 9.8K views'),
    {
      views: 9800,
      likes: 1234,
      reposts: 12,
      replies: 45,
      bookmarks: 2,
    },
  );
});


test('aria label parser handles singular and missing metrics as zero', () => {
  assert.deepEqual(
    parseMetricAriaLabel('1 reply, 1 like, 1 repost, 1 view'),
    {
      views: 1,
      likes: 1,
      reposts: 1,
      replies: 1,
      bookmarks: 0,
    },
  );
});


test('aria label parser ignores unrelated numbers', () => {
  assert.deepEqual(
    parseMetricAriaLabel('Post from Sep 17. 300 views, 8 likes'),
    {
      views: 300,
      likes: 8,
      reposts: 0,
      replies: 0,
      bookmarks: 0,
    },
  );
});


test('metrics collector launch is always headless', () => {
  const options = metricsCollectorLaunchOptions();
  assert.equal(options.headless, true);
  assert.equal(options.locale, 'en-US');
});


test('X profile guard accepts only the expected account handle', () => {
  assert.equal(typeof xProfileHrefMatchesExpectedHandle, 'function');
  assert.equal(xProfileHrefMatchesExpectedHandle('/fly_topology', 'fly_topology'), true);
  assert.equal(xProfileHrefMatchesExpectedHandle('/fly_topology/', '@fly_topology'), true);
  assert.equal(xProfileHrefMatchesExpectedHandle('/sphere_homotopy', 'fly_topology'), false);
  assert.equal(xProfileHrefMatchesExpectedHandle('/fly_topology/status/123', 'fly_topology'), false);
});
