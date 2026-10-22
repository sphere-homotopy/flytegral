const test = require('node:test');
const assert = require('node:assert/strict');

const {
  DEFAULT_ACCOUNTS,
  OPTIONAL_ACCOUNTS,
  collectorLaunchOptions,
  mergeTweetRows,
  statusBelongsToAccount,
} = require('../cook-corpus-core');


test('default John D Cook corpus accounts match the focused math seed', () => {
  assert.deepEqual(DEFAULT_ACCOUNTS, [
    'ProbFact',
    'DataSciFact',
    'AnalysisFact',
    'AlgebraFact',
    'NetworkFact',
    'LogicPractice',
    'TopologyFact',
    'diff_eq',
    'FunctorFact',
  ]);
  assert.deepEqual(OPTIONAL_ACCOUNTS, [
    'dsp_fact',
    'CompSciFact',
    'SciPyTip',
    'TeXtip',
  ]);
});


test('status ownership is case-insensitive and rejects timeline reposts', () => {
  assert.equal(
    statusBelongsToAccount('AnalysisFact', 'https://x.com/AnalysisFact/status/123'),
    true,
  );
  assert.equal(
    statusBelongsToAccount('@AnalysisFact', 'https://twitter.com/analysisfact/status/123?s=20'),
    true,
  );
  assert.equal(
    statusBelongsToAccount('AnalysisFact', 'https://x.com/JohnDCook/status/123'),
    false,
  );
  assert.equal(
    statusBelongsToAccount('AnalysisFact', 'https://x.com/AnalysisFact/photo/1'),
    false,
  );
});


test('collector launch options are unconditionally headless', () => {
  const options = collectorLaunchOptions();
  assert.equal(options.headless, true);
  assert.deepEqual(options.viewport, { width: 1400, height: 1000 });
  assert.equal(options.locale, 'en-US');
});


test('tweet merge deduplicates by canonical status URL and keeps newest data', () => {
  const map = new Map();
  const added = mergeTweetRows(map, [
    {
      account: 'TopologyFact',
      tweet_url: 'https://x.com/TopologyFact/status/44?s=20',
      tweet_datetime: '2026-01-01T00:00:00.000Z',
      text: 'old',
    },
    {
      account: 'TopologyFact',
      tweet_url: 'https://x.com/TopologyFact/status/44',
      tweet_datetime: '2026-01-01T00:00:00.000Z',
      text: 'new',
    },
  ]);

  assert.equal(added, 1);
  assert.equal(map.size, 1);
  assert.equal(map.get('https://x.com/TopologyFact/status/44').text, 'new');
});
