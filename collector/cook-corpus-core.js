'use strict';

const DEFAULT_ACCOUNTS = Object.freeze([
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

const OPTIONAL_ACCOUNTS = Object.freeze([
  'dsp_fact',
  'CompSciFact',
  'SciPyTip',
  'TeXtip',
]);

function canonicalHandle(handle) {
  return String(handle || '').trim().replace(/^@/, '');
}

function parseStatusUrl(tweetUrl) {
  let parsed;
  try {
    parsed = new URL(tweetUrl);
  } catch {
    return null;
  }

  const parts = parsed.pathname.split('/').filter(Boolean);
  if (parts.length < 3 || parts[1].toLowerCase() !== 'status' || !/^\d+$/.test(parts[2])) {
    return null;
  }
  return {
    handle: parts[0],
    statusId: parts[2],
    canonicalUrl: `https://x.com/${parts[0]}/status/${parts[2]}`,
  };
}

function statusBelongsToAccount(account, tweetUrl) {
  const status = parseStatusUrl(tweetUrl);
  if (!status) return false;
  return status.handle.toLowerCase() === canonicalHandle(account).toLowerCase();
}

function collectorLaunchOptions() {
  return {
    headless: true,
    viewport: { width: 1400, height: 1000 },
    locale: 'en-US',
  };
}

function mergeTweetRows(map, rows) {
  let added = 0;
  for (const row of rows) {
    const status = parseStatusUrl(row && row.tweet_url);
    if (!status) continue;
    if (!map.has(status.canonicalUrl)) added += 1;
    map.set(status.canonicalUrl, {
      ...(map.get(status.canonicalUrl) || {}),
      ...row,
      tweet_url: status.canonicalUrl,
    });
  }
  return added;
}

module.exports = {
  DEFAULT_ACCOUNTS,
  OPTIONAL_ACCOUNTS,
  canonicalHandle,
  collectorLaunchOptions,
  mergeTweetRows,
  parseStatusUrl,
  statusBelongsToAccount,
};
