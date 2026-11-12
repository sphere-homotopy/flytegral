'use strict';

function metricsCollectorLaunchOptions() {
  return {
    headless: true,
    viewport: { width: 1400, height: 1000 },
    locale: 'en-US',
  };
}

function parseCompactCount(raw) {
  const normalized = String(raw || '').trim().replace(/,/g, '').toUpperCase();
  const match = normalized.match(/^([0-9]+(?:\.[0-9]+)?)([KMB])?$/);
  if (!match) throw new Error(`invalid compact metric count: ${raw}`);
  const multiplier = {
    K: 1_000,
    M: 1_000_000,
    B: 1_000_000_000,
  }[match[2]] || 1;
  return Math.round(Number(match[1]) * multiplier);
}

function parseMetricAriaLabel(label) {
  const result = {
    views: 0,
    likes: 0,
    reposts: 0,
    replies: 0,
    bookmarks: 0,
  };
  const metricMap = {
    view: 'views',
    views: 'views',
    like: 'likes',
    likes: 'likes',
    repost: 'reposts',
    reposts: 'reposts',
    reply: 'replies',
    replies: 'replies',
    bookmark: 'bookmarks',
    bookmarks: 'bookmarks',
  };
  const regex = /([0-9][0-9,.]*(?:\.[0-9]+)?\s*[KMB]?)\s+(views?|likes?|reposts?|repl(?:y|ies)|bookmarks?)/gi;
  let match;
  while ((match = regex.exec(String(label || ''))) !== null) {
    const rawMetric = match[2].toLowerCase();
    const normalizedMetric = rawMetric === 'reply' || rawMetric === 'replies'
      ? rawMetric
      : rawMetric;
    const key = metricMap[normalizedMetric];
    if (key) result[key] = parseCompactCount(match[1].replace(/\s+/g, ''));
  }
  return result;
}


function xProfileHrefMatchesExpectedHandle(href, expectedHandle) {
  const normalizedHandle = String(expectedHandle || '').trim().replace(/^@/, '').toLowerCase();
  if (!/^[a-z0-9_]{1,15}$/.test(normalizedHandle)) {
    throw new Error(`invalid expected X handle: ${expectedHandle}`);
  }

  let pathname;
  try {
    pathname = new URL(String(href || ''), 'https://x.com').pathname;
  } catch {
    return false;
  }

  const parts = pathname.split('/').filter(Boolean);
  return parts.length === 1 && parts[0].toLowerCase() === normalizedHandle;
}

module.exports = {
  metricsCollectorLaunchOptions,
  parseCompactCount,
  parseMetricAriaLabel,
  xProfileHrefMatchesExpectedHandle,
};
