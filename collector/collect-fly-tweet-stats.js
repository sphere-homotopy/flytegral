'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const {
  metricsCollectorLaunchOptions,
  parseMetricAriaLabel,
} = require('./metrics-core');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) throw new Error(`unexpected argument: ${key}`);
    const value = argv[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`missing value for ${key}`);
    args[key.slice(2)] = value;
    i += 1;
  }
  if (!args.input) throw new Error('--input JSONL path is required');
  if (!args.output) throw new Error('--output JSONL path is required');
  return args;
}

function profileDir() {
  return process.env.X_PROFILE_DIR
    ? path.resolve(process.env.X_PROFILE_DIR)
    : path.resolve(__dirname, '..', 'artifacts', 'x-browser-profile');
}

function loadRows(filePath) {
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        const row = JSON.parse(line);
        if (!row.tweet_url) throw new Error('tweet_url is required');
        return row;
      } catch (error) {
        throw new Error(`Invalid JSONL at ${filePath}:${index + 1}: ${error.message}`);
      }
    });
}

function writeAtomic(filePath, rows) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temp = `${filePath}.tmp-${process.pid}`;
  const body = rows.map((row) => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : '');
  fs.writeFileSync(temp, body, 'utf8');
  fs.renameSync(temp, filePath);
}

async function assertLoggedIn(page) {
  await page.goto('https://x.com/home', { waitUntil: 'domcontentloaded', timeout: 60_000 });
  if (/\/(i\/flow\/login|login)(?:[/?#]|$)/i.test(page.url())) {
    throw new Error('X session is logged out; run the one-time bootstrap first.');
  }
  await page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first().waitFor({
    state: 'attached',
    timeout: 15_000,
  });
}

async function collectOne(page, sourceRow) {
  await page.goto(sourceRow.tweet_url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  if (/\/(i\/flow\/login|login)(?:[/?#]|$)/i.test(page.url())) {
    throw new Error('X session expired while collecting metrics.');
  }

  const article = page.locator('article[role="article"], [data-testid="tweet"]').first();
  await article.waitFor({ state: 'attached', timeout: 30_000 });
  const extracted = await article.evaluate((node) => {
    const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const text = clean(node.querySelector('[data-testid="tweetText"]')?.innerText || '');
    const datetime = node.querySelector('time')?.getAttribute('datetime') || '';
    const labels = [...node.querySelectorAll('[role="group"][aria-label], [aria-label]')]
      .map((element) => element.getAttribute('aria-label') || '')
      .filter((label) => /views?|likes?|reposts?|repl(?:y|ies)|bookmarks?/i.test(label));
    labels.sort((a, b) => {
      const score = (label) => (label.match(/views?|likes?|reposts?|repl(?:y|ies)|bookmarks?/gi) || []).length;
      return score(b) - score(a);
    });
    return { text, datetime, metricsLabel: labels[0] || '' };
  });

  if (!extracted.metricsLabel) {
    throw new Error(`No metrics aria-label found for ${sourceRow.tweet_url}`);
  }
  const metrics = parseMetricAriaLabel(extracted.metricsLabel);
  return {
    ...sourceRow,
    tweet_datetime: sourceRow.tweet_datetime || extracted.datetime,
    observed_text: extracted.text,
    ...metrics,
    collected_at: new Date().toISOString(),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const input = path.resolve(args.input);
  const output = path.resolve(args.output);
  const rows = loadRows(input);
  const context = await chromium.launchPersistentContext(
    profileDir(),
    metricsCollectorLaunchOptions(),
  );
  const page = context.pages()[0] || await context.newPage();
  const collected = [];
  try {
    await assertLoggedIn(page);
    for (const row of rows) {
      const metricsRow = await collectOne(page, row);
      collected.push(metricsRow);
      writeAtomic(output, collected);
      console.log(JSON.stringify({
        event: 'metrics_collected',
        tweet_url: metricsRow.tweet_url,
        views: metricsRow.views,
        likes: metricsRow.likes,
        reposts: metricsRow.reposts,
        replies: metricsRow.replies,
        bookmarks: metricsRow.bookmarks,
      }));
      await page.waitForTimeout(750);
    }
  } finally {
    await context.close();
  }
}

main().catch((error) => {
  console.error(`[fly-tweets metrics] ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
