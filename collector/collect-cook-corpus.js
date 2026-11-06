'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const {
  DEFAULT_ACCOUNTS,
  canonicalHandle,
  collectorLaunchOptions,
  mergeTweetRows,
  statusBelongsToAccount,
} = require('./cook-corpus-core');

const DEFAULT_OUTPUT_DIR = path.resolve(
  __dirname,
  '..',
  'artifacts',
  'corpus',
  'john-d-cook',
);
const MAX_SCROLLS = positiveInt(process.env.COOK_MAX_SCROLLS, 450);
const STABLE_ROUNDS = positiveInt(process.env.COOK_STABLE_ROUNDS, 18);
const MIN_DELAY_MS = positiveInt(process.env.COOK_MIN_DELAY_MS, 650);
const MAX_DELAY_MS = positiveInt(process.env.COOK_MAX_DELAY_MS, 1250);

function positiveInt(value, fallback) {
  const parsed = Number.parseInt(value || '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function profileDir() {
  return process.env.X_PROFILE_DIR
    ? path.resolve(process.env.X_PROFILE_DIR)
    : path.resolve(__dirname, '..', 'artifacts', 'x-browser-profile');
}

function outputDir() {
  return process.env.COOK_CORPUS_OUTPUT_DIR
    ? path.resolve(process.env.COOK_CORPUS_OUTPUT_DIR)
    : DEFAULT_OUTPUT_DIR;
}

function configuredAccounts() {
  const raw = String(process.env.COOK_ACCOUNTS || '').trim();
  if (!raw) return [...DEFAULT_ACCOUNTS];
  const accounts = [...new Set(
    raw.split(',').map(canonicalHandle).filter(Boolean),
  )];
  if (!accounts.length) throw new Error('COOK_ACCOUNTS did not contain any handles.');
  return accounts;
}

function delayMs() {
  const lo = Math.min(MIN_DELAY_MS, MAX_DELAY_MS);
  const hi = Math.max(MIN_DELAY_MS, MAX_DELAY_MS);
  return Math.floor(lo + Math.random() * (hi - lo + 1));
}

async function assertLoggedIn(page) {
  await page.goto('https://x.com/home', {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  if (/\/(i\/flow\/login|login)(?:[/?#]|$)/i.test(page.url())) {
    throw new Error(
      'X session is logged out. Run `npm run bootstrap:x` once with the same X_PROFILE_DIR.',
    );
  }

  try {
    await page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first().waitFor({
      state: 'attached',
      timeout: 15_000,
    });
  } catch {
    const body = await page.locator('body').innerText().catch(() => '');
    if (/sign in|log in|create account/i.test(body)) {
      throw new Error(
        'X session is not authenticated. Run `npm run bootstrap:x` once with the same X_PROFILE_DIR.',
      );
    }
    throw new Error('Could not verify the authenticated X session; refusing to scrape anonymously.');
  }
}

async function extractVisibleTweets(page, account) {
  return page.evaluate((expectedAccount) => {
    const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const rows = [];

    for (const article of document.querySelectorAll('article[role="article"], [data-testid="tweet"]')) {
      const time = article.querySelector('time');
      const timeLink = time && time.closest('a[href*="/status/"]');
      if (!timeLink) continue;

      const href = timeLink.getAttribute('href') || '';
      let tweetUrl = '';
      try {
        tweetUrl = new URL(href, location.origin).href.split('?')[0];
      } catch {
        continue;
      }

      const textNode = article.querySelector('[data-testid="tweetText"]');
      const text = clean(textNode ? textNode.innerText : '');
      if (!text) continue;

      const socialContext = clean(
        article.querySelector('[data-testid="socialContext"]')?.innerText || '',
      );
      rows.push({
        account: expectedAccount,
        tweet_url: tweetUrl,
        tweet_datetime: time.getAttribute('datetime') || '',
        text,
        is_repost: /reposted/i.test(socialContext),
        collected_at: new Date().toISOString(),
      });
    }
    return rows;
  }, account);
}

async function collectAccount(page, account, rowsByUrl) {
  const handle = canonicalHandle(account);
  await page.goto(`https://x.com/${encodeURIComponent(handle)}`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });

  if (/\/(i\/flow\/login|login)(?:[/?#]|$)/i.test(page.url())) {
    throw new Error(`X redirected @${handle} to login; session expired.`);
  }

  await page.locator('article[role="article"], [data-testid="tweet"]').first().waitFor({
    state: 'attached',
    timeout: 30_000,
  });

  let stableRounds = 0;
  let totalAdded = 0;
  let previousScrollY = -1;

  for (let scroll = 0; scroll < MAX_SCROLLS && stableRounds < STABLE_ROUNDS; scroll += 1) {
    const visible = await extractVisibleTweets(page, handle);
    const own = visible.filter((row) => (
      !row.is_repost && statusBelongsToAccount(handle, row.tweet_url)
    ));
    const added = mergeTweetRows(rowsByUrl, own);
    totalAdded += added;
    stableRounds = added === 0 ? stableRounds + 1 : 0;

    const scrollY = await page.evaluate(() => window.scrollY);
    await page.evaluate(() => {
      window.scrollBy(0, Math.max(900, Math.floor(window.innerHeight * 0.92)));
    });
    await page.waitForTimeout(delayMs());
    const nextScrollY = await page.evaluate(() => window.scrollY);
    if (nextScrollY === scrollY && nextScrollY === previousScrollY) stableRounds += 2;
    previousScrollY = nextScrollY;
  }

  return totalAdded;
}

function loadExistingJsonl(filePath) {
  const map = new Map();
  if (!fs.existsSync(filePath)) return map;
  const body = fs.readFileSync(filePath, 'utf8');
  const rows = body.split(/\r?\n/).filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line);
    } catch (error) {
      throw new Error(`Invalid JSONL at ${filePath}:${index + 1}: ${error.message}`);
    }
  });
  mergeTweetRows(map, rows);
  return map;
}

function writeAtomic(filePath, contents) {
  const temp = `${filePath}.tmp-${process.pid}`;
  fs.writeFileSync(temp, contents, 'utf8');
  fs.renameSync(temp, filePath);
}

function persistCorpus(rowsByUrl, accounts, dir) {
  fs.mkdirSync(dir, { recursive: true });
  const jsonlPath = path.join(dir, 'cook-math-tweets.jsonl');
  const manifestPath = path.join(dir, 'manifest.json');
  const rows = [...rowsByUrl.values()].sort((a, b) => {
    const byTime = String(a.tweet_datetime).localeCompare(String(b.tweet_datetime));
    return byTime || String(a.tweet_url).localeCompare(String(b.tweet_url));
  });

  const jsonl = rows.map((row) => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : '');
  writeAtomic(jsonlPath, jsonl);

  const counts = Object.fromEntries(accounts.map((account) => [
    account,
    rows.filter((row) => canonicalHandle(row.account).toLowerCase() === account.toLowerCase()).length,
  ]));
  const manifest = {
    schema_version: 1,
    source: 'john-d-cook-x-math-facts',
    collected_at: new Date().toISOString(),
    accounts,
    rows: rows.length,
    counts,
    data_file: path.basename(jsonlPath),
  };
  writeAtomic(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return { jsonlPath, manifestPath, rows: rows.length, counts };
}

async function main() {
  const accounts = configuredAccounts();
  const dir = outputDir();
  fs.mkdirSync(dir, { recursive: true });
  const jsonlPath = path.join(dir, 'cook-math-tweets.jsonl');
  const rowsByUrl = loadExistingJsonl(jsonlPath);

  const context = await chromium.launchPersistentContext(profileDir(), collectorLaunchOptions());
  const page = context.pages()[0] || await context.newPage();
  let result = persistCorpus(rowsByUrl, accounts, dir);
  try {
    await assertLoggedIn(page);
    for (const account of accounts) {
      const before = rowsByUrl.size;
      await collectAccount(page, account, rowsByUrl);
      result = persistCorpus(rowsByUrl, accounts, dir);
      console.log(`@${account}: ${rowsByUrl.size - before} new rows (${rowsByUrl.size} total)`);
      console.log(JSON.stringify({ checkpoint: account, ...result }, null, 2));
    }
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await context.close();
  }
}

main().catch((error) => {
  console.error(`[fly-tweets corpus] ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
