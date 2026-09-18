'use strict';

const path = require('node:path');
const readline = require('node:readline/promises');
const { stdin: input, stdout: output } = require('node:process');
const { chromium } = require('playwright');

function profileDir() {
  return process.env.X_PROFILE_DIR
    ? path.resolve(process.env.X_PROFILE_DIR)
    : path.resolve(__dirname, '..', 'artifacts', 'x-browser-profile');
}

async function main() {
  const dir = profileDir();
  const context = await chromium.launchPersistentContext(dir, {
    headless: false,
    viewport: { width: 1400, height: 1000 },
    locale: 'en-US',
  });
  const page = context.pages()[0] || await context.newPage();

  try {
    await page.goto('https://x.com/home', {
      waitUntil: 'domcontentloaded',
      timeout: 60_000,
    });

    console.log(`X session profile: ${dir}`);
    console.log('Log in to X in the opened Chromium window if needed.');
    console.log('This is the only Fly Tweets command that intentionally opens a browser window.');

    const rl = readline.createInterface({ input, output });
    try {
      await rl.question('When the X home timeline is visible, press Enter here... ');
    } finally {
      rl.close();
    }

    await page.goto('https://x.com/home', {
      waitUntil: 'domcontentloaded',
      timeout: 60_000,
    });
    const current = page.url();
    if (/\/(i\/flow\/login|login)(?:[/?#]|$)/i.test(current)) {
      throw new Error('X still appears logged out; session profile was not bootstrapped.');
    }
    console.log('X persistent session is ready for headless collection.');
  } finally {
    await context.close();
  }
}

main().catch((error) => {
  console.error(`[fly-tweets bootstrap] ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
