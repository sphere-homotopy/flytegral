'use strict';
const fs = require('node:fs');
const path = require('node:path');

function readQueue(file) {
  const value = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (!Array.isArray(value) || value.some(row => !row || typeof row !== 'object')) {
    throw new Error('native X queue must be a list of objects');
  }
  const ids = value.map(row => String(row.id || ''));
  if (ids.some(id => !id) || new Set(ids).size !== ids.length) {
    throw new Error('queue IDs must be nonempty and unique');
  }
  return value;
}

function writeAtomic(file, rows) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temp = file + '.tmp-' + process.pid;
  fs.writeFileSync(temp, JSON.stringify(rows, null, 2) + '\n', 'utf8');
  fs.renameSync(temp, file);
}

function normalizeText(text) {
  return String(text || '').normalize('NFKC').replace(/\s+/g, ' ').trim();
}

function validateTweet(row) {
  if (row.status !== 'ready') throw new Error('row must be ready');
  const text = normalizeText(row.text);
  if (text.length < 15 || text.length > 280) throw new Error('invalid tweet length');
  if (row.generator !== 'fly_compositional_style_v2') throw new Error('wrong generator');
  if (!['fly_cadence','manual_smoke'].includes(row.origin)) throw new Error('unrecognized origin');
  if (typeof row.id !== 'string' || !row.id) throw new Error('missing queue ID');
  const date = new Date(row.scheduled_at);
  if (Number.isNaN(date.valueOf()) || !/[zZ]|[+-]\d\d:\d\d$/.test(row.scheduled_at)) {
    throw new Error('invalid time');
  }
  return text;
}

function pickDue(rows, now = new Date(), opts = {}) {
  const minGapMs = (opts.minGapMinutes ?? 20) * 60 * 1000;
  const maxDelayMs = (opts.maxLateHours ?? 6) * 60 * 60 * 1000;
  const published = rows.filter(row => row.status === 'published' && row.posted_at)
    .map(row => Date.parse(row.posted_at)).filter(Number.isFinite);
  const recent = published.length ? Math.max(...published) : -Infinity;
  if (now.getTime() - recent < minGapMs) return null;
  const eligible = rows.filter(row => row.status === 'ready')
    .sort((a, b) => Date.parse(a.scheduled_at) - Date.parse(b.scheduled_at));
  for (const row of eligible) {
    validateTweet(row);
    const delta = now.getTime() - Date.parse(row.scheduled_at);
    if (delta < 0) return null; // The earliest future post blocks later ones.
    if (delta > maxDelayMs && row.origin !== 'manual_smoke') {
      continue; // Mark missed separately without surprising retroactive publication.
    }
    return row;
  }
  return null;
}

function expireMissed(rows, now = new Date(), maxLateHours = 6) {
  let n = 0;
  for (const row of rows) {
    if (row.status !== 'ready' || row.origin === 'manual_smoke') continue;
    if (Date.parse(row.scheduled_at) < now.getTime() - maxLateHours * 3600_000) {
      row.status = 'missed';
      row.last_error = 'X runner offline or rate-limited past 6h grace period';
      n++;
    }
  }
  return n;
}

function verifyTweetUrl(url, handle) {
  const expected = String(handle).replace(/^@/, '').toLowerCase();
  let obj;
  try { obj = new URL(url, 'https://x.com'); } catch { return false; }
  return ['x.com','twitter.com'].includes(obj.hostname.toLowerCase()) &&
    new RegExp('^/' + expected + '/status/[0-9]+/?$', 'i').test(obj.pathname);
}

function lock(lockPath) {
  try {
    const fd = fs.openSync(lockPath, 'wx');
    fs.writeFileSync(fd, String(process.pid));
    return () => { fs.closeSync(fd); fs.unlinkSync(lockPath); };
  } catch (err) {
    if (err.code === 'EEXIST') throw new Error('another native X runner owns queue lock: ' + lockPath);
    throw err;
  }
}

module.exports = { readQueue, writeAtomic, normalizeText, validateTweet, pickDue,
                   expireMissed, verifyTweetUrl, lock };
