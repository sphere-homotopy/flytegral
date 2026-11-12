'use strict';

const FLY_TWEETS_BUFFER_ENDPOINT = 'https://api.buffer.com';

function buildBufferCreatePostPayload(row, channelId) {
  const text = String(row && row.text || '').trim();
  const scheduledAt = String(row && row.scheduled_at || '').trim();
  const idempotencyKey = String(row && row.idempotency_key || '').trim();
  const channel = String(channelId || '').trim();
  if (!text) throw new Error('text is required');
  if (!scheduledAt) throw new Error('scheduled_at is required');
  if (!idempotencyKey) throw new Error('idempotency_key is required');
  if (!channel) throw new Error('Buffer channel id is required');
  const parsed = Date.parse(scheduledAt);
  if (!Number.isFinite(parsed)) throw new Error('scheduled_at must be a valid ISO datetime');

  return {
    query: [
      'mutation CreateFlyTweet($input: CreatePostInput!) {',
      '  createPost(input: $input) {',
      '    __typename',
      '    ... on PostActionSuccess { post { id dueAt } }',
      '    ... on MutationError { message }',
      '  }',
      '}',
    ].join('\n'),
    variables: {
      input: {
        text,
        channelId: channel,
        schedulingType: 'automatic',
        mode: 'customScheduled',
        dueAt: new Date(parsed).toISOString(),
        source: `flytegral:${idempotencyKey}`,
      },
    },
  };
}

function parseBufferCreatePostResponse(payload) {
  if (!payload || typeof payload !== 'object') throw new Error('Buffer response must be an object');
  if (Array.isArray(payload.errors) && payload.errors.length) {
    throw new Error(payload.errors.map((item) => item && item.message || String(item)).join('; '));
  }
  const result = payload.data && payload.data.createPost;
  if (!result) throw new Error('Buffer response is missing data.createPost');
  if (result.__typename === 'PostActionSuccess' && result.post && result.post.id) {
    return { id: String(result.post.id), dueAt: String(result.post.dueAt || '') };
  }
  throw new Error(String(result.message || `Buffer createPost failed: ${result.__typename || 'unknown'}`));
}

function flyTweetsBufferConfig_() {
  const properties = PropertiesService.getScriptProperties();
  const apiKey = String(properties.getProperty('BUFFER_API_KEY') || '').trim();
  const channelId = String(properties.getProperty('BUFFER_CHANNEL_ID') || '').trim();
  const sheetName = String(properties.getProperty('FLY_TWEETS_SHEET_NAME') || 'FlyTweets').trim();
  if (!apiKey) throw new Error('Script property BUFFER_API_KEY is required');
  if (!channelId) throw new Error('Script property BUFFER_CHANNEL_ID is required');
  return { apiKey, channelId, sheetName };
}

function queueGeneratedFlyTweetsToBuffer() {
  const config = flyTweetsBufferConfig_();
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = spreadsheet.getSheetByName(config.sheetName);
  if (!sheet) throw new Error(`Sheet not found: ${config.sheetName}`);
  const values = sheet.getDataRange().getValues();
  if (values.length < 2) return { submitted: 0, failed: 0 };

  const headers = values[0].map((value) => String(value || '').trim());
  const required = ['idempotency_key', 'text', 'status', 'scheduled_at', 'buffer_post_id', 'last_error'];
  const index = {};
  required.forEach((name) => {
    index[name] = headers.indexOf(name);
    if (index[name] < 0) throw new Error(`Fly Tweets sheet missing column: ${name}`);
  });

  let submitted = 0;
  let failed = 0;
  for (let rowIndex = 1; rowIndex < values.length; rowIndex += 1) {
    const valuesRow = values[rowIndex];
    const row = {};
    headers.forEach((header, columnIndex) => { row[header] = valuesRow[columnIndex]; });
    const status = String(row.status || '').trim();
    const bufferPostId = String(row.buffer_post_id || '').trim();
    if (status !== 'generated' || bufferPostId) continue;

    try {
      const payload = buildBufferCreatePostPayload(row, config.channelId);
      const response = UrlFetchApp.fetch(FLY_TWEETS_BUFFER_ENDPOINT, {
        method: 'post',
        contentType: 'application/json',
        headers: { Authorization: `Bearer ${config.apiKey}` },
        payload: JSON.stringify(payload),
        muteHttpExceptions: true,
      });
      const body = JSON.parse(response.getContentText());
      const created = parseBufferCreatePostResponse(body);
      sheet.getRange(rowIndex + 1, index.buffer_post_id + 1).setValue(created.id);
      sheet.getRange(rowIndex + 1, index.status + 1).setValue('scheduled');
      sheet.getRange(rowIndex + 1, index.last_error + 1).setValue('');
      submitted += 1;
    } catch (error) {
      sheet.getRange(rowIndex + 1, index.last_error + 1).setValue(String(error && error.message || error));
      failed += 1;
    }
  }
  return { submitted, failed };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    FLY_TWEETS_BUFFER_ENDPOINT,
    buildBufferCreatePostPayload,
    parseBufferCreatePostResponse,
  };
}
