'use strict';
/** Live X UI delivery when Buffer's 30d API quota is exhausted.
 * Requires an authorized persistent browser profile, full account validation,
 * explicit --allow-live, and independent verification of the tweet permalink.
 */
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const {metricsCollectorLaunchOptions, xProfileHrefMatchesExpectedHandle} = require('./metrics-core');
const {readQueue,writeAtomic,normalizeText,validateTweet,pickDue,expireMissed,
       verifyTweetUrl,lock} = require('./x-native-post-core');

function argumentsFrom(argv) {
 const args={};
 for(let i=0;i<argv.length;i++){
  const a=argv[i];
  if(['--allow-live','--dry-run','--reconcile-only'].includes(a)){args[a.slice(2)]=true;continue;}
  if(!['--queue-json','--profile-dir','--expected-handle'].includes(a)) throw new Error('unknown argument '+a);
  args[a.slice(2)] = argv[++i];
  if(!args[a.slice(2)])throw new Error('missing value for '+a);
 }
 if(!args['queue-json']||!args['profile-dir']||!args['expected-handle'])throw new Error('require queue-json profile-dir expected-handle');
 if(args['allow-live']&&args['dry-run'])throw new Error('conflicting modes');
 if(!/^[A-Za-z0-9_]{1,15}$/.test(args['expected-handle'].replace(/^@/,'')))throw new Error('invalid X handle');
 return args;
}

async function assertAccount(page, handle){
 await page.goto('https://x.com/home',{waitUntil:'domcontentloaded',timeout:60000});
 if (/\/(?:i\/flow\/login|login)/i.test(page.url()))throw new Error('X browser logged out');
 const link=page.locator('[data-testid="AppTabBar_Profile_Link"]').first();
 await link.waitFor({state:'attached',timeout:25000});
 if(!xProfileHrefMatchesExpectedHandle(await link.getAttribute('href'),handle)){
  throw new Error('wrong X account: refusing to send any post');
 }
}

async function findPublished(page, handle, targetText){
 await page.goto('https://x.com/'+handle,{waitUntil:'domcontentloaded',timeout:60000});
 await page.locator('article[role="article"],[data-testid="tweet"]').first().waitFor({timeout:25000});
 const wanted=normalizeText(targetText);
 for(let step=0;step<3;step++){
  const rows=await page.locator('article[role="article"],[data-testid="tweet"]').evaluateAll(nodes=>nodes.map(node=>({
    text:(node.querySelector('[data-testid="tweetText"]')?.innerText||'').replace(/\s+/g,' ').trim(),
    urls:[...node.querySelectorAll('a[href*="/status/"]')].map(a=>a.getAttribute('href'))
  })));
  for(const row of rows){
    if(normalizeText(row.text)!==wanted) continue;
    for(const href of row.urls){
      const url=new URL(href,'https://x.com').href;
      if(verifyTweetUrl(url,handle))return url;
    }
  }
  await page.mouse.wheel(0,650);
  await page.waitForTimeout(600);
 }
 return null;
}

async function actuallyPost(page, text) {
 await page.goto('https://x.com/compose/post',{waitUntil:'domcontentloaded',timeout:60000});
 const editor=page.locator('[data-testid="tweetTextarea_0"][contenteditable="true"]').first();
 await editor.waitFor({state:'visible',timeout:25000});
 await editor.fill(text);
 const actual=normalizeText(await editor.innerText());
 if(actual!==normalizeText(text))throw new Error('X composer text differs from intended content');
 const button=page.locator('[data-testid="tweetButton"]').first();
 await button.waitFor({state:'visible',timeout:20000});
 if(!await button.isEnabled())throw new Error('X Post button disabled');
 // NO caller may automatically retry after this click. Network ambiguities
 // must resolve via profile permalink lookup, not another send.
 await button.click({timeout:25000});
}

async function main(){
 const args=argumentsFrom(process.argv.slice(2));
 const queuePath=path.resolve(args['queue-json']);
 const profile=path.resolve(args['profile-dir']);
 const handle=args['expected-handle'].replace(/^@/,'').toLowerCase();
 const unlock=lock(queuePath+'.xrunner.lock');
 try{
  let rows=readQueue(queuePath);
  const changed=expireMissed(rows,new Date());
  if(changed)writeAtomic(queuePath,rows);
  const eligible=rows.filter(row=>row.status==='publishing' || row.status==='uncertain');
  const chosen=args['reconcile-only']?null:pickDue(rows);
  if(!chosen&&!eligible.length){console.log(JSON.stringify({published:0,due:0,expired:changed}));return;}
  if(!args['allow-live']){
   console.log(JSON.stringify({dry_run:true,due:chosen?.id||null,reconcile_pending:eligible.map(x=>x.id)}));
   return;
  }
  if(!fs.existsSync(profile))throw new Error('X persistent browser profile missing');
  const context=await chromium.launchPersistentContext(profile,metricsCollectorLaunchOptions());
  try{
   const page=context.pages()[0]||await context.newPage();
   await assertAccount(page,handle);
   for(const item of eligible){
    const url=await findPublished(page,handle,item.text).catch(()=>null);
    if(!url)continue;
    item.status='published';item.tweet_url=url;item.posted_at=item.posted_at||new Date().toISOString();
    item.last_error='';writeAtomic(queuePath,rows);
    console.log(JSON.stringify({event:'recovered_published',id:item.id,url}));
   }
   const candidate=args['reconcile-only']?null:pickDue(rows);
   if(!candidate){console.log(JSON.stringify({published:0,due:0,reconciled:true}));return;}
   const content=validateTweet(candidate);
   // Check identical post on profile BEFORE we click, not just after.
   const preexisting=await findPublished(page,handle,content);
   if(preexisting){
    candidate.status='published';candidate.tweet_url=preexisting;candidate.posted_at=new Date().toISOString();
    candidate.last_error='matched_existing_x_post';writeAtomic(queuePath,rows);
    console.log(JSON.stringify({event:'already_published',id:candidate.id,url:preexisting}));return;
   }
   candidate.status='publishing';candidate.attempt_started_at=new Date().toISOString();
   candidate.last_error='';
   writeAtomic(queuePath,rows);
   let clickStarted=false;
   try{
    // The editor may fail before post click; only then can this be safely retried.
    await page.goto('https://x.com/compose/post',{waitUntil:'domcontentloaded',timeout:60000});
    const editor=page.locator('[data-testid="tweetTextarea_0"][contenteditable="true"]').first();
    await editor.waitFor({state:'visible',timeout:25000});
    await editor.fill(content);
    if(normalizeText(await editor.innerText())!==content)throw new Error('composer content mismatch');
    const button=page.locator('[data-testid="tweetButton"]').first();
    if(!await button.isEnabled())throw new Error('Post button not enabled');
    clickStarted=true;
    await button.click({timeout:25000});
    await page.waitForTimeout(2500);
    let url=null;
    for(let attempt=0;attempt<3&&!url;attempt++){
      url=await findPublished(page,handle,content).catch(()=>null);
      if(!url)await page.waitForTimeout(2500);
    }
    if(!url)throw new Error('X did not return a verifiable permalink after clicking Post');
    candidate.status='published';candidate.tweet_url=url;candidate.posted_at=new Date().toISOString();
    candidate.last_error='';writeAtomic(queuePath,rows);
    console.log(JSON.stringify({event:'published',id:candidate.id,url,text:content}));
   }catch(error){
    candidate.status=clickStarted?'uncertain':'ready';
    candidate.last_error=String(error.message||error).slice(0,500);
    writeAtomic(queuePath,rows);
    throw new Error('X post '+(clickStarted?'uncertain; automatic retry forbidden':'not submitted')+': '+candidate.last_error);
   }
  }finally{await context.close();}
 }finally{unlock();}
}

if(require.main===module){
 main().catch(err=>{console.error('X_NATIVE_ERROR '+String(err.stack||err).slice(0,1300));process.exitCode=2});
}
module.exports={argumentsFrom,assertAccount,findPublished};
