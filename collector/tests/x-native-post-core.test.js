'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {pickDue,expireMissed,verifyTweetUrl,normalizeText,lock,readQueue,writeAtomic}=require('../x-native-post-core');

function entry(id,due,status='ready',extra={}){
 return {id, source_idempotency_key:id, text:'This fly has exactly one point.', status,
   generator:'fly_compositional_style_v2', origin:'fly_cadence',
   scheduled_at:new Date(due).toISOString(),posted_at:'',...extra};
}
test('due cannot trigger before the selected fly time',()=>{
 const now=new Date('2026-11-02T06:30:00Z');
 assert.equal(pickDue([entry('later','2026-11-02T08:30:00Z')],now),null);
 assert.equal(pickDue([entry('due','2026-11-02T06:20:00Z')],now).id,'due');
});
test('recent published tweet enforces twenty-minute minimum',()=>{
 const now=new Date('2026-11-02T06:30:00Z');
 const items=[entry('due','2026-11-02T06:10:00Z'),
              entry('sent','2026-11-02T04:00:00Z','published',{posted_at:'2026-11-02T06:25:00Z'})];
 assert.equal(pickDue(items,now),null);
 assert.equal(pickDue(items,new Date('2026-11-02T06:47:00Z')).id,'due');
});
test('missed schedules are expired; smoke can still be posted',()=>{
 const now=new Date('2026-11-02T07:00:00Z');
 const rows=[entry('old','2026-11-01T10:00:00Z'),
             entry('smoke','2026-11-01T10:00:00Z','ready',{origin:'manual_smoke'})];
 assert.equal(expireMissed(rows,now),1);
 assert.equal(rows[0].status,'missed');
 assert.equal(pickDue(rows,now).id,'smoke');
});
test('reliable permalink must point to authorized account',()=>{
 assert.equal(verifyTweetUrl('https://x.com/fly_topology/status/12345','fly_topology'),true);
 assert.equal(verifyTweetUrl('https://x.com/not_fly_topology/status/12345','fly_topology'),false);
 assert.equal(verifyTweetUrl('https://evil.tld/fly_topology/status/12345','fly_topology'),false);
 assert.equal(verifyTweetUrl('https://x.com/fly_topology/status/notanid','fly_topology'),false);
});
test('atomic file and exclusive lock prevent duplicate jobs',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'fly-x-'));
 try{
  const p=path.join(dir,'queue.json');
  writeAtomic(p,[entry('x','2026-11-02T06:30:00Z')]);
  assert.equal(readQueue(p).length,1);
  const unlock=lock(p+'.lock');
  try{assert.throws(()=>lock(p+'.lock'),/owns queue lock/);}finally{unlock()}
  const unlock2=lock(p+'.lock');unlock2();
  assert.equal(normalizeText(' A   fly\nexists '),'A fly exists');
 }finally{fs.rmSync(dir,{recursive:true,force:true})}
});
test('invalid or duplicate queue records are rejected',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'fly-x-'));
 try{
  const p=path.join(dir,'queue.json');
  fs.writeFileSync(p,JSON.stringify([entry('x','2026-11-02T06:30:00Z'),entry('x','2026-11-02T06:30:00Z')]));
  assert.throws(()=>readQueue(p),/unique/);
 }finally{fs.rmSync(dir,{recursive:true,force:true})}
});
