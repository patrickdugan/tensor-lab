import fs from 'node:fs';
import assert from 'node:assert/strict';
const [e,a]=process.argv.slice(2).map(p=>JSON.parse(fs.readFileSync(p,'utf8')));
function check(x,y,path='root'){
  if(typeof x==='number'){assert.ok(Math.abs(x-y)<=1e-9*Math.max(1,Math.abs(x)),path);return;}
  if(x===null||typeof x!=='object'){assert.equal(x,y,path);return;}
  assert.deepEqual(Object.keys(x),Object.keys(y),path+' keys');
  for(const k of Object.keys(x))check(x[k],y[k],path+'.'+k);
}
check(e,a);
for(const r of a.equivalence)if(r.policy!=='learned_group'){
  assert.equal(r.mismatches,0);
  assert.ok(r.max_posterior_gap<1e-10);
}
assert.ok(a.bounds.max_violation<1e-9);
if(a.version.startsWith('BPD-QPU03B')){
  assert.equal(a.equivalence.filter(x=>x.policy==='refined_group').reduce((s,x)=>s+x.paths,0),120);
  for(const l of a.learning)assert.equal(l.refined_audit.filter(x=>x.fallback).length,l.scenario==='out_of_family'?14:0);
}
console.log(a.version+' replay passed');
