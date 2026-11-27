import fs from 'node:fs';
import assert from 'node:assert/strict';
const [expected,actual]=process.argv.slice(2).map(p=>JSON.parse(fs.readFileSync(p,'utf8')));
assert.equal(actual.equivalence.action_mismatches,0);
assert.equal(actual.equivalence.paths,96);
assert.ok(actual.equivalence.max_score_gap<1e-10);
assert.ok(actual.equivalence.posterior_max_gap<1e-10);
assert.equal(actual.records.length,expected.records.length);
for(let i=0;i<expected.records.length;i++){
  const a=actual.records[i],b=expected.records[i];
  for(const k of Object.keys(b)){
    if(typeof b[k]==='number') assert.ok(Math.abs(a[k]-b[k])<1e-9*Math.max(1,Math.abs(b[k])),k+' trial '+i);
    else assert.equal(a[k],b[k]);
  }
}
for(const scenario of actual.scenarios)for(const budget of actual.budgets){
  const find=policy=>actual.summary.find(r=>r.scenario===scenario&&r.budget===budget&&r.policy===policy);
  const a=find('dense_eig'),b=find('group_exact');
  assert.equal(a.rmse,b.rmse);
  assert.ok(a.score_work/b.score_work>10);
}
console.log('BPD-QPU02 pilot regression passed');
