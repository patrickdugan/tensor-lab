// Independent manuscript audit. No simulator, IBM API or network access.
// Reconstruct Z expectations directly from archived counts and compare frozen rows.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'../../experiments/channel_mpo/ibm_qc23');
const base=path.join(root,'pgc_qpu/receipts/qpu04_live_20261127');
const read=p=>fs.readFileSync(p,'utf8').replace(/\r\n/g,'\n');
const json=p=>JSON.parse(read(p));
const hash=s=>createHash('sha256').update(s).digest('hex');
function canonical(x){
 if(x===null||typeof x!=='object')return JSON.stringify(x);
 if(Array.isArray(x))return '['+x.map(canonical).join(',')+']';
 return '{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canonical(x[k])).join(',')+'}';
}
const near=(a,b,message)=>assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-12,message+': '+a+' vs '+b);
const f=json(path.join(base,'frozen.json')),report=json(path.join(base,'qpu04_report.json'));
const {sha256,...body}=f;
assert.equal(hash(canonical(body)),sha256);
assert.equal(sha256,'da7bcb562c942d3715360260851b91e52ed6ce3fd80d2df8dbd30580bf39a99c');
assert.equal(canonical(f.params),canonical(json(path.join(base,'params.json'))));
for(const [file,expected] of Object.entries(f.code.manifest))
 assert.equal(hash(read(path.join(root,'qc26',file))),expected,'frozen code hash '+file);
const jobs=[],rows=[],models=['ideal','C0','C1','C2','C3','C3Z','C3ZD'];
for(const block of ['qc26a','q5','q23','eval']){
 const dir=path.join(base,block),m=json(path.join(dir,'compiled.json')),c=json(path.join(dir,'counts.json')),j=json(path.join(dir,'job.json')),r=json(path.join(dir,'execution_receipt.json'));
 assert.equal(j.plan_sha256,m.plan_sha256);
 assert.equal(r.plan_sha256,m.plan_sha256);
 assert.equal(r.job_id,j.job_id);
 assert.equal(Object.keys(c).length,m.circuits.length);
 let shots=0;
 for(const v of m.circuits){
  const e=c[v.id],counts=block==='eval'?e.counts:e,n=v.n??v.path.length;
  if(block==='eval')assert.deepEqual(e.physical_path,v.physical_path);
  let total=0;const z=Array(n).fill(0);
  for(const [bits,k] of Object.entries(counts)){
   assert.equal(bits.length,n);assert.match(bits,/^[01]+$/);assert.ok(Number.isInteger(k)&&k>=0);
   total+=k;for(let q=0;q<n;q++)z[q]+=k*(bits[n-1-q]==='1'?-1:1);
  }
  assert.equal(total,v.shots);shots+=total;
  if(block!=='eval')continue;
  assert.equal(hash(read(path.join(dir,'circuits',v.id+'.qasm'))),v.qasm3_sha256);
  assert.equal(f.build.circuits[v.id].qasm_sha256,v.qasm3_sha256);
  for(let q=0;q<n;q++){
   const s=f.predictions[v.id][q],d=f.qpu04.predictions[v.id][q];
   assert.equal(s.physical,v.physical_path[q]);assert.equal(s.bit,q);assert.equal(d.physical,s.physical);
   const values=Object.fromEntries(models.map(k=>[k,k==='C3ZD'?d.C3ZD:s[k]]));
   const archived=report.rows.find(x=>x.id===v.id&&x.physical===s.physical);
   near(z[q]/total,archived.measured,'measurement '+v.id+' '+q);
   for(const k of models)near(values[k],archived[k],'prediction '+v.id+' '+q+' '+k);
   rows.push({physical:s.physical,id:v.id,measured:z[q]/total,...values});
  }
 }
 if(block==='eval'){
  assert.equal(j.frozen_sha256,sha256);assert.equal(r.frozen_sha256,sha256);
  assert.equal(j.freeze_commit,'1f120d1250d49d6c91aa21b12215e6a90ff6564a');
  assert.equal(f.build.plan_sha256,j.plan_sha256);
  assert.equal(hash(read(path.join(dir,'compiled.json'))),f.build.compiled_json_sha256);
  assert.equal(hash(read(path.join(dir,'calibration_snapshot.json'))),f.build.calibration_snapshot_sha256);
 }
 jobs.push({block,job:j.job_id,circuits:m.circuits.length,shots,provider_seconds:r.job_usage});
}
assert.equal(jobs.reduce((s,j)=>s+j.shots,0),131328);
assert.equal(jobs.reduce((s,j)=>s+j.provider_seconds,0),44);
assert.equal(rows.length,240);assert.equal(report.rows.length,240);
const groups={all:rows,targets:rows.filter(r=>[5,23].includes(r.physical)),q6:rows.filter(r=>r.physical===6)};
const mae=(rs,k)=>rs.reduce((s,r)=>s+Math.abs(r.measured-r[k]),0)/rs.length;
const results=Object.fromEntries(Object.entries(groups).map(([g,rs])=>[g,{n:rs.length,...Object.fromEntries(models.map(k=>[k,mae(rs,k)]))}]));
for(const k of models){
 const a=k==='C3ZD'?report.qpu04.mae.all.C3ZD:report.analysis.groups.all[k];
 const b=k==='C3ZD'?report.qpu04.mae.targets.C3ZD:report.analysis.groups.targets_5_23[k];
 near(results.all[k],a,'pooled '+k);near(results.targets[k],b,'target '+k);
}
assert.equal(results.targets.n,24);assert.equal(results.q6.n,6);
for(const k of ['C2','C3','C3Z'])assert.equal(report.analysis.candidates[k].verdict,'FAIL');
assert.equal(report.qpu04.verdict,'FAIL');
assert.equal(report.qpu04.incremental_vs_static,'INCONCLUSIVE_OR_NEGATIVE');
const figure=['C2','C3','C3Z','C3ZD'].map(k=>({model:k,pooled_pct:100*(1-results.all[k]/results.all.C0),target_pct:100*(1-results.targets[k]/results.targets.C0),site6_delta:results.q6[k]-results.q6.C0}));
console.log(JSON.stringify({label:'Independent receipt and arithmetic audit; no IBM API or forward simulator',jobs,total_shots:131328,total_provider_seconds:44,observables:rows.length,frozen_sha256:sha256,results,figure,verdict:'PASS'},null,2));
