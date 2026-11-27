// Offline integration smoke. The source QASM is archived, parameters and
// observations are synthetic. No live submission or credentials.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {fixturePack} from './test_qpu04_drift.mjs';
import {derive} from './qpu04_drift.mjs';
import {buildFrozen,verifyFrozen,canonical,sha256,readLF} from './qc26b_freeze.mjs';
import {buildRows,analyse,analyseQPU04} from './qc26b_prospective_eval.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const results=path.join(here,'../results');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'qpu04-smoke-'));
try{
  const build=path.join(tmp,'build'),paramsPath=path.join(tmp,'params.json');
  fs.mkdirSync(path.join(build,'circuits'),{recursive:true});
  const src=JSON.parse(readLF(path.join(results,'hw_root/compiled/compiled_nc_eval.json')));
  const rows=[41,42].map(seed=>src.circuits.find(r=>r.n===8&&r.depth===1&&r.seed===seed));
  assert.ok(rows.every(Boolean));
  const meta={...src,circuits:rows,compiled_at:'2026-11-26T00:20:00Z',calibration_timestamp:'2026-11-26T00:10:00Z',
    plan_sha256:'0'.repeat(64),mode:'synthetic-test'};
  fs.writeFileSync(path.join(build,'compiled.json'),JSON.stringify(meta));
  fs.copyFileSync(path.join(results,'hw_root/compiled/calibration_snapshot_nc_eval.json'),path.join(build,'calibration_snapshot.json'));
  for(const r of rows){
    const q=readLF(path.join(results,'ibm_run_nc/circuits',r.id+'.qasm'));
    assert.equal(sha256(q),r.qasm3_sha256);
    fs.writeFileSync(path.join(build,'circuits',r.id+'.qasm'),q);
  }
  const pack=fixturePack(),fit=derive(pack,meta.compiled_at,meta.calibration_timestamp);
  const params={...fit.latest,meta:{pgc_qpu04:pack}};
  fs.writeFileSync(paramsPath,JSON.stringify(params));
  const frozen=buildFrozen(build,paramsPath);
  const result=verifyFrozen(frozen,build,paramsPath);
  assert.ok(result.ok,JSON.stringify(result.failures));
  function reject(change){
    const x=structuredClone(frozen);change(x);delete x.sha256;x.sha256=sha256(canonical(x));
    assert.equal(verifyFrozen(x,build,paramsPath).ok,false);
  }
  reject(x=>{x.qpu04.predictions[rows[0].id][0].C3ZD+=.001});
  reject(x=>{x.qpu04.predictions[rows[0].id][0].ledger.min_bound+=.001});
  reject(x=>{x.qpu04.fit.posterior['sx.3'].gain+=.001});
  reject(x=>{delete x.qpu04});
  const measured=Object.fromEntries(Object.entries(frozen.qpu04.predictions).map(([id,a])=>[id,a.map(x=>x.C3ZD*.98)]));
  const obs=buildRows(frozen,measured);
  const original=analyse(frozen,obs),ext=analyseQPU04(frozen,obs);
  assert.deepEqual(Object.keys(original.candidates),['C3','C2','C3Z']);
  assert.ok(ext&&Number.isFinite(ext.mae.all.C3ZD));
  const baseParams={...fit.latest};
  fs.writeFileSync(paramsPath,JSON.stringify(baseParams));
  const baseline=buildFrozen(build,paramsPath);
  assert.deepEqual(baseline.predictions,frozen.predictions);
  assert.equal(baseline.qpu04,undefined);
  assert.deepEqual(analyse(baseline,buildRows(baseline,measured)),original);
  console.log('QPU04 integrated freeze, four forgery checks, original verdict invariance: PASS (synthetic)');
}finally{fs.rmSync(tmp,{recursive:true,force:true})}
