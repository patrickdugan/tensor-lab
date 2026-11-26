// QC-26B prospective protocol: freeze and verify predictions BEFORE any fresh counts exist.
//   node qc26b_freeze.mjs freeze BUILD_DIR PARAMS.json OUT_frozen.json
//   node qc26b_freeze.mjs verify BUILD_DIR PARAMS.json frozen.json        (exit 0 only if everything recomputes)
// BUILD_DIR holds compiled.json, circuits/<id>.qasm and calibration_snapshot.json (qc26b_prospective_plan.py build).
// PARAMS.json: {sx:{phys:rad}, cz:{"a_b":{delta:rad, alpha:{phys:rad}}}} (a<b physical qubits; optional "meta" is hashed, not used).
// Models: C0 twirled stochastic, C1 + non-unital relaxation, C2 + SX offsets, C3 + CZ coherent ZZ phases (params.cz.*.delta only,
// endpoint terms alpha forced to 0), and the DECLARED EXTENSION C3Z = C3 + endpoint-local Z terms (params.cz.*.alpha). qc26b_engine.mjs
// is unchanged: C3 is obtained by stripping alpha from the cz config before it reaches the engine.
// Format qc26b-frozen-v2 stores C3Z; legacy qc26b-frozen-v1 files (no C3Z, C3 used alpha as given in params) still verify.
// Offline. No network, no credentials. All text is hashed after CRLF->LF normalisation so Windows checkouts agree.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parse,schedule,evaluate} from './qc26b_engine.mjs';

export const FORMAT_V1='qc26b-frozen-v1',FORMAT='qc26b-frozen-v2';
export const TOL=1e-12;
export const MODELS_V1=['C0','C1','C2','C3'];
export const MODELS=['C0','C1','C2','C3','C3Z'];
export const modelsFor=format=>format===FORMAT_V1?MODELS_V1:MODELS;
const HERE=path.dirname(fileURLToPath(import.meta.url));
export const sha256=s=>crypto.createHash('sha256').update(s).digest('hex');
export const lf=s=>s.replace(/\r\n/g,'\n');
export const readLF=p=>lf(fs.readFileSync(p,'utf8'));
export function canonical(x){
  if(x===null)return 'null';
  if(Array.isArray(x))return '['+x.map(canonical).join(',')+']';
  switch(typeof x){
    case 'number':if(!Number.isFinite(x))throw Error('non-finite number in canonical JSON');return JSON.stringify(x);
    case 'string':case 'boolean':return JSON.stringify(x);
    case 'object':return '{'+Object.keys(x).sort().filter(k=>x[k]!==undefined).map(k=>JSON.stringify(k)+':'+canonical(x[k])).join(',')+'}';
    default:throw Error('unsupported type '+typeof x);
  }
}
export const engineSha=()=>sha256(readLF(path.join(HERE,'qc26b_engine.mjs')));

export function checkParams(p){
  if(!p||typeof p!=='object'||Array.isArray(p))throw Error('params must be an object');
  for(const k of Object.keys(p))if(!['sx','cz','meta'].includes(k))throw Error('unknown params key '+k);
  const num=(v,w)=>{if(typeof v!=='number'||!Number.isFinite(v))throw Error('params: non-numeric '+w);};
  for(const [k,v] of Object.entries(p.sx||{})){if(!/^\d+$/.test(k))throw Error('params.sx key '+k);num(v,'sx.'+k);}
  for(const [k,e] of Object.entries(p.cz||{})){
    const m=k.match(/^(\d+)_(\d+)$/);if(!m||+m[1]>=+m[2])throw Error('params.cz key must be a_b with a<b: '+k);
    for(const f of Object.keys(e))if(!['delta','alpha'].includes(f))throw Error('params.cz.'+k+' key '+f);
    if(e.delta!==undefined)num(e.delta,'cz.'+k+'.delta');
    for(const [q,v] of Object.entries(e.alpha||{})){if(!/^\d+$/.test(q))throw Error('alpha key '+q);num(v,'cz.'+k+'.alpha.'+q);}
  }
  return p;
}
export const readParams=file=>checkParams(JSON.parse(readLF(file)));
// C3 uses delta only (the original comparator); C3Z adds the endpoint Z terms alpha. legacyV1 reproduces the v1 definition of C3 (alpha as given).
export const deltaOnly=cz=>Object.fromEntries(Object.entries(cz||{}).map(([k,e])=>[k,{delta:e.delta||0}]));
export function configs(p,legacyV1=false){
  const sx=p.sx||{},cz=p.cz||{};
  if(legacyV1)return {C0:{nonunital:false},C1:{nonunital:true},C2:{nonunital:true,sx},C3:{nonunital:true,sx,cz}};
  return {C0:{nonunital:false},C1:{nonunital:true},C2:{nonunital:true,sx},C3:{nonunital:true,sx,cz:deltaOnly(cz)},C3Z:{nonunital:true,sx,cz}};
}

export function loadBuild(dir){
  const compiledText=readLF(path.join(dir,'compiled.json')),snapText=readLF(path.join(dir,'calibration_snapshot.json'));
  const meta=JSON.parse(compiledText),snap=JSON.parse(snapText),circuits=[];
  for(const r of meta.circuits){
    const q=readLF(path.join(dir,'circuits',r.id+'.qasm'));
    circuits.push({r,q,qasm_sha256:sha256(q)});
  }
  return {meta,snap,circuits,compiled_json_sha256:sha256(compiledText),snapshot_sha256:sha256(snapText)};
}

// Predictions for every circuit and every measured site (bit k <-> physical_path[k]).
export function computePredictions(build,params,format=FORMAT){
  const cfg=configs(params,format===FORMAT_V1),models=modelsFor(format),preds={};
  for(const {r,q} of build.circuits){
    const ops=parse(r,q),seq=schedule(r,ops,build.snap),n=r.n;
    const ro=r.physical_path.map(p=>build.snap.qubits[p].readout_error||0);
    const idealOps=ops.filter(x=>x.name!=='barrier'),sites=[];
    for(let k=0;k<n;k++){
      const row={bit:k,physical:r.physical_path[k],ideal:evaluate(n,idealOps,k,0,{}).z,ledger:{}};
      for(const m of models){
        const v=evaluate(n,seq,k,ro[k],cfg[m]);
        row[m]=v.z;
        row.ledger[m]={l1:v.ledger.l1,old_bound:v.ledger.old_bound,min_bound:v.ledger.min_bound,peak:v.ledger.peak};
      }
      sites.push(row);
    }
    preds[r.id]=sites;
  }
  return preds;
}

export function buildFrozen(buildDir,paramsFile){
  const params=readParams(paramsFile),build=loadBuild(buildDir);
  const circuits={};
  for(const c of build.circuits){
    const r=c.r;
    if(sha256(c.q)!==r.qasm3_sha256)throw Error('QASM hash differs from compiled.json: '+r.id);
    circuits[r.id]={qasm_sha256:c.qasm_sha256,physical_path:r.physical_path,n:r.n,depth:r.depth,seed:r.seed,shots:r.shots,path_class:r.path_class||null};
  }
  const body={
    format:FORMAT,
    label:'QC-26B frozen prospective predictions. Written before any fresh counts exist; evaluate with qc26b_prospective_eval.mjs.',
    models:{C0:'gate-local twirled stochastic noise',C1:'C0 + non-unital T1/T2 relaxation',C2:'C1 + coherent SX angle offsets (params.sx)',C3:'C2 + coherent CZ ZZ phases (params.cz.*.delta only; alpha ignored)',C3Z:'DECLARED EXTENSION, not the primary comparator: C3 + endpoint-local Z terms (params.cz.*.alpha)'},
    params,params_sha256:sha256(canonical(params)),
    build:{backend:build.meta.backend,plan_sha256:build.meta.plan_sha256,calibration_timestamp:build.meta.calibration_timestamp,
      compiled_json_sha256:build.compiled_json_sha256,calibration_snapshot_sha256:build.snapshot_sha256,circuits},
    code:{engine_file:'qc26b_engine.mjs',engine_sha256:engineSha()},
    tolerance:TOL,
    predictions:computePredictions(build,params)
  };
  return {...body,sha256:sha256(canonical(body))};
}

// Returns {ok, failures[], checks[]}; never throws on a mismatch.
export function verifyFrozen(frozen,buildDir,paramsFile){
  const failures=[],checks=[];
  const chk=(name,ok,detail)=>{checks.push({name,ok:!!ok,detail:detail??null});if(!ok)failures.push(name+(detail?': '+detail:''));};
  try{
    const {sha256:top,...body}=frozen;
    chk('format',frozen.format===FORMAT||frozen.format===FORMAT_V1,frozen.format);
    const models=modelsFor(frozen.format);
    chk('frozen self-hash',sha256(canonical(body))===top,'top-level sha256 does not match contents');
    const params=readParams(paramsFile);
    chk('parameter file equals frozen parameters',canonical(params)===canonical(frozen.params),'parameter file differs from the frozen parameters');
    chk('params_sha256',sha256(canonical(params))===frozen.params_sha256);
    chk('engine code hash',engineSha()===frozen.code.engine_sha256,'qc26b_engine.mjs differs from the frozen version');
    const build=loadBuild(buildDir);
    chk('compiled.json hash',build.compiled_json_sha256===frozen.build.compiled_json_sha256);
    chk('plan_sha256',build.meta.plan_sha256===frozen.build.plan_sha256);
    chk('calibration snapshot hash',build.snapshot_sha256===frozen.build.calibration_snapshot_sha256);
    const ids=build.circuits.map(c=>c.r.id).sort(),fids=Object.keys(frozen.build.circuits).sort();
    chk('circuit id set',JSON.stringify(ids)===JSON.stringify(fids));
    for(const c of build.circuits){
      const f=frozen.build.circuits[c.r.id];
      if(!f)continue;
      if(c.qasm_sha256!==f.qasm_sha256||c.qasm_sha256!==c.r.qasm3_sha256)chk('qasm hash '+c.r.id,false,'QASM differs from the frozen hash');
      if(JSON.stringify(c.r.physical_path)!==JSON.stringify(f.physical_path))chk('path '+c.r.id,false);
    }
    if(!failures.length){
      const preds=computePredictions(build,params,frozen.format);
      let maxGap=0,bad=0,cnt=0;
      for(const id of ids){
        const a=preds[id],b=frozen.predictions[id];
        if(!b||a.length!==b.length){bad++;continue;}
        for(let i=0;i<a.length;i++)for(const k of ['ideal',...models]){
          const g=Math.abs(a[i][k]-b[i][k]);cnt++;
          if(!(g<=TOL)){bad++;}
          if(g>maxGap||Number.isNaN(g))maxGap=g;
        }
      }
      chk('predictions recompute within 1e-12',bad===0,`${bad} of ${cnt} values differ, max gap ${maxGap}`);
      checks.push({name:'max recompute gap',ok:true,detail:maxGap});
    }
  }catch(e){chk('verification error',false,String(e&&e.message||e));}
  return {ok:failures.length===0,failures,checks};
}

function main(){
  const [cmd,a,b,c]=process.argv.slice(2);
  if(cmd==='freeze'&&a&&b&&c){
    const f=buildFrozen(a,b);
    fs.writeFileSync(c,JSON.stringify(f,null,1)+'\n');
    const nsite=Object.values(f.predictions).reduce((s,x)=>s+x.length,0);
    console.log(JSON.stringify({frozen:c,sha256:f.sha256,params_sha256:f.params_sha256,engine_sha256:f.code.engine_sha256,plan_sha256:f.build.plan_sha256,circuits:Object.keys(f.predictions).length,sites:nsite}));
  }else if(cmd==='verify'&&a&&b&&c){
    const r=verifyFrozen(JSON.parse(readLF(c)),a,b);
    console.log(JSON.stringify({ok:r.ok,failures:r.failures}));
    process.exit(r.ok?0:2);
  }else{console.error('usage: freeze BUILD PARAMS OUT | verify BUILD PARAMS FROZEN');process.exit(64);}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main();
