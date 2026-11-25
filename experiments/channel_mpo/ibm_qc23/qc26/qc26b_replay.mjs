// QC-26B retrospective replay of all 528 existing Kingston observables with channel models C0..C3.
// Offline only; zero QPU. RETROSPECTIVE: every fitted parameter below is fitted on previously seen counts.
// Usage: node qc26b_replay.mjs OUT.json [--params independent_params.json]
//   --params supplies INDEPENDENTLY calibrated sx/cz parameters (from QC-26A hardware) and labels the run 'independent'.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {parse,schedule,evaluate,czKey} from './qc26b_engine.mjs';
const root=new URL('../results/',import.meta.url);
const rd=p=>fs.readFileSync(new URL(p,root),'utf8');
const j=p=>JSON.parse(rd(p));
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const qc25=JSON.parse(fs.readFileSync(new URL('../qc25/qc25c_fit.json',import.meta.url),'utf8'));
const argv=process.argv.slice(2),out=argv[0]||'qc26b_replay.json';
const pi=argv.indexOf('--params'),indep=pi>=0?JSON.parse(fs.readFileSync(argv[pi+1],'utf8')):null;

function loadSet(name){
  if(name==='nc_eval'){
    const m=j('hw_root/compiled/compiled_nc_eval.json'),snap=j('hw_root/compiled/calibration_snapshot_nc_eval.json'),raw=j('hw_root/ibm/nc_eval_counts.json');
    return m.circuits.map(r=>{const q=rd('ibm_run_nc/circuits/'+r.id+'.qasm');if(sha(q)!==r.qasm3_sha256)throw Error('QASM hash mismatch '+r.id);return {r,q,snap,raw:raw[r.id],set:name};});
  }
  const m=j('hw_root/compiled/compiled_eval_echo.json'),snap=j('hw_root/compiled/calibration_snapshot.json'),raw=j('hw_root/ibm/eval_echo_counts.json');
  return m.circuits.map(r=>{const q=rd('hw_root/compiled/circuits/'+r.id+'.qasm');if(sha(q)!==r.qasm3_sha256)throw Error('QASM hash mismatch '+r.id);return {r,q,snap,raw:raw[r.id],set:name};});
}
function prep(e){
  const {r,q,snap,raw}=e,ops=parse(r,q),seq=schedule(r,ops,snap);
  if(JSON.stringify(raw.physical_path)!==JSON.stringify(r.physical_path))throw Error('path '+r.id);
  const n=r.n,m=Array(n).fill(0);let shots=0;
  for(const [bits,c] of Object.entries(raw.counts)){if(bits.length!==n||!/^[01]+$/.test(bits))throw Error('bits');shots+=c;for(let k=0;k<n;k++)m[k]+=(bits[n-1-k]==='1'?-1:1)*c;}
  if(shots!==r.shots)throw Error('shots '+r.id);
  const czEdges=[...new Set(ops.filter(o=>o.name==='cz').map(o=>czKey(o.p[0],o.p[1])))];
  return {...e,ops,seq,n,m:m.map(x=>x/shots),ro:r.physical_path.map(p=>snap.qubits[p].readout_error||0),czEdges};
}
const circuits=[...loadSet('nc_eval'),...loadSet('eval_echo')].map(prep);
const TARGETS=[5,23];
const isFit=o=>o.set==='nc_eval'&&o.n===8&&o.r.depth<=2;     // QC-25C fit circuits (width 8, depth 1-2)
const fitSet=circuits.filter(isFit);
function targetLoss(set,cfg){
  const rows=[];
  for(const o of set)for(let q=0;q<o.n;q++){if(!TARGETS.includes(o.r.physical_path[q]))continue;rows.push((evaluate(o.n,o.seq,q,o.ro[q],cfg).z-o.m[q])**2);}
  return mean(rows);
}
// Retrospective coordinate-search fit of conditional ZZ phases on edges touching the suspect sites, fit circuits only.
function fitDelta(sxOffsets){
  const edges=[...new Set(fitSet.flatMap(o=>o.czEdges).filter(k=>k.split('_').some(s=>TARGETS.includes(+s))))].sort();
  const mk=x=>({nonunital:true,sx:sxOffsets,cz:Object.fromEntries(edges.map((k,i)=>[k,{delta:x[i]}]))});
  let x=edges.map(()=>0),v=targetLoss(fitSet,mk(x)),evals=1;const v0=v;
  for(const step of [.025,.0125,.00625,.003125,.0015625,.00078125,.000390625]){
    for(let pass=0;pass<30;pass++){
      let improved=false;
      for(let i=0;i<x.length;i++){
        let best=x,bv=v;
        for(const sg of [-1,1]){const y=x.slice();y[i]+=sg*step;if(Math.abs(y[i])>.100000001)continue;const w=targetLoss(fitSet,mk(y));evals++;if(w<bv-1e-14){best=y;bv=w;}}
        if(bv<v-1e-14){x=best;v=bv;improved=true;}
      }
      if(!improved)break;
    }
  }
  return {edges,delta:Object.fromEntries(edges.map((k,i)=>[k,x[i]])),target_mse_before:v0,target_mse_after:v,evals,bound_rad:.1,at_bound:edges.filter((k,i)=>Math.abs(x[i])>=.0999)};
}
let sxOffsets,czCfg,label;
if(indep){sxOffsets=indep.sx;czCfg=indep.cz;label='independent: parameters from separately acquired calibration; evaluation on these counts is retrospective only if those counts were not used in calibration';}
else{sxOffsets=qc25.fit.nonunital.offsets;const f=fitDelta(sxOffsets);czCfg=Object.fromEntries(Object.entries(f.delta).map(([k,d])=>[k,{delta:d}]));label='retrospective: SX offsets from QC-25C fit, ZZ phases fitted here, both on width-8 depth 1-2 nc circuits only';var fitReport=f;}
const CFG={C0:{nonunital:false},C1:{nonunital:true},C2:{nonunital:true,sx:sxOffsets},C3:{nonunital:true,sx:sxOffsets,cz:czCfg}};
const rows=[];let maxL1=0,maxOld=0,maxMin=0,peak=0;
for(const o of circuits)for(let q=0;q<o.n;q++){
  const row={set:o.set,id:o.r.id,fit:isFit(o),n:o.n,depth:o.r.depth,seed:o.r.seed,bit:q,physical:o.r.physical_path[q],measured:o.m[q]};
  for(const [name,cfg] of Object.entries(CFG)){const v=evaluate(o.n,o.seq,q,o.ro[q],cfg);row[name]=v.z;maxL1=Math.max(maxL1,v.ledger.l1);maxOld=Math.max(maxOld,v.ledger.old_bound);maxMin=Math.max(maxMin,v.ledger.min_bound);peak=Math.max(peak,v.ledger.peak);}
  row.ideal=evaluate(o.n,o.ops.filter(x=>x.name!=='barrier').map(x=>x),q,0,{}).z;
  rows.push(row);
}
// Cross-verification of C0/C1/C2 against the committed QC-25C receipts (independent earlier implementation).
let crossMax=0,crossRows=0;
for(const f of ['qc25c_transfer_8.json','qc25c_transfer_12_shallow.json','qc25c_transfer_12_deep.json','qc25c_transfer_16.json']){
  const rec=JSON.parse(fs.readFileSync(new URL('../qc25/'+f,import.meta.url),'utf8'));
  for(const p of rec.rows){const r=rows.find(x=>x.id===p.id&&x.bit===p.bit&&x.set==='nc_eval');if(!r)throw Error('missing '+p.id);crossRows++;
    if(!indep)crossMax=Math.max(crossMax,Math.abs(r.C0-p.twirled),Math.abs(r.C1-p.nonunital),Math.abs(r.C2-p.nonunital_sx));}
}
const names=['ideal','C0','C1','C2','C3'];
const within=(rs,k)=>rs.filter(r=>Math.abs(r[k]-r.measured)<=.05).length;
const met=rs=>Object.fromEntries(names.map(k=>[k,{mae:mean(rs.map(r=>Math.abs(r[k]-r.measured))),within:within(rs,k),n:rs.length}]));
const isT=r=>TARGETS.includes(r.physical);
const subsets={
  all_528:rows,fit_circuits:rows.filter(r=>r.fit),nc_transfer:rows.filter(r=>r.set==='nc_eval'&&!r.fit),echo_cross_day:rows.filter(r=>r.set==='eval_echo')};
const metrics={};
for(const [s,rs] of Object.entries(subsets))metrics[s]={all:met(rs),target_5_23:met(rs.filter(isT)),other_sites:met(rs.filter(r=>!isT(r)))};
// Paired circuit bootstrap of MAE improvement relative to C0 (equal circuit weights, fixed seed).
let seed=26112501;const rnd=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
function paired(rs,k,B=20000){
  const byC=new Map();for(const r of rs){const e=byC.get(r.id)||[];e.push(r);byC.set(r.id,e);}
  const per=[...byC.values()].map(a=>({d:mean(a.map(r=>Math.abs(r.C0-r.measured)-Math.abs(r[k]-r.measured))),b:mean(a.map(r=>Math.abs(r.C0-r.measured)))}));
  const est=mean(per.map(x=>x.d)),bs=[];
  for(let i=0;i<B;i++){let s=0;for(let t=0;t<per.length;t++)s+=per[Math.floor(rnd()*per.length)].d;bs.push(s/per.length);}
  bs.sort((a,b)=>a-b);
  return {circuits:per.length,mean_improvement:est,ci95:[bs[Math.floor(.025*B)],bs[Math.floor(.975*B)]],wins:per.filter(x=>x.d>0).length,relative_to_C0_mae:est/mean(per.map(x=>x.b))};
}
const pairedOut={};
for(const [s,rs] of Object.entries(subsets)){if(s==='fit_circuits')continue;pairedOut[s]={};for(const k of ['C1','C2','C3']){pairedOut[s][k]={all:paired(rs,k),target_5_23:paired(rs.filter(isT),k)};}}
// Acceptance-style bookkeeping (RETROSPECTIVE; the prospective test is a separate, frozen, fresh-count experiment).
const acc={};
for(const s of ['nc_transfer','echo_cross_day','all_528'])acc[s]={};
for(const s of Object.keys(acc))for(const k of ['C1','C2','C3']){
  const m=metrics[s];const rel=(m.all.C0.mae-m.all[k].mae)/m.all.C0.mae;
  acc[s][k]={relative_mae_improvement_all:rel,relative_mae_improvement_target:(m.target_5_23.C0.mae-m.target_5_23[k].mae)/m.target_5_23.C0.mae,
    other_sites_mae_change:m.other_sites[k].mae-m.other_sites.C0.mae,other_degradation_exceeds_0p005:(m.other_sites[k].mae-m.other_sites.C0.mae)>.005,
    paired_ci_lower_positive:pairedOut[s]?.[k]?.all.ci95[0]>0,meets_10pct:rel>=.10};
}
const result={label:'QC-26B retrospective replay; zero QPU; '+label,provenance:indep?'independent-parameter run':'RETROSPECTIVE fitted parameters (not independent)',
  sx_offsets_rad:sxOffsets,cz_phases:czCfg,fit_report:typeof fitReport==='undefined'?null:fitReport,
  engine_verification:{c0_c1_c2_vs_qc25c_receipts:{rows:crossRows,max_abs_gap:crossMax}},
  ledger:{max_l1_discarded:maxL1,max_old_bound_2pow_n2_sum_eta:maxOld,max_min_bound:maxMin,max_live_terms:peak,note:'l1 bound is the discarded operator trace-norm-free bound sum|c|, valid because each Pauli has operator norm 1; old bound is 2^(n/2) sum_j eta_j'},
  counts:{circuits:circuits.length,observables:rows.length},metrics,paired_bootstrap_vs_C0:pairedOut,acceptance_retrospective:acc,rows};
fs.writeFileSync(out,JSON.stringify(result));
console.log(JSON.stringify({label:result.label,fit:result.fit_report,verify:result.engine_verification,ledger:result.ledger,counts:result.counts,metrics:Object.fromEntries(Object.entries(metrics).map(([s,m])=>[s,Object.fromEntries(names.map(k=>[k,+m.all[k].mae.toFixed(5)]))])),target:Object.fromEntries(Object.entries(metrics).map(([s,m])=>[s,Object.fromEntries(names.map(k=>[k,+m.target_5_23[k].mae.toFixed(5)]))])),acceptance:acc},null,1));
