// QC-26B prospective evaluation of frozen predictions against fresh counts.
//   node qc26b_prospective_eval.mjs --frozen frozen.json --build BUILD_DIR --params params.json --counts counts.json [--out report.json]
// counts.json: {circuit_id:{counts:{bitstring:count},physical_path:[...]}}  (qiskit little-endian bitstrings; bit k <-> physical_path[k]).
// The run REFUSES (exit 2) unless frozen.json verifies: self-hash, parameter file, engine code hash, build hashes, and every
// prediction recomputes from the build dir and parameter file within 1e-12. Scores use the frozen predictions.
// Verdict is evaluated for C3 (primary), C2 (secondary) and, for qc26b-frozen-v2 files, C3Z (DECLARED EXTENSION: C3 plus endpoint-local
// Z terms; never the primary comparator). Legacy qc26b-frozen-v1 files carry no C3Z and are scored exactly as before.
// Negative results are reported as such.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {readLF,sha256,verifyFrozen,MODELS,modelsFor} from './qc26b_freeze.mjs';

export const TARGETS=[5,23],CONTROLS=[3,21];
export const NAMES=['ideal',...MODELS];
export const namesFor=frozen=>['ideal',...modelsFor(frozen.format)];
export const THRESH={rel_improvement:0.10,other_site_degradation:0.005,ledger_fraction_of_mae:0.01};
export const BOOT={resamples:20000,seed:26112501};
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const isT=r=>TARGETS.includes(r.physical),isC=r=>CONTROLS.includes(r.physical);

export function makeRng(seed){let s=seed>>>0;return()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296);}

export function loadCounts(file,frozen){
  const raw=JSON.parse(readLF(file)),ids=Object.keys(frozen.build.circuits).sort();
  if(JSON.stringify(Object.keys(raw).sort())!==JSON.stringify(ids))throw Error('counts.json circuit ids differ from the frozen build');
  const meas={};let synthetic=0;
  for(const id of ids){
    const f=frozen.build.circuits[id],e=raw[id],n=f.n;
    if(JSON.stringify(e.physical_path)!==JSON.stringify(f.physical_path))throw Error('physical_path differs from build for '+id);
    const m=Array(n).fill(0);let shots=0;
    for(const [bits,c] of Object.entries(e.counts)){
      if(bits.length!==n||!/^[01]+$/.test(bits)||!Number.isInteger(c)||c<0)throw Error('malformed counts '+id);
      shots+=c;for(let k=0;k<n;k++)m[k]+=(bits[n-1-k]==='1'?-1:1)*c;
    }
    if(shots!==f.shots)throw Error(`shots ${shots} != ${f.shots} for ${id}`);
    meas[id]=m.map(x=>x/shots);
    if(e.synthetic===true)synthetic++;
  }
  if(synthetic!==0&&synthetic!==ids.length)throw Error('counts mix synthetic and non-synthetic circuits');
  return {meas,synthetic:synthetic===ids.length,raw};
}

export function buildRows(frozen,meas){
  const rows=[];
  for(const [id,sites] of Object.entries(frozen.predictions)){
    const f=frozen.build.circuits[id];
    for(const s of sites){
      const row={id,n:f.n,depth:f.depth,seed:f.seed,path_class:f.path_class,bit:s.bit,physical:s.physical,measured:meas[id][s.bit]};
      for(const k of ['ideal',...modelsFor(frozen.format)])row[k]=s[k];
      row.ledger=s.ledger;rows.push(row);
    }
  }
  return rows;
}

const err=(r,k)=>Math.abs(r[k]-r.measured);
export const mae=(rs,k)=>rs.length?mean(rs.map(r=>err(r,k))):NaN;
const metrics=(rs,names=NAMES)=>({n:rs.length,...Object.fromEntries(names.map(k=>[k,mae(rs,k)]))});

// Paired circuit bootstrap of MAE(base)-MAE(k) (base defaults to C0): circuits are resampled with equal weight; each circuit
// contributes its mean over the selected sites. Fixed seed, 20000 resamples. The C3Z-vs-C3 line uses base='C3'.
export function paired(rs,k,B=BOOT.resamples,seed=BOOT.seed,base='C0'){
  const by=new Map();for(const r of rs){if(!by.has(r.id))by.set(r.id,[]);by.get(r.id).push(r);}
  const per=[...by.values()].map(a=>({d:mean(a.map(r=>err(r,base)-err(r,k))),b:mean(a.map(r=>err(r,base)))}));
  const rnd=makeRng(seed),m=per.length,diffs=[],rels=[];
  for(let i=0;i<B;i++){let d=0,b=0;for(let t=0;t<m;t++){const p=per[Math.floor(rnd()*m)];d+=p.d;b+=p.b;}diffs.push(d/m);rels.push(d/b);}
  diffs.sort((x,y)=>x-y);rels.sort((x,y)=>x-y);
  const D=per.reduce((s,x)=>s+x.d,0),Bs=per.reduce((s,x)=>s+x.b,0);
  return {circuits:m,mean_improvement:D/m,ci95:[diffs[Math.floor(.025*B)],diffs[Math.floor(.975*B)]],
    relative_improvement_equal_circuit_weight:D/Bs,relative_ci95:[rels[Math.floor(.025*B)],rels[Math.floor(.975*B)]],
    wins:per.filter(x=>x.d>0).length,losses:per.filter(x=>x.d<0).length,resamples:B,seed,base};
}

// Sensitivity analysis: resample whole width/seed paths, keeping all depths and
// sites together. Ratio of sums preserves the observable-weighted MAE estimand.
// Eight path clusters are few; treat this interval as a descriptive robustness check.
export function pathClustered(rs,k,B=BOOT.resamples,seed=BOOT.seed,base='C0'){
  const by=new Map();
  for(const r of rs){const key=r.n+':'+r.seed;if(!by.has(key))by.set(key,[]);by.get(key).push(r);}
  const per=[...by.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([path,a])=>({
    path,n_obs:a.length,circuits:new Set(a.map(r=>r.id)).size,
    d:a.reduce((s,r)=>s+err(r,base)-err(r,k),0),b:a.reduce((s,r)=>s+err(r,base),0)
  }));
  if(!per.length)return {clusters:0,mean_improvement:null,ci95:[null,null],per_path:[]};
  const rnd=makeRng(seed),m=per.length,diffs=[];
  for(let i=0;i<B;i++){let d=0,n=0;for(let j=0;j<m;j++){const x=per[Math.floor(rnd()*m)];d+=x.d;n+=x.n_obs;}diffs.push(d/n);}
  diffs.sort((a,b)=>a-b);
  return {clusters:m,mean_improvement:per.reduce((a,x)=>a+x.d,0)/rs.length,
    ci95:[diffs[Math.floor(.025*B)],diffs[Math.floor(.975*B)]],base,resamples:B,seed,
    per_path:per.map(x=>({path:x.path,n_obs:x.n_obs,circuits:x.circuits,mean_improvement:x.d/x.n_obs,
      relative_improvement:x.b?x.d/x.b:null}))};
}

export function siteTable(rows,names=NAMES){
  const out=[];
  for(const p of [...new Set(rows.map(r=>r.physical))].sort((a,b)=>a-b)){
    const rs=rows.filter(r=>r.physical===p),m=metrics(rs,names);
    const d2=m.C2-m.C0,d3=m.C3-m.C0,target=TARGETS.includes(p);
    const row={physical:p,role:target?'target':CONTROLS.includes(p)?'control':'other',n_obs:rs.length,circuits:new Set(rs.map(r=>r.id)).size,
      mae:Object.fromEntries(names.map(k=>[k,m[k]])),change_C2_vs_C0:d2,change_C3_vs_C0:d3,
      flag_C2:!target&&d2>THRESH.other_site_degradation,flag_C3:!target&&d3>THRESH.other_site_degradation};
    if(names.includes('C3Z')){const dz=m.C3Z-m.C0;row.change_C3Z_vs_C0=dz;row.flag_C3Z=!target&&dz>THRESH.other_site_degradation;}
    out.push(row);
  }
  return out;
}

export function ledgerSummary(rows,k,all=['C0','C1','C2','C3']){
  const get=(key,models)=>Math.max(...rows.flatMap(r=>models.map(m=>r.ledger[m][key])));
  const one=[k];
  return {models_considered:all.length===4?'all four models (worst case)':'all models '+all.join(', ')+' (worst case)',max_l1_discard:get('l1',all),max_old_bound:get('old_bound',all),max_min_bound:get('min_bound',all),max_live_terms:get('peak',all),
    candidate_max_l1_discard:get('l1',one),candidate_max_old_bound:get('old_bound',one),candidate_max_min_bound:get('min_bound',one)};
}

export function analyse(frozen,rows){
  const names=namesFor(frozen),hasZ=names.includes('C3Z'),cands=hasZ?['C3','C2','C3Z']:['C3','C2'];
  const T=rows.filter(isT),Ctl=rows.filter(isC),Oth=rows.filter(r=>!isT(r));
  const groups={all:rows,targets_5_23:T,controls_3_21:Ctl,other_than_5_23:Oth,
    ...Object.fromEntries(TARGETS.map(p=>['site_'+p,rows.filter(r=>r.physical===p)]))};
  const classes=[...new Set(rows.map(r=>r.path_class).filter(Boolean))].sort();
  for(const c of classes)groups['class_'+c]=rows.filter(r=>r.path_class===c);
  const mt=Object.fromEntries(Object.entries(groups).map(([g,rs])=>[g,metrics(rs,names)]));
  const sites=siteTable(rows,names),res={};
  for(const k of cands){
    const rel=g=>(mt[g].C0-mt[g][k])/mt[g].C0;
    const pAll=paired(rows,k),pT=paired(T,k);
    const flagged=sites.filter(s=>s['flag_'+k]);
    const L=ledgerSummary(rows,k,k==='C3Z'?['C0','C1','C2','C3','C3Z']:undefined),maeK=mt.all[k];
    const ratio=L.candidate_max_min_bound/maeK;
    const eff=mean(rows.map(r=>Math.abs(r[k]-r.C0)));
    res[k]={
      relative_mae_improvement:{pooled_all_sites:rel('all'),targets_5_23:rel('targets_5_23'),controls_3_21:rel('controls_3_21'),other_than_5_23:rel('other_than_5_23'),
        site_5:rel('site_5'),site_23:rel('site_23')},
      paired_bootstrap_vs_C0:{all_sites:pAll,targets_5_23:pT},
      path_clustered_bootstrap_vs_C0:{all_sites:pathClustered(rows,k),targets_5_23:pathClustered(T,k)},
      other_site_degradation:{pooled_other_sites_change:mt.other_than_5_23[k]-mt.other_than_5_23.C0,flagged_sites:flagged.map(s=>({physical:s.physical,change:s['change_'+k+'_vs_C0'],n_obs:s.n_obs}))},
      numerical_vs_physical:{...L,physical_prediction_error_mae:maeK,mean_abs_model_effect_vs_C0:eff,
        ratio_candidate_min_bound_to_mae:ratio,ratio_candidate_l1_to_mae:L.candidate_max_l1_discard/maeK,ratio_candidate_old_bound_to_mae:L.candidate_max_old_bound/maeK,
        ratio_candidate_min_bound_to_model_effect:L.candidate_max_min_bound/eff,
        definition:`'substantially below' means max min_bound (min of the l1 and 2^(n/2)*sum(eta) bounds, worst case over all sites for ${k}) < ${THRESH.ledger_fraction_of_mae*100}% of the pooled MAE of ${k}`}
    };
    const R=res[k];
    R.criteria={
      a_relative_improvement_ge_10pct:{pass:rel('all')>=THRESH.rel_improvement&&rel('targets_5_23')>=THRESH.rel_improvement,
        detail:`pooled ${(100*rel('all')).toFixed(2)}% and targeted(5,23) ${(100*rel('targets_5_23')).toFixed(2)}% (both must be >= 10%)`},
      b_paired_ci_lower_positive:{pass:pAll.ci95[0]>0&&pT.ci95[0]>0,
        detail:`pooled CI95 [${pAll.ci95[0].toFixed(5)}, ${pAll.ci95[1].toFixed(5)}], targeted CI95 [${pT.ci95[0].toFixed(5)}, ${pT.ci95[1].toFixed(5)}] (both lower bounds must be > 0)`},
      c_no_other_site_degradation_gt_0p005:{pass:flagged.length===0,
        detail:flagged.length?'flagged physical sites: '+flagged.map(s=>`${s.physical} (+${s['change_'+k+'_vs_C0'].toFixed(4)})`).join(', '):`no site other than 5/23 worsens by more than 0.005 (pooled other-site change ${(mt.other_than_5_23[k]-mt.other_than_5_23.C0).toFixed(5)})`},
      d_ledger_bound_below_1pct_of_mae:{pass:ratio<THRESH.ledger_fraction_of_mae,
        detail:`max min_bound ${L.candidate_max_min_bound.toExponential(3)} / MAE ${maeK.toFixed(5)} = ${ratio.toExponential(3)} (need < 0.01)`},
      e_no_tuning_freeze_verified:{pass:true,detail:'frozen.json self-hash, parameter file, engine hash, build hashes and 1e-12 prediction recompute all verified before scoring'}
    };
    R.verdict=Object.values(R.criteria).every(c=>c.pass)?'PASS':'FAIL';
    const pc=R.path_clustered_bootstrap_vs_C0;
    R.path_robustness=(pc.all_sites.ci95[0]>0&&pc.targets_5_23.ci95[0]>0)?'POSITIVE':'INCONCLUSIVE_OR_NEGATIVE';
  }
  if(hasZ){
    const R=res.C3Z,pA=paired(rows,'C3Z',BOOT.resamples,BOOT.seed,'C3'),pT=paired(T,'C3Z',BOOT.resamples,BOOT.seed,'C3');
    R.status='DECLARED EXTENSION: C3 plus endpoint-local Z terms (params.cz.*.alpha). Not the primary comparator; the C3 verdict and criteria are unaffected by it.';
    R.vs_C3={paired_bootstrap_MAE_C3_minus_MAE_C3Z:{all_sites:pA,targets_5_23:pT},
      path_clustered_bootstrap:{all_sites:pathClustered(rows,'C3Z',BOOT.resamples,BOOT.seed,'C3'),targets_5_23:pathClustered(T,'C3Z',BOOT.resamples,BOOT.seed,'C3')},
      mae_change_C3Z_minus_C3:{all:mt.all.C3Z-mt.all.C3,targets_5_23:mt.targets_5_23.C3Z-mt.targets_5_23.C3,other_than_5_23:mt.other_than_5_23.C3Z-mt.other_than_5_23.C3}};
    R.target_site_mae=Object.fromEntries(TARGETS.map(p=>['site_'+p,{n_obs:mt['site_'+p].n,C0:mt['site_'+p].C0,C2:mt['site_'+p].C2,C3:mt['site_'+p].C3,C3Z:mt['site_'+p].C3Z}]));
  }
  return {groups:mt,sites,candidates:res};
}

function fmt(x,d=5){return Number.isFinite(x)?x.toFixed(d):String(x);}
export function renderText(rep){
  const L=[];
  if(rep.synthetic)L.push('*** SYNTHETIC DRY RUN: counts were fabricated from model predictions. Pipeline check only; not hardware evidence. ***');
  L.push(`frozen sha256 ${rep.frozen_sha256}  plan ${rep.plan_sha256}`);
  L.push(`circuits ${rep.circuits}, observables ${rep.observables}, counts sha256 ${rep.counts_sha256}`);
  const hasZ=rep.analysis.candidates.C3Z!==undefined,names=hasZ?NAMES:NAMES.filter(k=>k!=='C3Z');
  L.push('','MAE vs measured <Z> (group, n, '+names.join(', ')+')');
  for(const [g,m] of Object.entries(rep.analysis.groups))L.push(`  ${g.padEnd(24)} ${String(m.n).padStart(4)}  `+names.map(k=>fmt(m[k])).join('  '));
  L.push('','Per physical site (role, n, MAE C0, C2, C3'+(hasZ?', C3Z':'')+', change C2, change C3'+(hasZ?', change C3Z':'')+'; ! = flagged > +0.005)');
  const chg=(v,f)=>(v>=0?'+':'')+fmt(v)+(f?'!':' ');
  for(const s of rep.analysis.sites)L.push(`  q${String(s.physical).padStart(3)} ${s.role.padEnd(7)} n=${String(s.n_obs).padStart(3)}  ${fmt(s.mae.C0)} ${fmt(s.mae.C2)} ${fmt(s.mae.C3)}${hasZ?' '+fmt(s.mae.C3Z):''}  ${chg(s.change_C2_vs_C0,s.flag_C2)} ${chg(s.change_C3_vs_C0,s.flag_C3)}${hasZ?' '+chg(s.change_C3Z_vs_C0,s.flag_C3Z):''}`);
  for(const k of hasZ?['C3','C2','C3Z']:['C3','C2']){
    const c=rep.analysis.candidates[k],p=c.paired_bootstrap_vs_C0;
    const pc=c.path_clustered_bootstrap_vs_C0;
    L.push('',`=== ${k} ${k==='C3'?'(PRIMARY)':k==='C3Z'?'(DECLARED EXTENSION, not the primary comparator)':'(secondary)'}: relative MAE improvement over C0: pooled ${(100*c.relative_mae_improvement.pooled_all_sites).toFixed(2)}%, targets 5/23 ${(100*c.relative_mae_improvement.targets_5_23).toFixed(2)}% (5: ${(100*c.relative_mae_improvement.site_5).toFixed(2)}%, 23: ${(100*c.relative_mae_improvement.site_23).toFixed(2)}%), controls 3/21 ${(100*c.relative_mae_improvement.controls_3_21).toFixed(2)}%`);
    L.push(`    paired bootstrap MAE(C0)-MAE(${k}): all ${fmt(p.all_sites.mean_improvement)} CI95 [${fmt(p.all_sites.ci95[0])}, ${fmt(p.all_sites.ci95[1])}] wins ${p.all_sites.wins}/${p.all_sites.circuits}; targets ${fmt(p.targets_5_23.mean_improvement)} CI95 [${fmt(p.targets_5_23.ci95[0])}, ${fmt(p.targets_5_23.ci95[1])}] wins ${p.targets_5_23.wins}/${p.targets_5_23.circuits}`);
    const n=c.numerical_vs_physical;
    L.push(`    path-clustered observable-weighted CI95: all [${fmt(pc.all_sites.ci95[0])}, ${fmt(pc.all_sites.ci95[1])}] across ${pc.all_sites.clusters} paths; targets [${fmt(pc.targets_5_23.ci95[0])}, ${fmt(pc.targets_5_23.ci95[1])}] across ${pc.targets_5_23.clusters} paths; ${c.path_robustness}`);
    L.push(`    ledger: max l1 discard ${n.max_l1_discard.toExponential(3)}, old_bound ${n.max_old_bound.toExponential(3)}, min_bound ${n.max_min_bound.toExponential(3)}; physical MAE ${fmt(n.physical_prediction_error_mae)}; ratio min_bound/MAE ${n.ratio_candidate_min_bound_to_mae.toExponential(3)}, l1/MAE ${n.ratio_candidate_l1_to_mae.toExponential(3)}, old_bound/MAE ${n.ratio_candidate_old_bound_to_mae.toExponential(3)}`);
    if(c.relative_mae_improvement.pooled_all_sites<0)L.push(`    NEGATIVE RESULT: ${k} is WORSE than C0 pooled (${(100*c.relative_mae_improvement.pooled_all_sites).toFixed(2)}%).`);
    if(c.relative_mae_improvement.targets_5_23<0)L.push(`    NEGATIVE RESULT: ${k} is WORSE than C0 on targets 5/23 (${(100*c.relative_mae_improvement.targets_5_23).toFixed(2)}%).`);
    L.push('    criterion                                         result  detail');
    for(const [name,v] of Object.entries(c.criteria))L.push(`    ${name.padEnd(48)}  ${v.pass?'PASS':'FAIL'}   ${v.detail}`);
    if(k==='C3Z'){
      const v=c.vs_C3.paired_bootstrap_MAE_C3_minus_MAE_C3Z,t=c.target_site_mae;
      L.push(`    paired bootstrap MAE(C3)-MAE(C3Z): all ${fmt(v.all_sites.mean_improvement)} CI95 [${fmt(v.all_sites.ci95[0])}, ${fmt(v.all_sites.ci95[1])}] wins ${v.all_sites.wins}/${v.all_sites.circuits}; targets ${fmt(v.targets_5_23.mean_improvement)} CI95 [${fmt(v.targets_5_23.ci95[0])}, ${fmt(v.targets_5_23.ci95[1])}] wins ${v.targets_5_23.wins}/${v.targets_5_23.circuits}`);
      L.push('    target-site MAE (C0, C2, C3, C3Z): '+Object.entries(t).map(([s,x])=>`${s.replace('site_','q')} n=${x.n_obs} ${fmt(x.C0)} ${fmt(x.C2)} ${fmt(x.C3)} ${fmt(x.C3Z)}`).join('; '));
    }
    L.push(`    VERDICT ${k}${k==='C3Z'?' (declared extension)':''}: ${c.verdict}${rep.synthetic?' (SYNTHETIC DRY RUN, not evidence)':''}`);
  }
  return L.join('\n');
}

export function run({frozenFile,buildDir,paramsFile,countsFile}){
  const frozen=JSON.parse(readLF(frozenFile));
  const v=verifyFrozen(frozen,buildDir,paramsFile);
  if(!v.ok)return {refused:true,failures:v.failures,checks:v.checks};
  const {meas,synthetic}=loadCounts(countsFile,frozen);
  const rows=buildRows(frozen,meas),analysis=analyse(frozen,rows);
  return {refused:false,label:synthetic?'SYNTHETIC DRY RUN - NOT HARDWARE DATA':'PROSPECTIVE EVALUATION',synthetic,
    frozen_sha256:frozen.sha256,plan_sha256:frozen.build.plan_sha256,params_sha256:frozen.params_sha256,
    counts_sha256:sha256(readLF(countsFile)),circuits:Object.keys(frozen.predictions).length,observables:rows.length,
    freeze_check:{passed:true,checks:v.checks},thresholds:THRESH,bootstrap:BOOT,targets:TARGETS,controls:CONTROLS,analysis,rows};
}

function main(){
  const argv=process.argv.slice(2),opt={};
  for(let i=0;i<argv.length;i+=2){if(!argv[i].startsWith('--'))throw Error('bad arg '+argv[i]);opt[argv[i].slice(2)]=argv[i+1];}
  if(!opt.frozen||!opt.build||!opt.params||!opt.counts){console.error('usage: --frozen F --build DIR --params P --counts C [--out report.json]');process.exit(64);}
  const rep=run({frozenFile:opt.frozen,buildDir:opt.build,paramsFile:opt.params,countsFile:opt.counts});
  if(rep.refused){
    console.error('REFUSED: frozen predictions do not verify; evaluation not run. Criterion (e) FAILS.\n'+rep.failures.map(f=>'  - '+f).join('\n'));
    if(opt.out)fs.writeFileSync(opt.out,JSON.stringify(rep,null,1)+'\n');
    process.exit(2);
  }
  console.log(renderText(rep));
  if(opt.out)fs.writeFileSync(opt.out,JSON.stringify(rep)+'\n');
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main();
