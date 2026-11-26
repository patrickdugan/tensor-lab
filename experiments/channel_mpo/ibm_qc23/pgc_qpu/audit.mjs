// PGC-QPU01: offline native channel identifiability audit, no IBM access.
import fs from 'node:fs';
import crypto from 'node:crypto';
import * as engine from '../qc26/qc26b_engine.mjs';
const root=new URL('../',import.meta.url);
const rd=p=>fs.readFileSync(new URL(p,root),'utf8');
const j=p=>JSON.parse(rd(p));
function spectrum(J){
  const n=J[0].length,A=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>J.reduce((s,r)=>s+r[i]*r[j],0)));
  const norms=A.map((r,i)=>Math.sqrt(r[i]));
  const V=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
  for(let iter=0;iter<100*n*n;iter++){
    let p=0,q=1,best=0;
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(A[i][j])>best){best=Math.abs(A[i][j]);p=i;q=j}
    if(best<1e-14*Math.max(1,...A.map((r,i)=>Math.abs(r[i]))))break;
    const a=A[p][p],b=A[q][q],c=A[p][q],t=.5*Math.atan2(2*c,b-a),co=Math.cos(t),si=Math.sin(t);
    for(let k=0;k<n;k++)if(k!==p&&k!==q){const x=A[k][p],y=A[k][q];A[k][p]=A[p][k]=co*x-si*y;A[k][q]=A[q][k]=si*x+co*y}
    A[p][p]=co*co*a-2*co*si*c+si*si*b; A[q][q]=si*si*a+2*co*si*c+co*co*b;A[p][q]=A[q][p]=0;
    for(let k=0;k<n;k++){const x=V[k][p],y=V[k][q];V[k][p]=co*x-si*y;V[k][q]=si*x+co*y}
  }
  const order=Array.from({length:n},(_,i)=>i).sort((i,j)=>A[j][j]-A[i][i]);
  const sv=order.map(i=>Math.sqrt(Math.max(0,A[i][i])));
  const v=V.map(row=>row[order[n-1]]);
  return {rows:J.length,columns:n,singular_values:sv,numerical_rank_relative_1e6:sv.filter(x=>x>sv[0]*1e-6).length,condition_number:sv[n-1]>sv[0]*1e-6?sv[0]/sv[n-1]:null,column_norms:norms,weakest_right_vector:v};
}
function analyze(engine,entries,snap,rec){
  const edgeKeys=Object.keys(rec.cz_phases).sort(),sites=Object.keys(rec.sx_offsets_rad).sort((a,b)=>+a-+b);
  const pars=[...sites.map(site=>({kind:"sx",site,label:"SX_"+site,value:rec.sx_offsets_rad[site]})),...edgeKeys.map(edge=>({kind:"delta",edge,label:"ZZ_"+edge,value:rec.cz_phases[edge].delta})),...edgeKeys.flatMap(edge=>edge.split("_").map(site=>({kind:"alpha",edge,site,label:"Z_"+edge+"_"+site,value:0})))];
  const base=pars.map(x=>x.value);
  function config(x){const cfg={nonunital:true,sx:{},cz:{}};for(let i=0;i<pars.length;i++){const p=pars[i],v=x[i];if(p.kind==="sx")cfg.sx[p.site]=v;else{const e=cfg.cz[p.edge]??(cfg.cz[p.edge]={delta:0,alpha:{}});if(p.kind==="delta")e.delta=v;else e.alpha[p.site]=v}}return cfg}
  let maxReplay=0,maxCounts=0,maxDerivativeGap=0,maxLedger=0;
  const rows=[];
  for(const e of entries){
    const {r,qasm,counts}=e,seq=engine.schedule(r,engine.parse(r,qasm),snap);
    if(JSON.stringify(counts.physical_path)!==JSON.stringify(r.physical_path))throw Error("path mismatch");
    const s=Object.values(counts.counts).reduce((a,b)=>a+b,0);if(s!==r.shots)throw Error("shots mismatch");
    for(let q=0;q<r.n;q++){
      const old=rec.rows.find(x=>x.id===r.id&&x.bit===q&&x.set==="nc_eval");
      if(!old)throw Error("missing receipt");
      const measured=Object.entries(counts.counts).reduce((a,[bits,c])=>a+(bits[r.n-1-q]==="1"?-c:c),0)/s;
      maxCounts=Math.max(maxCounts,Math.abs(measured-old.measured));
      const predict=x=>{let z=engine.evaluate(r.n,seq,q,snap.qubits[r.physical_path[q]].readout_error||0,config(x));if(!Number.isFinite(z.z))throw Error("nonfinite");maxLedger=Math.max(maxLedger,z.discard);return z.z};
      const z=predict(base);maxReplay=Math.max(maxReplay,Math.abs(z-old.C3));
      const grad=[];
      for(let i=0;i<pars.length;i++){
        const diff=h=>{const a=base.slice(),b=base.slice();a[i]+=h;b[i]-=h;return (predict(a)-predict(b))/(2*h)};
        const d1=diff(0.0002),d2=diff(0.0001);maxDerivativeGap=Math.max(maxDerivativeGap,Math.abs(d1-d2));grad.push(d2);
      }
      rows.push({id:r.id,bit:q,physical:r.physical_path[q],depth:r.depth,measured,prediction:z,jacobian:grad});
    }
  }
  if(maxReplay>1e-9||maxCounts>1e-12||maxDerivativeGap>1e-5)throw Error("validation gate");
  const targets=rows.filter(x=>[5,23].includes(x.physical));
  const summarize=rs=>({restricted:spectrum(rs.map(x=>x.jacobian.slice(0,6))),expanded:spectrum(rs.map(x=>x.jacobian))});
  return {scope:"RETROSPECTIVE local derivative audit of the eight actual width-8 depth-1/2 fit circuits; no new fit or hardware",parameters:pars,steps_rad:[0.0002,0.0001],checks:{circuits:entries.length,observations:rows.length,max_replay_gap:maxReplay,max_measured_gap:maxCounts,max_step_halving_derivative_gap:maxDerivativeGap,max_truncation_l1:maxLedger},all_sites:summarize(rows),target_sites:summarize(targets),rows};
}
function makeSupplement(plan,rec){
  const edges=Object.entries(rec.cz_phases).sort((a,b)=>Math.abs(b[1].delta)-Math.abs(a[1].delta)).slice(0,2).map(([k])=>k);
  const additions=[],all=[];
  for(const key of edges){
    const forward=plan.edges.find(e=>e.slice().sort((a,b)=>a-b).join("_")===key);
    if(!forward)throw Error("missing edge "+key);
    const existing=plan.rows.filter(r=>r.family==="cz"&&r.path.join(",")===forward.join(","));
    if(existing.length!==12)throw Error("unexpected existing plan");
    for(const e of existing)all.push({...e,edge:key,orientation:"forward",source:"QC26A"});
    for(const [orientation,path,ms] of [["forward",forward,[1]],["reverse",forward.slice().reverse(),[0,1,2]]]){
      for(const state of [0,1])for(const m of ms)for(const axis of ["X","Y"]){
        const row={id:`pgc_${path.join("_")}_${state}_${m}_${axis}`,family:"cz",path,state,m,axis,shots:plan.shots,edge:key,orientation,source:"PGC"};
        additions.push(row);all.push(row);
      }
    }
  }
  const phase=(r,x)=>r.m*(x[r.orientation==="forward"?1:0]+(r.state===0?1:-1)*x[2]+r.state*Math.PI);
  const value=(r,x)=>r.axis==="X"?Math.cos(phase(r,x)):Math.sin(phase(r,x));
  const base=[.071,-.043,.11],aliases={control_shift:[.071+.37,-.043,.11],even_lattice:[.071,-.043+Math.PI/2,.11+Math.PI/2]};
  const metrics={};
  for(const edge of edges){
    const rs=all.filter(r=>r.edge===edge),old=rs.filter(r=>r.source==="QC26A");
    const jac=rows=>spectrum(rows.map(r=>{let phi=phase(r,base),v=(r.axis==="X"?-Math.sin(phi):Math.cos(phi))*r.m;return [r.orientation==="reverse"?v:0,r.orientation==="forward"?v:0,(r.state===0?1:-1)*v]}));
    const gap=(rows,x)=>Math.max(...rows.map(r=>Math.abs(value(r,x)-value(r,base))));
    metrics[edge]={original:jac(old),augmented:jac(rs),original_alias_gaps:Object.fromEntries(Object.entries(aliases).map(([k,x])=>[k,gap(old,x)])),augmented_alias_gaps:Object.fromEntries(Object.entries(aliases).map(([k,x])=>[k,gap(rs,x)]))};
    if(metrics[edge].original.numerical_rank_relative_1e6!==2||metrics[edge].augmented.numerical_rank_relative_1e6!==3||Object.values(metrics[edge].original_alias_gaps).some(x=>x>1e-12)||Object.values(metrics[edge].augmented_alias_gaps).some(x=>x<.1))throw Error("design invariant");
  }
  return {scope:"IDEAL DIAGONAL ERROR MODEL; logical proposal only, no IBM compilation/authorization; compare to same-session fresh QC26A baselines",backend:plan.backend,parameter_order:["alpha_control","alpha_target","delta"],reference_parameters_rad:base,additions,metrics,estimate:{shots_per_circuit:plan.shots,additional_circuits:additions.length,total_additional_shots:additions.length*plan.shots,rough_seconds:additions.length*(plan.shots*(0.00025+0.00001)+0.1),note:"same simplified timing allowances as QC26A, excludes actual backend-dependent cost; compile and obtain owner authorization before submission"}};
}
const manifest=j('results/hw_root/compiled/compiled_nc_eval.json');
const snapshot=j('results/hw_root/compiled/calibration_snapshot_nc_eval.json');
const replay=j('qc26/receipts/qc26b_replay_retrospective.json');
const counts=j('results/hw_root/ibm/nc_eval_counts.json');
const plan=j('qc26/qc26a_plan.json');
const entries=manifest.circuits.filter(r=>r.n===8&&r.depth<=2).map(r=>{const qasm=rd('results/ibm_run_nc/circuits/'+r.id+'.qasm').replace(/\r\n/g,'\n');if(crypto.createHash('sha256').update(qasm).digest('hex')!==r.qasm3_sha256)throw Error('QASM hash mismatch '+r.id);return {r,qasm,counts:counts[r.id]}});
const result=analyze(engine,entries,snapshot,replay),supplement=makeSupplement(plan,replay);
const out={result,supplement};
if(process.argv.includes('--check')){const expected=JSON.parse(fs.readFileSync(new URL('receipt.json',import.meta.url),'utf8'));function compare(a,b,p=''){if(typeof a==='number'){if(!Number.isFinite(b)||Math.abs(a-b)>1e-7*Math.max(1,Math.abs(a)))throw Error('receipt numeric mismatch '+p+' '+a+' '+b)}else if(a&&typeof a==='object'){if(!b||Object.keys(a).join()!==Object.keys(b).join())throw Error('receipt shape '+p);for(const k of Object.keys(a))compare(a[k],b[k],p+'/'+k)}else if(a!==b)throw Error('receipt mismatch '+p)}compare(out,expected);console.log('PASS: native QASM hashes, source counts, replay, step-halving, local spectra, analytic Ramsey aliases');}else console.log(JSON.stringify(out,null,2));
