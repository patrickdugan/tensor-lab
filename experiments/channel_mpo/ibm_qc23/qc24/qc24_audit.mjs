// QC-24 retrospective local coherent kick audit; no QPU/API calls.
import fs from 'node:fs';
function qc24(raw,published){
  // CPython Random integer seed initialization and 53-bit random().
  function pyRandom(seed){const mt=new Uint32Array(624);mt[0]=19650218;for(let i=1;i<624;i++)mt[i]=(Math.imul(1812433253,mt[i-1]^(mt[i-1]>>>30))+i)>>>0;let i=1;for(let k=624;k>0;k--){mt[i]=((mt[i]^Math.imul(mt[i-1]^(mt[i-1]>>>30),1664525))+seed)>>>0;if(++i>=624){mt[0]=mt[623];i=1;}}for(let k=623;k>0;k--){mt[i]=((mt[i]^Math.imul(mt[i-1]^(mt[i-1]>>>30),1566083941))-i)>>>0;if(++i>=624){mt[0]=mt[623];i=1;}}mt[0]=0x80000000;let idx=624;function uint(){if(idx>=624){for(let k=0;k<624;k++){let y=((mt[k]&0x80000000)|(mt[(k+1)%624]&0x7fffffff))>>>0;mt[k]=(mt[(k+397)%624]^(y>>>1)^((y&1)?0x9908b0df:0))>>>0;}idx=0;}let y=mt[idx++];y^=y>>>11;y^=(y<<7)&0x9d2c5680;y^=(y<<15)&0xefc60000;y^=y>>>18;return y>>>0;}return ()=>((uint()>>>5)*67108864+(uint()>>>6))/9007199254740992;}
  function sim(rec,offsets={}){let {n,depth:d,seed,physical_path:path}=rec,rng=pyRandom(seed),N=1<<n,R=new Float64Array(N),I=new Float64Array(N);R[0]=1;for(let step=0;step<d;step++){const rx=Array.from({length:n},()=>.55+.3*(rng()-.5)),rz=Array.from({length:n},()=>.15*(rng()-.5));const phase=(q,a,q2=-1)=>{let c=Math.cos(a/2),s=Math.sin(a/2);for(let j=0;j<N;j++){let sign=(((j>>q)&1) ^ (q2<0?0:((j>>q2)&1)))?1:-1;let v=R[j],w=I[j];R[j]=c*v-sign*s*w;I[j]=c*w+sign*s*v;}};for(let q=0;q<n;q++){let a=rx[q]+(offsets.global||0)+(offsets[path[q]]||0),c=Math.cos(a/2),s=Math.sin(a/2),b=1<<q;for(let j=0;j<N;j++)if(!(j&b)){let k=j|b,v=R[j],h=I[j],u=R[k],w=I[k];R[j]=c*v+s*w;I[j]=c*h-s*u;R[k]=c*u+s*h;I[k]=c*w-s*v;}phase(q,rz[q]);}for(let q=step%2;q<n-1;q+=2)phase(q,.7+.3*(rng()-.5),q+1);}let z=new Float64Array(n),norm=0;for(let j=0;j<N;j++){let v=R[j]**2+I[j]**2;norm+=v;for(let q=0;q<n;q++)z[q]+=((j>>q)&1)?-v:v;}if(Math.abs(norm-1)>1e-10)throw Error("state norm");return z;}
  const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
  const obs=Object.entries(raw).sort(([a],[b])=>a.localeCompare(b)).map(([id,r])=>{let z=sim(r),m=Array(r.n).fill(0),shots=0;for(const [bits,count] of Object.entries(r.counts)){if(bits.length!==r.n||!/^[01]+$/.test(bits))throw Error("bad bitstring");shots+=count;for(let q=0;q<r.n;q++)m[q]+=(bits[bits.length-1-q]==="1"?-1:1)*count;}if(shots!==r.shots)throw Error("shots mismatch");return {id,r,z,m:m.map(x=>x/shots)};});
  let maxIdeal=0,maxCount=0;
  for(const p of published.rows.filter(x=>x.family==="kicked_ising")){const o=obs.find(x=>x.id===p.id);maxIdeal=Math.max(maxIdeal,Math.abs(o.z[p.bit]-p.ideal));maxCount=Math.max(maxCount,Math.abs(o.m[p.bit]-p.measured));}
  if(maxIdeal>1e-10||maxCount>1e-12||obs.length!==36)throw Error("QC23 replay mismatch");
  const train=obs.filter(o=>o.r.n===8);
  const isTarget=q=>q===5||q===23;
  // Fit only width 8, all qubits, bounded effective logical kick errors.
  const loss=(par,global)=>mean(train.flatMap(o=>{let z=sim(o.r,{"5":par[0],"23":par[1],global:global?par[2]:0});return o.m.map((m,q)=>(z[q]-m)**2);}));
  function fit(global){let x=global?[0,0,0]:[0,0],val=loss(x,global),evals=1;for(const step of [.04,.02,.01,.005,.0025,.00125,.000625,.0003125]){for(let pass=0;pass<40;pass++){let improved=false;for(let k=0;k<x.length;k++){let best=x,bv=val;for(const sign of [-1,1]){let y=x.slice();y[k]+=sign*step;if(Math.abs(y[k])>.3)continue;let v=loss(y,global);evals++;if(v<bv-1e-14){bv=v;best=y;}}if(bv<val-1e-14){x=best;val=bv;improved=true;}}if(!improved)break;}}return {offsets:{5:x[0],23:x[1],global:global?x[2]:0},train_mse:val,evaluations:evals};}
  const local=fit(false), global=fit(true);
  // Readout-only CPTP baseline: independent 0->1 and 1->0 flips, each in [0,.3].
  const ro={};
  for(const p of [5,23]){let rows=train.map(o=>{let q=o.r.physical_path.indexOf(p);return [o.z[q],o.m[q]]});let best=Infinity;for(let i=0;i<=300;i++)for(let j=0;j<=300;j++){let e0=i/1000,e1=j/1000,v=mean(rows.map(([z,m])=>((1-e0-e1)*z+e1-e0-m)**2));if(v<best){best=v;ro[p]={e0,e1};}}}
  // Nonphysical empirical depth-offset control; deliberately simple.
  const depthSlope={};for(const p of [5,23]){const r=train.map(o=>{let q=o.r.physical_path.indexOf(p);return [o.r.depth,o.m[q]-o.z[q]]});depthSlope[p]=r.reduce((s,[d,v])=>s+d*v,0)/r.reduce((s,[d])=>s+d*d,0);}
  const rows=[];for(const o of obs){let l=sim(o.r,local.offsets),g=sim(o.r,global.offsets);for(let q=0;q<o.r.n;q++){let p=o.r.physical_path[q],z=o.z[q],m=o.m[q],r=ro[p];rows.push({id:o.id,n:o.r.n,depth:o.r.depth,seed:o.r.seed,logical:q,physical:p,measured:m,ideal:z,local_rx:l[q],global_local_rx:g[q],readout:r?(1-r.e0-r.e1)*z+r.e1-r.e0:z,depth_offset:Math.max(-1,Math.min(1,z+(depthSlope[p]||0)*o.r.depth))});}}
  const keys=["ideal","local_rx","global_local_rx","readout","depth_offset"];
  const metrics={};for(const [split,sel] of Object.entries({fit_n8:r=>r.n===8,transfer_n12:r=>r.n===12,transfer_n16:r=>r.n===16,transfer_all:r=>r.n!==8})){const group=rows.filter(sel);metrics[split]={};for(const [subset,filter] of Object.entries({all:r=>true,target:r=>isTarget(r.physical),other:r=>!isTarget(r.physical)})){let data=group.filter(filter);metrics[split][subset]={rows:data.length,circuits:new Set(data.map(r=>r.id)).size,...Object.fromEntries(keys.map(k=>[k,{mae:mean(data.map(r=>Math.abs(r[k]-r.measured))),rmse:Math.sqrt(mean(data.map(r=>(r[k]-r.measured)**2))),within_005:mean(data.map(r=>+(Math.abs(r[k]-r.measured)<=.05)))}]))};}}
  const cells={};for(const r of rows){const k=[r.n,r.depth,r.physical].join("/");(cells[k]??=[]).push(r);}
  const cellResiduals=Object.entries(cells).map(([k,v])=>({cell:k,seeds:v.length,...Object.fromEntries(keys.map(key=>[key,mean(v.map(r=>r.measured-r[key]))]))}));
  return {status:"retrospective width transfer; qubits selected after inspecting all widths; no prospective claim",fit:{local,global,readout:ro,depthSlope},validation:{circuits:obs.length,rows:rows.length,prior_ideal_max_gap:maxIdeal,prior_count_max_gap:maxCount,seed:41,train_width:8,transfer_widths:[12,16]},metrics,cellResiduals,rows};
}
const root=new URL('../',import.meta.url);
const raw=JSON.parse(fs.readFileSync(new URL('results/ibm_run_nc/ibm/nc_eval_counts.json',root),'utf8'));
const prior=JSON.parse(fs.readFileSync(new URL('results/hw_root/analysis_ibm.json',root),'utf8'));
const result=qc24(raw,prior);
const output=process.argv[2]||new URL('qc24_results.json',import.meta.url);
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({fit:result.fit,validation:result.validation,metrics:result.metrics},null,2));
