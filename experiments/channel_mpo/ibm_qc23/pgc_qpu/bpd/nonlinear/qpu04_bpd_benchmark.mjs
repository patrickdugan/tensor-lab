// Fixed synthetic BPD benchmark: full setup, learning/audit, scoring and updates.
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import os from 'node:os';
const N=1024,M=8,K=11,S=10,TRIALS=24,STEPS=8,TAU=2*Math.PI;
const rows=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,-1,0],[0,1,-1],[1,0,-1],[1,1,1],[2,1,0],[0,2,1]];
const extra=[-1,-1,-1,2,0,1,2,0,1,0,2,0];
const vec=[];for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++)if(a||b||c)vec.push([a,b,c]);
const dot=(a,x)=>a.reduce((s,v,i)=>s+v*x[i],0), mod=x=>((x%M)+M)%M;
const phase=(a,j,x)=>TAU*dot(a,x)/M+(j<0?0:.35*Math.sin(TAU*x[j]/M));
const H=p=>p.reduce((s,v)=>v>0?s-v*Math.log(v):s,0);
const rng=seed=>{let x=seed>>>0||1;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296}};
const cpu=t=>{const a=process.cpuUsage(t);return (a.user+a.system)/1000};
function binom(p){const a=[(1-p)**S];for(let j=1;j<K;j++)a.push(a[j-1]*(S-j+1)/j*p/(1-p));return a;}
function probability(p,v,ax){return Math.max(1e-10,Math.min(1-1e-10,(1+.976*[.97,.84][v]*(ax?Math.sin(p):Math.cos(p)))/2));}
function run(mode){
 const start=process.cpuUsage(),states=[];
 for(let a=0;a<M;a++)for(let b=0;b<M;b++)for(let c=0;c<M;c++)for(let v=0;v<2;v++)states.push({x:[a,b,c],v});
 if(states.length!==N)throw Error('shape');
 let fit=rows.map((row,i)=>({row,j:extra[i]}));
 if(mode==='learned'){
  const r=rng(261127045),ids=new Set();while(ids.size<40)ids.add(Math.floor(r()*N));const train=[...ids].map(i=>states[i].x);
  fit=rows.map((row,i)=>{
   const truth=train.map(x=>phase(row,extra[i],x));let best=vec[0],cost=Infinity;
   for(const v of vec){let z=0;for(let k=0;k<train.length;k++)z+=2-2*Math.cos(truth[k]-TAU*dot(v,train[k])/M);if(z<cost-1e-12){cost=z;best=v;}}
   const residual=train.map((x,k)=>{const p=truth[k]-TAU*dot(best,x)/M;return Math.atan2(Math.sin(p),Math.cos(p))});
   let bestJ=-1,bestErr=residual.reduce((s,x)=>s+x*x,0);
   for(let j=0;j<3;j++){let xx=0,xy=0;for(let k=0;k<train.length;k++){const u=Math.sin(TAU*train[k][j]/M);xx+=u*u;xy+=u*residual[k];}const a=xy/xx;let err=0;for(let k=0;k<train.length;k++)err+=(residual[k]-a*Math.sin(TAU*train[k][j]/M))**2;if(err<bestErr-1e-12){bestErr=err;bestJ=j;}}
   return {row:best,j:bestJ};
  });
 }
 let fallbacks=0;
 const actions=[];
 for(let i=0;i<rows.length;i++)for(let ax=0;ax<2;ax++){
  const probs=states.map(s=>probability(phase(rows[i],extra[i],s.x),s.v,ax));
  const truth=probs.map(binom),entropy=truth.map(H);let ids=null,groups=null;
  if(mode!=='dense'){
   const g=new Map();ids=new Int32Array(N);let bad=false;
   for(let j=0;j<N;j++){const s=states[j],f=fit[i],key=[mod(dot(f.row,s.x)),s.v,f.j<0?-1:s.x[f.j]].join(':');if(!g.has(key))g.set(key,{id:g.size,t:truth[j],h:entropy[j]});const a=g.get(key);ids[j]=a.id;for(let k=0;k<K;k++)if(Math.abs(a.t[k]-truth[j][k])>1e-12)bad=true;}
   if(bad){fallbacks++;ids=null;groups=null;}else groups=[...g.values()];
  }
  actions.push({i,ax,probs,truth,entropy,ids,groups});
 }
 const prep=cpu(start);let digest=2166136261,posteriorChecksum=0;
 for(let t=0;t<TRIALS;t++){
  const r=rng(12345+t),truthIndex=Math.floor(r()*N);let w=new Float64Array(N).fill(1/N),used=new Set();
  for(let step=0;step<STEPS;step++){
   let winner=-1,best=-Infinity;const cache=new Map();
   for(let a=0;a<actions.length;a++)if(!used.has(a)){
    const x=actions[a],mix=new Float64Array(K);let e=0;
    if(x.groups){let mass=cache.get(x.i);if(!mass){mass=new Float64Array(x.groups.length);for(let j=0;j<N;j++)mass[x.ids[j]]+=w[j];cache.set(x.i,mass);}for(let j=0;j<x.groups.length;j++){const g=x.groups[j],p=mass[j];e+=p*g.h;for(let k=0;k<K;k++)mix[k]+=p*g.t[k];}}
    else for(let j=0;j<N;j++){const p=w[j];e+=p*x.entropy[j];for(let k=0;k<K;k++)mix[k]+=p*x.truth[j][k];}
    const score=H(mix)-e;if(score>best+1e-10){best=score;winner=a;}
   }
   const a=actions[winner];used.add(winner);digest=Math.imul(digest^winner,16777619)>>>0;
   let count=0;for(let i=0;i<S;i++)count+=Number(r()<a.probs[truthIndex]);let mass=0;for(let j=0;j<N;j++){w[j]*=a.truth[j][count];mass+=w[j];}for(let j=0;j<N;j++)w[j]/=mass;
  }
  posteriorChecksum+=w[truthIndex];
 }
 return {mode,cpu_ms:cpu(start),setup_cpu_ms:prep,max_rss_kib:process.resourceUsage().maxRSS,digest,posteriorChecksum,fallbacks,states:N,trials:TRIALS,steps:STEPS};
}
if(process.argv[2]==='--worker'){console.log(JSON.stringify(run(process.argv[3])));}
else{
 const runs=[];for(let rep=0;rep<5;rep++)for(const mode of ['dense','oracle','learned']){
  const r=spawnSync(process.execPath,[fileURLToPath(import.meta.url),'--worker',mode],{encoding:'utf8'});if(r.status!==0)throw Error(r.stderr);runs.push({rep,...JSON.parse(r.stdout)});
 }
 const same=runs.every(r=>r.digest===runs[0].digest&&Math.abs(r.posteriorChecksum-runs[0].posteriorChecksum)<1e-12);if(!same)throw Error('policy mismatch');
 const median=a=>a.sort((a,b)=>a-b)[Math.floor(a.length/2)];const summary={};for(const mode of ['dense','oracle','learned']){const a=runs.filter(r=>r.mode===mode);summary[mode]={cpu_ms_median:median(a.map(r=>r.cpu_ms)),setup_cpu_ms_median:median(a.map(r=>r.setup_cpu_ms)),max_rss_kib_median:median(a.map(r=>r.max_rss_kib)),fallbacks:a[0].fallbacks};}
 console.log(JSON.stringify({label:'SYNTHETIC fixed workload, actual per-process CPU/RSS; not QPU time',node:process.version,cpu:os.cpus()[0]?.model,platform:process.platform,matched_paths:same,summary,runs},null,2));
}
