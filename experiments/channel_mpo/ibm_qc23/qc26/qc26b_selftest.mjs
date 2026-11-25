// QC-26B: generate random heterogeneous circuits and sparse-Pauli results for the dense Python cross-check.
import fs from 'node:fs';
import {evaluate} from './qc26b_engine.mjs';
let seed=2611251;const rnd=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
const ri=m=>Math.floor(rnd()*m);
const cases=[];
for(let c=0;c<48;c++){
  const n=3+ri(3),seq=[],sx={},cz={};
  for(let q=0;q<n;q++)sx[q]=(rnd()-.5)*.2;
  for(let a=0;a<n-1;a++)cz[`${a}_${a+1}`]={alpha:{[a]:(rnd()-.5)*.1,[a+1]:(rnd()-.5)*.1},delta:(rnd()-.5)*.2};
  for(let s=0;s<36;s++){
    const t=ri(8),q=ri(n);
    if(t===0)seq.push({name:"rz",q:[q],p:[q],angle:(rnd()-.5)*6.2});
    else if(t===1)seq.push({name:"sx",q:[q],p:[q]});
    else if(t===2)seq.push({name:"x",q:[q],p:[q]});
    else if(t===3||t===4){const a=ri(n-1);const rev=rnd()<.5;const qs=rev?[a+1,a]:[a,a+1];seq.push({name:"cz",q:qs,p:qs});seq.push({name:"czerr",q:qs,p:qs});}
    else if(t===5){let pxy,pz;do{pxy=rnd()*.03;pz=rnd()*.03;}while((1-2*(pxy+pz))**2>1-4*pxy);seq.push({name:"relax",q:[q],p:[q],e:[1,1-2*(pxy+pz),1-2*(pxy+pz),1-4*pxy]});}
    else if(t===6){const dp=rnd()*.03;seq.push({name:"diag",q:[q],p:[q],e:[1,1-4*dp/3,1-4*dp/3,1-4*dp/3]});}
    else {const a=ri(n-1);const dp=rnd()*.03;seq.push({name:"diag2",q:[a,a+1],p:[a,a+1],e:1-16*dp/15});}
  }
  const ro=rnd()*.03,res=[];
  for(const nonunital of [false,true]){
    const cfg={nonunital,sx,cz};
    res.push({nonunital,z:Array.from({length:n},(_,q)=>evaluate(n,seq,q,ro,cfg).z),
              z_no_coherent:Array.from({length:n},(_,q)=>evaluate(n,seq,q,ro,{nonunital}).z)});
  }
  cases.push({n,seq,sx,cz,ro,res});
}
fs.writeFileSync(process.argv[2]||'qc26b_selftest_cases.json',JSON.stringify({seed:2611251,cases}));
console.log('cases',cases.length);
