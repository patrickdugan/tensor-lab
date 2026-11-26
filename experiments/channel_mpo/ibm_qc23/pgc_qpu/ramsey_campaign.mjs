// Independent four-edge phase pilot design and synthetic checks; no QPU calls.
import fs from 'node:fs';
function campaign(){
  const groups=[{id:"q23",pairs:[[16,23],[22,23]]},{id:"q5",pairs:[[4,5],[6,5]]}];
  const shots=384,cap=15,reps=[0,1,2,8],estimate={shot_repetition_delay_s:.00025,per_shot_circuit_allowance_s:.00001,per_circuit_overhead_allowance_s:.1};
  const rand=(seed)=>()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
  const key=path=>path.slice().sort((a,b)=>a-b).join("_");
  const blocks=groups.map((g,i)=>{
    const sites=[...new Set(g.pairs.flat())].sort((a,b)=>a-b),rows=[];
    for(const state of [0,1])rows.push({id:`pgc_${g.id}_spam_${state}`,family:"spam",path:sites,state,shots});
    for(const forward of g.pairs)for(const orientation of ["forward","reverse"]){
      const path=orientation==="forward"?forward:forward.slice().reverse();
      for(const state of [0,1])for(const m of reps)for(const axis of ["X","Y"])rows.push({id:`pgc_${g.id}_${path.join("_")}_${state}_${m}_${axis}`,family:"cz",edge:key(path),orientation,path,state,m,axis,shots});
    }
    const seed=26112601+i,rng=rand(seed);
    for(let j=rows.length-1;j>0;j--){let k=Math.floor(rng()*(j+1));[rows[j],rows[k]]=[rows[k],rows[j]]}
    if(rows.length!==66||new Set(rows.map(r=>r.id)).size!==66)throw Error("manifest");
    return {id:g.id,backend:"ibm_kingston",pairs:g.pairs,sites,shots,rows,ordering_seed:seed,cap_seconds:cap,estimate:{...estimate,nominal_seconds:rows.length*(shots*(estimate.shot_repetition_delay_s+estimate.per_shot_circuit_allowance_s)+estimate.per_circuit_overhead_allowance_s)}};
  });
  const truth={"16_23":{local:{16:.031,23:-.044},delta:-.10},"22_23":{local:{22:-.012,23:.035},delta:.027},"4_5":{local:{4:.026,5:-.021},delta:-.032},"5_6":{local:{5:.048,6:-.018},delta:.10}};
  const theta=(row,x)=>row.m*(x.local[row.path[1]]+(row.state===0?1:-1)*x.delta+row.state*Math.PI);
  function dense(row,x){
    // Independent Schrödinger-picture diagonal-gate oracle; basis index bit 0=control, bit 1=probe.
    const a=Array.from({length:4},(_,i)=>[(i&1)===row.state?1/Math.sqrt(2):0,0]);
    for(let k=0;k<row.m;k++)for(let i=0;i<4;i++){
      const zc=1-2*(i&1),zt=1-2*((i>>1)&1),ph=-(x.local[row.path[0]]*zc+x.local[row.path[1]]*zt+x.delta*zc*zt)/2;
      const s=i===3?-1:1,co=s*Math.cos(ph),si=s*Math.sin(ph),[r,im]=a[i];
      a[i]=[r*co-im*si,r*si+im*co];
    }
    let X=0,Y=0,norm=0;
    for(let i=0;i<4;i++)norm+=a[i][0]**2+a[i][1]**2;
    for(let i=0;i<2;i++){let [ar,ai]=a[i],[br,bi]=a[i+2];X+=2*(ar*br+ai*bi);Y+=2*(ar*bi-ai*br)}
    return {expectation:row.axis==="X"?X:Y,norm};
  }
  let denseCases=0,denseGap=0,normGap=0;
  for(const block of blocks)for(const row of block.rows)if(row.family==="cz")for(const mode of ["injected","null","alias"]){
    let x=truth[row.edge];if(mode==="null")x={local:Object.fromEntries(row.path.map(q=>[q,0])),delta:0};if(mode==="alias")x={local:Object.fromEntries(Object.entries(x.local).map(([q,v])=>[q,v+Math.PI/2])),delta:x.delta+Math.PI/2};
    const actual=dense(row,x),phase=theta(row,x),pred=row.axis==="X"?Math.cos(phase):Math.sin(phase);
    denseCases++;denseGap=Math.max(denseGap,Math.abs(pred-actual.expectation));normGap=Math.max(normGap,Math.abs(actual.norm-1));
  }
  if(denseGap>1e-12||normGap>1e-12)throw Error("oracle disagreement");
  // Finite-shot synthetic pilot with effective asymmetric SPAM and per-state intercept.
  // Each probe marginal sampled independently; this is not a correlated-readout model.
  const wrap=(a,ref)=>a+2*Math.PI*Math.round((ref-a)/(2*Math.PI));
  const e0=p=>.005+.001*(p%4),e1=p=>.009+.001*(p%3);
  function synth(block,model,rng){
    const bin=(v,p)=>{const obs=e1(p)-e0(p)+(1-e0(p)-e1(p))*v,prob=(1-obs)/2;if(prob<0||prob>1)throw Error("bad probability");let one=0;for(let i=0;i<shots;i++)one+=+(rng()<prob);return 1-2*one/shots};
    const spam={};
    for(const p of block.sites){const a=bin(1,p),b=bin(-1,p);spam[p]={e0:(1-a)/2,e1:(1+b)/2}}
    const data={};
    for(const r of block.rows)if(r.family==="cz"){
      const x=model[r.edge],p=r.path[1],ph=theta(r,x)+(r.state===0?.035:-.025)+(r.orientation==="reverse"?.014:0),vis=.97*Math.exp(-.004*r.m),v=vis*(r.axis==="X"?Math.cos(ph):Math.sin(ph));
      const z=bin(v,p),s=spam[p],contrast=1-s.e0-s.e1;if(contrast<.5)throw Error("synthetic SPAM");
      data[r.id]=(z-(s.e1-s.e0))/contrast;
    }
    const estimates={};
    for(const forward of block.pairs){
      let local={},deltas=[],minVisibility=Infinity;
      for(const orientation of ["forward","reverse"]){
        const path=orientation==="forward"?forward:forward.slice().reverse(),slopes=[];
        for(const state of [0,1]){
          const phases=[];
          for(const m of reps){
            const id=`pgc_${block.id}_${path.join("_")}_${state}_${m}_`,X=data[id+"X"],Y=data[id+"Y"];
            minVisibility=Math.min(minVisibility,Math.hypot(X,Y));
            let ph=Math.atan2(Y,X)-m*state*Math.PI;
            const prediction=m===0?0:m===1?phases[0]:m===2?phases[0]+2*(phases[1]-phases[0]):phases[0]+m*(phases[2]-phases[0])/2;
            phases.push(wrap(ph,prediction));
          }
          const xm=reps.reduce((a,b)=>a+b,0)/reps.length,ym=phases.reduce((a,b)=>a+b,0)/phases.length;
          slopes.push(reps.reduce((s,m,i)=>s+(m-xm)*(phases[i]-ym),0)/reps.reduce((s,m)=>s+(m-xm)**2,0));
        }
        local[path[1]]=(slopes[0]+slopes[1])/2;deltas.push((slopes[0]-slopes[1])/2);
      }
      estimates[key(forward)]={local,delta:(deltas[0]+deltas[1])/2,orientation_delta_gap:Math.abs(deltas[0]-deltas[1]),min_visibility:minVisibility};
    }
    return estimates;
  }
  const trials=256,stats={};
  for(const mode of ["injected","null"]){
    const rng=rand(mode==="injected"?26112655:26112656),model=JSON.parse(JSON.stringify(truth));
    if(mode==="null")for(const x of Object.values(model)){x.delta=0;for(const k of Object.keys(x.local))x.local[k]=0}
    const samples=Object.fromEntries(Object.keys(model).map(k=>[k,[]]));
    for(let i=0;i<trials;i++)for(const b of blocks)for(const [k,v] of Object.entries(synth(b,model,rng)))samples[k].push(v);
    stats[mode]={};
    for(const [k,x] of Object.entries(model)){
      const rs=samples[k],fields=[["delta",x.delta,r=>r.delta],...Object.entries(x.local).map(([p,v])=>["local_"+p,v,r=>r.local[p]])];
      stats[mode][k]={parameters:Object.fromEntries(fields.map(([label,t,f])=>{const vals=rs.map(f);return [label,{truth:t,mean:vals.reduce((a,b)=>a+b,0)/trials,rmse:Math.sqrt(vals.reduce((a,b)=>a+(b-t)**2,0)/trials),correct_sign:t===0?null:vals.filter(v=>Math.sign(v)===Math.sign(t)).length,abs_over_0p02:vals.filter(v=>Math.abs(v)>.02).length}]})),orientation_gap_over_0p04:rs.filter(r=>r.orientation_delta_gap>.04).length,min_visibility:Math.min(...rs.map(r=>r.min_visibility))};
    }
  }
  return {version:"PGC-QPU01-RAMSEY-v1",status:"offline plan; uncompiled and unauthorized",blocks,truth,oracle:{cases:denseCases,max_expectation_gap:denseGap,max_norm_gap:normGap},synthetic:{trials_per_arm:trials,stats,scope:"synthetic probe marginals, asymmetric effective SPAM, phase offsets and visibility decay; no crosstalk, leakage, hardware drift, joint uncertainty, or hardware significance claim"},total_estimated_seconds:blocks.reduce((a,b)=>a+b.estimate.nominal_seconds,0)};
}
const out=campaign();
if(process.argv.includes('--check')){const expected=JSON.parse(fs.readFileSync(new URL('ramsey_campaign.json',import.meta.url),'utf8'));function compare(a,b,p=''){if(typeof a==='number'){if(!Number.isFinite(b)||Math.abs(a-b)>1e-10*Math.max(1,Math.abs(a)))throw Error('numeric receipt mismatch '+p)}else if(a&&typeof a==='object'){if(!b||Object.keys(a).join()!==Object.keys(b).join())throw Error('receipt shape '+p);for(const k of Object.keys(a))compare(a[k],b[k],p+'/'+k)}else if(a!==b)throw Error('receipt mismatch '+p)}compare(out,expected);console.log('PASS: four-edge logical design, dense oracle and synthetic replay');}else console.log(JSON.stringify(out,null,2));
