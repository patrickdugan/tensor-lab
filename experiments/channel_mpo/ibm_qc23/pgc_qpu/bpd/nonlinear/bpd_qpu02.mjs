// BPD-QPU02: deterministic, dependency-free discrete posterior design gym.
function runGym(trials=32) {
  const reps=[1,2,4,8], S=12, K=S+1, E=2, pi=Math.PI;
  const H=p=>p.reduce((v,x)=>x>0?v-x*Math.log(x):v,0);
  const mkRng=seed=>{let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};};
  const seedFor=(...a)=>a.reduce((s,x)=>Math.imul(s^x,16777619)>>>0,2166136261)||1;
  const locals=[];
  for(const alpha of [-.12,0,.16])for(const delta of [-.035,.035])for(let alias=0;alias<2;alias++)
    locals.push({alpha:alpha+alias*pi/2,delta:delta+alias*pi/2,alias});
  const L=locals.length, modes=[{vis:.97,ret:.99},{vis:.88,ret:.95}], G=L*modes.length;
  const states=[], prior=[];
  for(let a=0;a<L;a++)for(let b=0;b<L;b++)for(let mode=0;mode<modes.length;mode++){
    states.push({local:[a,b],mode,group:[a*2+mode,b*2+mode]});
    prior.push(locals[a].alias===locals[b].alias?0.7:0.3);
  }
  const norm=x=>{let s=x.reduce((a,b)=>a+b,0);return x.map(v=>v/s);};
  const p0=norm(prior);
  const actions=[];
  for(let edge=0;edge<E;edge++)for(const m of reps)for(let b=0;b<2;b++)for(const axis of ["X","Y"])
    actions.push({id:actions.length,edge,m,b,axis,cost:1+.04*m});
  const held=[];
  for(let edge=0;edge<E;edge++)for(const m of [3,5,7])for(let b=0;b<2;b++)for(const axis of ["X","Y"])held.push({edge,m,b,axis});
  function probability(a,local,mode,gamma=0,shift=0){
    let z=0;const phi=j=>a.m*(local.alpha+local.delta+shift)+j*(pi-2*local.delta);
    const f=j=>a.axis==="X"?Math.cos(phi(j)):Math.sin(phi(j));
    if(a.b===0)z=f(0);else{
      z=(1-gamma)**a.m*f(a.m);
      for(let j=1;j<=a.m;j++)z+=gamma*(1-gamma)**(j-1)*f(j);
    }
    return Math.min(1-1e-12,Math.max(1e-12,(1+.976*mode.vis*mode.ret**a.m*z)/2));
  }
  function binomial(p){
    const arr=[(1-p)**S];
    for(let k=1;k<=S;k++)arr.push(arr[k-1]*(S-k+1)/k*p/(1-p));
    return norm(arr);
  }
  const table=actions.map(a=>Array.from({length:G},(_,g)=>binomial(probability(a,locals[g>>1],modes[g%2]))));
  const condH=table.map(t=>t.map(H));
  const heldTable=held.map(a=>Array.from({length:G},(_,g)=>probability(a,locals[g>>1],modes[g%2])));
  function marginal(w){
    let g=[new Float64Array(G),new Float64Array(G)];
    for(let i=0;i<w.length;i++){const s=states[i];g[0][s.group[0]]+=w[i];g[1][s.group[1]]+=w[i];}
    return g;
  }
  function scoreDense(w,a){
    const pred=new Float64Array(K);let ce=0;const t=table[a.id],h=condH[a.id];
    for(let i=0;i<w.length;i++){const g=states[i].group[a.edge],wi=w[i];ce+=wi*h[g];for(let k=0;k<K;k++)pred[k]+=wi*t[g][k];}
    return (H(pred)-ce)/a.cost;
  }
  function scoreGroup(g,a,aliasOnly=false){
    const pred=new Float64Array(K);let ce=0;const t=table[a.id],h=condH[a.id];
    const ag=[new Float64Array(K),new Float64Array(K)],aw=[0,0];
    for(let i=0;i<G;i++){const wi=g[i];ce+=wi*h[i];const al=locals[i>>1].alias;aw[al]+=wi;
      for(let k=0;k<K;k++){const v=wi*t[i][k];pred[k]+=v;if(aliasOnly)ag[al][k]+=v;}}
    if(aliasOnly){ce=0;for(let al=0;al<2;al++)if(aw[al]>0)ce+=aw[al]*H(ag[al].map(v=>v/aw[al]));}
    return (H(pred)-ce)/a.cost;
  }
  const select=(eligible,score)=>{let best=null,v=-Infinity;for(const a of eligible){const q=score(a);if(q>v+1e-12){best=a;v=q;}}return best;};
  function drawIndex(w,r){let z=r();for(let i=0;i<w.length;i++){z-=w[i];if(z<=0)return i;}return w.length-1;}
  const scenarios=["matched","offgrid","drift_relax"];
  const policies=["fixed_even","random","dense_eig","group_exact","group_alias"];
  const budgets=[6.5,13];const horizon=13;
  const records=[];let equivalence={paths:0,action_mismatches:0,posterior_max_gap:0,max_score_gap:0};
  for(let sc=0;sc<scenarios.length;sc++)for(let trial=0;trial<trials;trial++){
    const r=mkRng(seedFor(20261127,sc,trial));
    const trueS=states[drawIndex(p0,r)];
    const truth=trueS.local.map(i=>({...locals[i]}));
    if(sc>0)for(const l of truth){l.alpha+=(r()-.5)*.09;l.delta+=(r()-.5)*.035;}
    const drift=(r()-.5)*.14, gamma=.045;
    const truthP=(a,spent)=>probability(a,truth[a.edge],modes[trueS.mode],sc===2?gamma:0,sc===2&&spent>=6.5?drift:0);
    let densePath=null;
    for(let pol=0;pol<policies.length;pol++){
      const policy=policies[pol], rr=mkRng(seedFor(801,sc,trial));
      let w=p0.slice(),spent=0,step=0,work=0,used=new Set(),path=[],snapshots=[];
      function record(budget){
        const g=marginal(w);
        let rmse=0,aliasOK=0;
        for(let e=0;e<E;e++){let q=0;for(let i=0;i<G;i++)if(locals[i>>1].alias)q+=g[e][i];aliasOK+=Number((q>=.5-1e-9?1:0)===truth[e].alias);}
        for(let i=0;i<held.length;i++){const a=held[i],t=heldTable[i];let q=0;for(let j=0;j<G;j++)q+=g[a.edge][j]*t[j];rmse+=(q-truthP(a,budget))**2;}
        snapshots.push({budget,spent,shots:step*S,work,alias_both:Number(aliasOK===E),alias_fraction:aliasOK/E,rmse:Math.sqrt(rmse/held.length)});
      }
      for(const budget of budgets){
        for(;;){
          const eligible=actions.filter(a=>!used.has(a.id)&&a.cost+spent<=budget+1e-10&&(policy!=="fixed_even"||a.m%2===0));
          if(!eligible.length)break;
          let a;
          if(policy==="fixed_even")a=eligible[step%eligible.length];
          else if(policy==="random")a=eligible[Math.floor(rr()*eligible.length)];
          else if(policy==="dense_eig"){a=select(eligible,a=>scoreDense(w,a));work+=eligible.length*states.length*K;}
          else {
            const g=marginal(w);
            a=select(eligible,a=>scoreGroup(g[a.edge],a,policy==="group_alias"));
            work+=states.length*2+eligible.length*G*K;
            if(policy==="group_exact"&&trial===0){
              for(const x of eligible)equivalence.max_score_gap=Math.max(equivalence.max_score_gap,Math.abs(scoreDense(w,x)-scoreGroup(g[x.edge],x)));
            }
          }
          const p=truthP(a,spent),dr=mkRng(seedFor(991,sc,trial,step,a.id));let k=0;
          for(let j=0;j<S;j++)k+=Number(dr()<p);
          w=norm(w.map((wi,i)=>wi*table[a.id][states[i].group[a.edge]][k]));
          path.push(a.id);spent+=a.cost;step++;used.add(a.id);
        }
        record(budget);
      }
      if(policy==="dense_eig")densePath={path,w};
      if(policy==="group_exact"){
        equivalence.paths++;if(path.join(",")!==densePath.path.join(","))equivalence.action_mismatches++;
        for(let i=0;i<w.length;i++)equivalence.posterior_max_gap=Math.max(equivalence.posterior_max_gap,Math.abs(w[i]-densePath.w[i]));
      }
      for(const v of snapshots)records.push({scenario:scenarios[sc],trial,policy,...v});
    }
  }
  const summary=[];
  for(const scenario of scenarios)for(const budget of budgets)for(const policy of policies){
    const rows=records.filter(x=>x.scenario===scenario&&x.budget===budget&&x.policy===policy);
    const mean=k=>rows.reduce((s,x)=>s+x[k],0)/rows.length;
    summary.push({scenario,budget,policy,alias_both:mean("alias_both"),alias_fraction:mean("alias_fraction"),rmse:mean("rmse"),spent:mean("spent"),shots:mean("shots"),score_work:mean("work")});
  }
  return {version:"BPD-QPU02-v0.1",evidence:"offline discrete nonlinear binomial simulation; toy cost units, not IBM seconds",trials_per_scenario:trials,states:states.length,local_groups:G,actions:actions.length,shots_per_action:S,budgets,scenarios,policies,equivalence,summary,records,
    limitations:["Only two edges; local sufficient marginal factorization is exact for this likelihood family, global posterior still updated.","Off-grid and drift/relax scenarios are misspecified. No claim about hardware or measured runtime.","Score-work counts predictive likelihood-bin accumulations plus marginalization, excludes common updates and table construction.","Budget checkpoints constrain action selection; each action at most once. All adaptive policies use full posterior updates."]};
}
const trials=Number(process.argv[2]||32);
if(!Number.isInteger(trials)||trials<1)throw Error("positive integer trial count required");
console.log(JSON.stringify(runGym(trials),null,2));
