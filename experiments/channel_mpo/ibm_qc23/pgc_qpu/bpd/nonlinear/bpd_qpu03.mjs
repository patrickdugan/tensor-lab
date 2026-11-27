// BPD-QPU03: learned integer phase groups, exact oracle and certified approximate design.
function runQPU03(trials=24){
 const M=8,D=3,S=10,K=S+1,TAU=2*Math.PI,EPS=1e-10;
 const norm=x=>{const s=x.reduce((a,b)=>a+b,0);return x.map(v=>v/s)};
 const H=p=>p.reduce((s,v)=>v>0?s-v*Math.log(v):s,0);
 const mod=x=>((x%M)+M)%M;
 const rng=seed=>{let x=seed>>>0||1;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296}};
 const hash=(...a)=>a.reduce((s,x)=>Math.imul(s^x,16777619)>>>0,2166136261);
 const draw=(p,r)=>{let z=r();for(let i=0;i<p.length;i++){z-=p[i];if(z<=0)return i}return p.length-1};
 const dot=(a,x)=>a.reduce((s,v,j)=>s+v*x[j],0);
 const vis=[.97,.84];
 const states=[], prior=[];
 for(let a=0;a<M;a++)for(let b=0;b<M;b++)for(let c=0;c<M;c++)for(let v=0;v<2;v++){
   const x=[a,b,c],alias=(a>=4?1:0)|(b>=4?2:0)|(c>=4?4:0);
   states.push({x,v,alias});prior.push(alias===0||alias===7?2:1);
 }
 const p0=norm(prior), N=states.length;
 const rows=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,-1,0],[0,1,1],[0,1,-1],[1,0,1],[1,0,-1],[1,1,1],[2,1,0],[0,2,1],[1,0,2],[2,0,0],[0,2,0],[0,0,2]];
 const scoutIds=[13,14,15];
 // Coupling nonlinearities on selected cross rows, invisible to a pure integer projection.
 const nonlinear=new Map([[3,2],[5,0],[7,1],[9,0],[10,2],[11,0],[12,1]]);
 const candidates=[];
 for(let i=0;i<rows.length;i++)for(let ax=0;ax<2;ax++)candidates.push({id:candidates.length,row:i,ax,cost:rows[i].filter(v=>v!==0).length>1?1.25:1});
 const heldRows=[[1,2,-1],[2,-1,1],[1,1,-2],[1,-1,-1],[3,0,0],[0,3,0],[0,0,3]];
 function phase(row,x,extra,amp){return TAU*dot(row,x)/M+(extra===undefined?0:amp*Math.sin(TAU*x[extra]/M))}
 function prob(ph,v,ax){const z=ax?Math.sin(ph):Math.cos(ph);return Math.max(1e-12,Math.min(1-1e-12,(1+.976*vis[v]*z)/2))}
 function binom(p){let t=[(1-p)**S];for(let k=1;k<=S;k++)t.push(t[k-1]*(S-k+1)/k*p/(1-p));return norm(t)}
 const allVectors=[];
 for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++)if(a||b||c)allVectors.push([a,b,c]);
 function mask(row){return row.reduce((m,x,j)=>m|((Math.abs(x)%2)<<j),0)}
 function extend(span,r){const out=new Set(span);for(const x of span)out.add(x^mask(r));return out}
 const entropyContinuity=t=>{t=Math.max(0,Math.min(t,1-1/K));return t===0?0:-t*Math.log(t)-(1-t)*Math.log(1-t)+t*Math.log(K-1)};
 const scenes=[{name:"integer",amp:0,drift:false},{name:"nonlinear",amp:.35,drift:false},{name:"drift_offgrid",amp:.35,drift:true}];
 const policies=["random","dense","oracle_group","learned_group","certified_group","smith_rank_first"];
 const budgets=[5,10], records=[], learning=[], agreement=[], bounds={checked:0,max_violation:0};
 for(let sc=0;sc<scenes.length;sc++){
   const cfg=scenes[sc],calR=rng(hash(20261127,sc,50)),train=[];
   const selected=new Set();while(selected.size<40)selected.add(Math.floor(calR()*N));for(const i of selected)train.push(states[i]);
   // Learn a single sparse integer observation row from model-generated X/Y calibration quadratures.
   const learned=rows.map((row,i)=>{
     const extra=nonlinear.get(i);
     const loss=v=>train.reduce((s,t)=>{
       const p=phase(row,t.x,extra,cfg.amp),q=phase(v,t.x,undefined,0);
       return s+(Math.cos(p)-Math.cos(q))**2+(Math.sin(p)-Math.sin(q))**2;
     },0)/train.length;
     let best=allVectors[0],err=Infinity;for(const v of allVectors){const z=loss(v);if(z<err-1e-12){best=v;err=z}}
     const val=states.filter((_,j)=>!selected.has(j)).reduce((s,t)=>{const p=phase(row,t.x,extra,cfg.amp),q=phase(best,t.x,undefined,0);return s+(Math.cos(p)-Math.cos(q))**2+(Math.sin(p)-Math.sin(q))**2},0)/(N-train.length);
     return {row:i,learned:best,oracle:row,train_mse:err,validation_mse:val,correct:best.join()===row.join()};
   });
   learning.push({scenario:cfg.name,train_states:train.length,validation_states:N-train.length,rows:learned});
   // Precompute exact state likelihoods and approximate learned groups. Oracle groups use
   // residue, visibility and (where needed) the extra coordinate, all known to the model.
   const tables=candidates.map(a=>{
     const row=rows[a.row], extra=cfg.amp?nonlinear.get(a.row):undefined;
     const groups=new Map(), approx=new Map(), oi=[], ai=[], truth=[], hs=[], tv=[], hd=[];
     for(const s of states){
       const res=mod(dot(row,s.x)),qres=mod(dot(learned[a.row].learned,s.x));
       const key=[res,s.v,extra===undefined?-1:s.x[extra]].join(":");
       const qkey=[qres,s.v].join(":");
       const p=prob(phase(row,s.x,extra,cfg.amp),s.v,a.ax);
       const t=binom(p); truth.push(t);hs.push(H(t));
       if(!groups.has(key))groups.set(key,{i:groups.size,t,h:H(t)});
       const g=groups.get(key);oi.push(g.i);
       if(g.t.some((z,k)=>Math.abs(z-t[k])>1e-12))throw Error("oracle grouping not sufficient");
       if(!approx.has(qkey)){const qt=binom(prob(TAU*qres/M,s.v,a.ax));approx.set(qkey,{i:approx.size,t:qt,h:H(qt)})}
       const q=approx.get(qkey);ai.push(q.i);
       tv.push(t.reduce((z,v,k)=>z+Math.abs(v-q.t[k]),0)/2);hd.push(Math.abs(hs.at(-1)-q.h));
     }
     return {truth,hs,oi,ai,og:[...groups.values()],ag:[...approx.values()],tv,hd,
       exactApprox:Math.max(...tv)<1e-12};
   });
   function dense(w,a){
     const t=tables[a.id],mix=new Float64Array(K);let h=0;
     for(let i=0;i<N;i++){h+=w[i]*t.hs[i];for(let j=0;j<K;j++)mix[j]+=w[i]*t.truth[i][j]}
     return (H(mix)-h)/a.cost;
   }
   function grouped(w,a,oracle,cache){
     const t=tables[a.id],ids=oracle?t.oi:t.ai,g=oracle?t.og:t.ag;
     // X/Y of a row share the grouping; reuse the projection marginal within a decision.
     const key=a.row+":"+oracle;let mass=cache.get(key),work=0;
     if(!mass){mass=new Float64Array(g.length);for(let i=0;i<N;i++)mass[ids[i]]+=w[i];cache.set(key,mass);work=N}
     const mix=new Float64Array(K);let h=0;
     for(let i=0;i<g.length;i++){h+=mass[i]*g[i].h;for(let j=0;j<K;j++)mix[j]+=mass[i]*g[i].t[j]}
     work+=g.length*K;
     let error=0;
     if(!oracle&&!t.exactApprox){
       let d=0,e=0;for(let i=0;i<N;i++){d+=w[i]*t.tv[i];e+=w[i]*t.hd[i]}
       work+=2*N;error=(entropyContinuity(d)+e)/a.cost;
     }
     return {a,score:(H(mix)-h)/a.cost,error,work};
   }
   function best(list){return list.reduce((b,x)=>!b||x.score>b.score+EPS||(Math.abs(x.score-b.score)<=EPS&&x.a.id<b.a.id)?x:b,null)}
   for(let trial=0;trial<trials;trial++){
     const rr=rng(hash(102,sc,trial)),truth=states[draw(p0,rr)];
     const offset=truth.x.map(()=>cfg.drift?(rr()-.5)*.24:0),shift=truth.x.map(()=>cfg.drift?(rr()-.5)*.65:0);
     function trueP(a,spent){const x=truth.x.map((v,j)=>v+offset[j]+(cfg.drift&&spent>=5?shift[j]:0));
       return prob(phase(rows[a.row],x,nonlinear.get(a.row),cfg.amp),truth.v,a.ax)}
     function sample(a,spent,step){const r=rng(hash(909,sc,trial,step,a.id));let k=0,p=trueP(a,spent);for(let j=0;j<S;j++)k+=Number(r()<p);return k}
     function update(w,a,k){return norm(w.map((v,i)=>v*tables[a.id].truth[i][k]))}
     let start=p0.slice();for(const j of scoutIds)for(let ax=0;ax<2;ax++){const a=candidates[2*j+ax];start=update(start,a,sample(a,-1,-1-a.id))}
     let densePath=null;
     for(let pol=0;pol<policies.length;pol++){
       const policy=policies[pol],r=rng(hash(601,sc,trial)),used=new Set(scoutIds.flatMap(i=>[2*i,2*i+1]));
       let w=start.slice(),spent=0,step=0,work=0,fallbacks=0,rankspan=new Set([0]),path=[],snap=[];
       for(const budget of budgets){
         for(;;){
           const eligible=candidates.filter(a=>!used.has(a.id)&&spent+a.cost<=budget+EPS);if(!eligible.length)break;
           let a;
           if(policy==="random")a=eligible[Math.floor(r()*eligible.length)];
           else if(policy==="dense"){a=best(eligible.map(a=>({a,score:dense(w,a)}))).a;work+=eligible.length*N*K}
           else {
             const cache=new Map(),oracle=policy==="oracle_group"||policy==="smith_rank_first";
             const scores=eligible.map(a=>grouped(w,a,oracle,cache));work+=scores.reduce((s,x)=>s+x.work,0);
             if(trial===0&&policy==="certified_group")for(const x of scores){const gap=Math.abs(dense(w,x.a)-x.score);bounds.checked++;bounds.max_violation=Math.max(bounds.max_violation,gap-x.error)}
             if(policy==="certified_group"){
               const sorted=scores.slice().sort((a,b)=>(b.score+b.error)-(a.score+a.error)||a.a.id-b.a.id);
               let incumbent=null;
               for(const x of sorted){
                 if(incumbent&&x.score+x.error<incumbent.score-EPS)continue;
                 let score=x.score;
                 if(x.error>EPS){score=dense(w,x.a);fallbacks++;work+=N*K}
                 incumbent=best([...(incumbent?[incumbent]:[]),{a:x.a,score}]);
               }
               a=incumbent.a;
             }else if(policy==="smith_rank_first"){
               a=best(scores.map(x=>({...x,score:x.score+(extend(rankspan,rows[x.a.row]).size>rankspan.size?10:0)}))).a;
             }else a=best(scores).a;
           }
           const k=sample(a,spent,step);w=update(w,a,k);used.add(a.id);rankspan=extend(rankspan,rows[a.row]);path.push(a.id);spent+=a.cost;step++;
         }
         // After the even scout only 2^D candidate integer shifts remain. Selected ideal
         // phase rows constrain their parity. A single quadrature can retain extra ambiguity.
         const aliasP=new Float64Array(1<<D);for(let i=0;i<N;i++)aliasP[states[i].alias]+=w[i];
         const predAlias=Array.from(aliasP).indexOf(Math.max(...aliasP)), kernel=2**D/rankspan.size;
         let brute=0;for(let bits=0;bits<1<<D;bits++){const shift=Array.from({length:D},(_,j)=>4*((bits>>j)&1));if(path.every(id=>mod(dot(rows[candidates[id].row],shift))===0))brute++}
         if(kernel!==brute)throw Error("Smith/parity kernel mismatch");
         let sq=0,n=0;
         for(const row of heldRows)for(let ax=0;ax<2;ax++){
           let p=0;for(let i=0;i<N;i++)p+=w[i]*prob(phase(row,states[i].x,undefined,0),states[i].v,ax);
           const x=truth.x.map((v,j)=>v+offset[j]+(cfg.drift&&spent>=5?shift[j]:0));
           sq+=(p-prob(phase(row,x,undefined,0),truth.v,ax))**2;n++;
         }
         snap.push({scenario:cfg.name,trial,policy,budget,spent,actions:step,score_work:work,fallbacks,kernel,alias_correct:Number(predAlias===truth.alias),alias_mass:aliasP[truth.alias],rmse:Math.sqrt(sq/n)});
       }
       if(policy==="dense")densePath={path,w};
       if(policy==="oracle_group"||policy==="certified_group"||policy==="learned_group"){
         let gap=0;for(let i=0;i<N;i++)gap=Math.max(gap,Math.abs(w[i]-densePath.w[i]));
         agreement.push({scenario:cfg.name,trial,policy,path_equal:path.join()===densePath.path.join(),posterior_gap:gap});
       }
       records.push(...snap);
     }
   }
 }
 const summary=[];for(const sc of scenes)for(const budget of budgets)for(const policy of policies){
   const rs=records.filter(r=>r.scenario===sc.name&&r.budget===budget&&r.policy===policy);
   const mean=k=>rs.reduce((s,x)=>s+x[k],0)/rs.length;
   summary.push({scenario:sc.name,budget,policy,rmse:mean("rmse"),alias_recovery:mean("alias_correct"),true_alias_mass:mean("alias_mass"),kernel_mean:mean("kernel"),kernel_one:mean("kernel")===1?1:rs.filter(x=>x.kernel===1).length/rs.length,score_work:mean("score_work"),fallbacks:mean("fallbacks"),spent:mean("spent")});
 }
 const equivalence=[];for(const sc of scenes)for(const policy of ["oracle_group","learned_group","certified_group"]){
   const a=agreement.filter(x=>x.scenario===sc.name&&x.policy===policy);
   equivalence.push({scenario:sc.name,policy,paths:a.length,mismatches:a.filter(x=>!x.path_equal).length,max_posterior_gap:Math.max(...a.map(x=>x.posterior_gap))});
 }
 if(equivalence.some(x=>x.policy!=="learned_group"&&x.mismatches)||bounds.max_violation>1e-9)throw Error("certified equivalence failed");
 return {version:"BPD-QPU03-v0.1",evidence:"offline model-generated finite-grid nonlinear binomial pilot; no hardware",M,D,states:N,shots_per_action:S,scout_actions:6,trials_per_scenario:trials,budgets,learning,equivalence,bounds,summary,agreement,records,
 limitations:["Calibration groups learned from model-generated noiseless quadratures, not hardware or trained neural BPD.","Oracle grouping knows integer residues, visibility and nonlinear extra coordinate; all policies update the exact joint posterior.","Score-work counts marginal accumulations, entropy-bin accumulations and weighted error sums; excludes shared likelihood updates, calibration and precomputation.","Certified policy uses an entropy-continuity bound and dense fallback; the certificate concerns the inference model and floating-point tolerance, not model correctness.","Integer kernel after even scout assumes ideal phase observations and ignores nonlinear response; a single quadrature can have additional aliases.","Drift truth differs from the stationary inference model. Held-out rows are noiseless generating probabilities of unmeasured cross/local phase combinations."]};
}
const trials=Number(process.argv[2]||24);
if(!Number.isInteger(trials)||trials<1)throw Error('positive trial count required');
console.log(JSON.stringify(runQPU03(trials),null,2));
