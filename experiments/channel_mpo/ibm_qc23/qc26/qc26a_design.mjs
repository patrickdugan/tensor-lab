// QC-26A offline plan and phase-estimator design test. No IBM calls.
import fs from 'node:fs';
function makePlan(){
 const sites=[3,5,21,23],edges=[[4,5],[6,5],[16,23],[22,23],[4,3],[22,21]],all=[3,4,5,6,16,21,22,23],shots=384,rows=[];
 const add=(r)=>rows.push({...r,shots});
 for(let state of [0,1])add({id:"spam_"+state,family:"spam",path:all,state});
 for(let m of [0,4,12,24])for(let axis of ["Z","Y"])add({id:`sx_${m}_${axis}`,family:"sx",path:sites,m,axis});
 for(let [c,t] of edges)for(let state of [0,1])for(let m of [0,2,8])for(let axis of ["X","Y"])add({id:`cz_${c}_${t}_${state}_${m}_${axis}`,family:"cz",path:[c,t],state,m,axis});
 // Deterministic Fisher-Yates ordering, independent of data.
 let seed=261101;const rng=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
 for(let i=rows.length-1;i>0;i--){let j=Math.floor(rng()*(i+1));[rows[i],rows[j]]=[rows[j],rows[i]];}
 return {version:"QC26A-v1",status:"proposal; no owner authorization",backend:"ibm_kingston",sites,edges,shots,ordering_seed:261101,rows,cap_seconds:20,estimate:{shot_repetition_delay_s:250e-6,per_circuit_overhead_allowance_s:.1,per_shot_circuit_allowance_s:10e-6,nominal_seconds:rows.length*shots*(.00025+.00001)+.1*rows.length}};
}
function analyze(plan,counts){
 const means={};for(let r of plan.rows){let c=counts[r.id],n=r.path.length,m=Array(n).fill(0),shots=0;if(!c)throw Error("missing "+r.id);for(let [s,k] of Object.entries(c)){if(s.length!==n||!/^[01]+$/.test(s)||!Number.isInteger(k)||k<0)throw Error("bad counts");shots+=k;for(let j=0;j<n;j++)m[j]+=(s[n-1-j]==="1"?-1:1)*k;}if(shots!==r.shots)throw Error("shot mismatch");means[r.id]=m.map(v=>v/shots);}
 const spam={};let a=plan.rows.find(r=>r.id==="spam_0"),b=plan.rows.find(r=>r.id==="spam_1");
 for(let [i,p] of a.path.entries()){let e0=(1-means[a.id][i])/2,e1=(1+means[b.id][i])/2,contrast=1-e0-e1;if(contrast<.5)throw Error("SPAM contrast too low");spam[p]={e0,e1,contrast,bias:e1-e0};}
 const corr=(id,j,p)=>(means[id][j]-spam[p].bias)/spam[p].contrast;
 const sx={};for(let [i,p] of plan.sites.entries()){let ms=[0,4,12,24],data=ms.map(m=>[m,corr(`sx_${m}_Z`,i,p),corr(`sx_${m}_Y`,i,p)]);let best={loss:Infinity};
 // Profile out per-repetition visibility, allow a shared preparation phase.
 // Grid is fixed, deterministic; no extrapolation beyond ±0.10 rad.
 for(let k=-200;k<=200;k++){let d=k*.0005;for(let l=-20;l<=20;l++){let phase=l*.01,loss=0;for(let [m,z,y] of data){let c=Math.cos(m*d+phase),s=-Math.sin((m+1)*d+phase),v=Math.max(0,Math.min(1.1,(z*c+y*s)/(c*c+s*s)));loss+=(z-v*c)**2+(y-v*s)**2;}if(loss<best.loss)best={offset_rad:d,phase_rad:phase,loss};}}
 sx[p]={...best,at_boundary:Math.abs(best.offset_rad)>=.099999};
 }
 const cz={};for(let [c,t] of plan.edges){let states=[];for(let state of [0,1]){let ms=[0,2,8],phases=[],visibility=[];for(let m of ms){let x=corr(`cz_${c}_${t}_${state}_${m}_X`,1,t),y=corr(`cz_${c}_${t}_${state}_${m}_Y`,1,t),ph=Math.atan2(y,x);if(phases.length){while(ph-phases.at(-1)>Math.PI)ph-=2*Math.PI;while(ph-phases.at(-1)<-Math.PI)ph+=2*Math.PI;}phases.push(ph);visibility.push(Math.hypot(x,y));}const mx=ms.reduce((a,b)=>a+b)/ms.length,my=phases.reduce((a,b)=>a+b)/ms.length;let slope=ms.reduce((s,m,i)=>s+(m-mx)*(phases[i]-my),0)/ms.reduce((s,m)=>s+(m-mx)**2,0);states.push({slope_rad:slope,intercept_rad:my-slope*mx,min_visibility:Math.min(...visibility),phases,control_means:ms.map(m=>[means[`cz_${c}_${t}_${state}_${m}_X`][0],means[`cz_${c}_${t}_${state}_${m}_Y`][0]])});}cz[`${c}_${t}`]={local_phase_rad:(states[0].slope_rad+states[1].slope_rad)/2,zz_phase_rad:(states[0].slope_rad-states[1].slope_rad)/2,states,low_visibility:states.some(x=>x.min_visibility<.5)};}
 return {spam,sx,cz};
}
function synthetic(plan,analyze){
 const truth={sx:{3:.004,5:-.025,21:0,23:.04},cz:{"4_5":.032,"6_5":-.022,"16_23":.047,"22_23":-.038,"4_3":0,"22_21":0},local:{"4_5":.013,"6_5":-.012,"16_23":.01,"22_23":.006,"4_3":-.008,"22_21":.009}};
 let seed=260126;const rng=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
 const abs=x=>Math.abs(x),mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
 function generate(zero=false){let counts={};for(let r of plan.rows){let z=r.path.map((p,i)=>{if(r.family==="spam")return 1-2*r.state;if(r.family==="sx"){let d=zero?0:truth.sx[p],phase=.017*(i-1.5),v=.99*Math.exp(-.0018*r.m);return r.axis==="Z"?v*Math.cos(r.m*d+phase):-v*Math.sin((r.m+1)*d+phase);}if(i===0)return 1-2*r.state;let key=r.path.join("_"),d=zero?0:truth.cz[key],phase=(r.state===0?.035:-.025)+r.m*(truth.local[key]+(r.state===0?d:-d)),v=.97*Math.exp(-.004*r.m);return v*(r.axis==="X"?Math.cos(phase):Math.sin(phase));});let probs=z.map((v,i)=>{let p=r.path[i],e0=.004+(p%5)*.002,e1=.008+(p%3)*.003;return (1-((e1-e0)+(1-e0-e1)*v))/2;});let ct={};for(let j=0;j<r.shots;j++){let bits=probs.map(p=>rng()<p?"1":"0").reverse().join("");ct[bits]=(ct[bits]||0)+1;}counts[r.id]=ct;}return counts;}
 let trials=[];const N=48;
 for(let zero of [false,true])for(let i=0;i<N;i++){let a=analyze(plan,generate(zero));trials.push({zero,sx:Object.fromEntries(Object.entries(a.sx).map(([p,v])=>[p,v.offset_rad])),cz:Object.fromEntries(Object.entries(a.cz).map(([p,v])=>[p,v.zz_phase_rad])),local:Object.fromEntries(Object.entries(a.cz).map(([p,v])=>[p,v.local_phase_rad])),boundaries:Object.values(a.sx).filter(v=>v.at_boundary).length,lowVisibility:Object.values(a.cz).filter(v=>v.low_visibility).length});}
 const summary={};for(let zero of [false,true]){let t=trials.filter(v=>v.zero===zero),arm={};for(let family of ["sx","cz","local"]){arm[family]={};for(let [p,val] of Object.entries(truth[family])){let target=zero&&family!=="local"?0:val,values=t.map(v=>v[family][p]);arm[family][p]={truth:target,mean:mean(values),rmse:Math.sqrt(mean(values.map(v=>(v-target)**2))),signed_correct:target===0?null:values.filter(v=>Math.sign(v)===Math.sign(target)).length,over_0_02:values.filter(v=>abs(v)>.02).length};}}summary[zero?"null":"injected"]={...arm,boundaries:t.reduce((s,v)=>s+v.boundaries,0),lowVisibility:t.reduce((s,v)=>s+v.lowVisibility,0)};}
 return {label:"SYNTHETIC ONLY; independent Bernoulli readout model, no hardware data",trials_per_arm:N,truth,summary,trials};
}
const mode=process.argv[2]||'synthetic';const plan=makePlan();if(mode==='synthetic'){const result=synthetic(plan,analyze);fs.writeFileSync(process.argv[3]||'/tmp/qc26a_synthetic.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result.summary,null,2));}else if(mode==='plan'){fs.writeFileSync(process.argv[3]||'/tmp/qc26a_plan.json',JSON.stringify(plan,null,2)+'\n');}else if(mode==='analyze'){const raw=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));const result=analyze(plan,raw);fs.writeFileSync(process.argv[4]||'/tmp/qc26a_analysis.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));}else throw Error('unknown mode');
