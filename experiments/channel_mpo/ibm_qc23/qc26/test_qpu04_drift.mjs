// Deterministic QPU04 calibration/filter smoke test and offline stress pilot.
import {VERSION,BLOCKS,parameterSpecs,derive,kalmanTwo,toChannel} from './qpu04_drift.mjs';
import {pathToFileURL} from 'node:url';
const assert=(v,msg)=>{if(!v)throw Error(msg)};
const near=(a,b,t=1e-12)=>Math.abs(a-b)<t;
export function fixturePack(){
  const epoch=(i)=>({root:'SYNTHETIC_FIXTURE/round'+i,
    blocks:Object.fromEntries(BLOCKS.map((k,j)=>[k,{job_id:'synthetic-'+i+'-'+k,
      plan_sha256:String(i+1).repeat(64),calibration_timestamp:'2026-11-2'+(5+i)+'T00:00:00Z',
      submitted_at:'2026-11-2'+(5+i)+'T00:0'+j+':00Z',ordering_seed:i?261127041+j:null}])),
    observations:Object.fromEntries(parameterSpecs().map((s,j)=>[s.key,{block:s.block,estimate:(j%3-1)*.012+i*.004,se:.006}])),
    sources:Array.from({length:12},(_,j)=>({path:'SYNTHETIC_FIXTURE/'+i+'/'+j,sha256:'a'.repeat(64)}))});
  return {version:VERSION,backend:'ibm_kingston',epochs:[epoch(0),epoch(1)]};
}
export function runCoreTests(){
  const pack=fixturePack(),build='2026-11-26T00:20:00Z',cal='2026-11-26T00:10:00Z';
  const d=derive(pack,build,cal),checks=[];
  const test=(name,v)=>{assert(v,name);checks.push(name)};
  const reject=(name,change)=>{const p=JSON.parse(JSON.stringify(pack));change(p);let threw=false;try{derive(p,build,cal)}catch{threw=true}test(name,threw)};
  test('exactly sixteen calibrated coordinates',parameterSpecs().length===16&&Object.keys(d.posterior).length===16);
  test('static channel equals fresh points',near(d.latest.sx['3'],pack.epochs[1].observations['sx.3'].estimate));
  const t=kalmanTwo({estimate:0,se:.01},{estimate:.02,se:.01},1,1,.01,.001);
  test('analytic gain 2/3',near(t.gain,2/3));
  test('analytic posterior mean',near(t.mean,.02*2/3));
  test('analytic predictive variance',near(t.forecast_variance,.0001*2/3+.0001));
  test('forecast variance nondecreasing',Object.values(d.posterior).every(x=>x.forecast_variance>=x.variance&&x.gain>0&&x.gain<1));
  test('drift shrinkage lies between independent observations',parameterSpecs().every(s=>{
    const a=pack.epochs[0].observations[s.key].estimate,b=pack.epochs[1].observations[s.key].estimate,m=d.posterior[s.key].mean;
    return m>=Math.min(a,b)-1e-12&&m<=Math.max(a,b)+1e-12;
  }));
  reject('duplicate calibration job rejected',p=>p.epochs[1].blocks.q5.job_id=p.epochs[0].blocks.q5.job_id);
  reject('unchanged calibration snapshot rejected',p=>p.epochs[1].blocks.q23.calibration_timestamp=p.epochs[0].blocks.q23.calibration_timestamp);
  reject('missing fresh permutation rejected',p=>delete p.epochs[1].blocks.q5.ordering_seed);
  reject('unqualified phase branch rejected',p=>p.epochs[1].observations['cz.5_6.delta'].estimate=1);
  reject('invalid uncertainty rejected',p=>p.epochs[1].observations['sx.3'].se=0);
  reject('future acquisition rejected',p=>p.epochs[1].blocks.q5.submitted_at='2026-11-26T01:00:00Z');
  reject('provenance traversal rejected',p=>p.epochs[1].sources[0].path='../bad');
  let stale=false;try{derive(pack,'2026-11-26T06:00:00Z',cal)}catch{stale=true}test('stale calibration rejected',stale);
  return {checks,passed:checks.length,parameter_count:16};
}
export function runPilot(trials=512){
  let seed=261127044;
  const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296};
  const normal=()=>Math.sqrt(-2*Math.log(rnd()))*Math.cos(2*Math.PI*rnd());
  const rows=[];
  for(const scenario of ['stationary','matched_random_walk','unmodelled_jump']){
    let mse={latest:0,pooled:0,drift:0},coverage=0,win=0;
    for(let i=0;i<trials;i++){
      let err={latest:0,pooled:0,drift:0};
      for(const s of parameterSpecs()){
        const q=s.kind==='sx'?.003:.01,se=s.kind==='sx'?.003:.006,floor=s.kind==='sx'?.003:.005;
        const a=.025*normal(),dt=1+3*rnd(),f=.15;
        const step=scenario==='stationary'?0:q*Math.sqrt(dt)*normal()+(scenario==='unmodelled_jump'?5*q*normal():0);
        const b=a+step,target=b+(scenario==='stationary'?0:q*Math.sqrt(f)*normal());
        const old={estimate:a+se*normal(),se},fresh={estimate:b+se*normal(),se};
        const filter=kalmanTwo(old,fresh,dt,f,q,floor),pool=kalmanTwo(old,fresh,dt,f,0,floor);
        err.latest+=(fresh.estimate-target)**2;
        err.pooled+=(pool.mean-target)**2;
        err.drift+=(filter.mean-target)**2;
        coverage+=Number(Math.abs(filter.mean-target)<=1.96*Math.sqrt(filter.forecast_variance));
      }
      for(const k of Object.keys(mse))mse[k]+=err[k];
      win+=Number(err.drift<err.latest);
    }
    const n=trials*parameterSpecs().length;
    rows.push({scenario,trials,rmse_rad:Object.fromEntries(Object.entries(mse).map(([k,v])=>[k,Math.sqrt(v/n)])),
      drift_interval_coverage:coverage/n,drift_beats_latest_trials:win});
  }
  return {label:'SYNTHETIC PARAMETER-SPACE PILOT; not circuit predictions or hardware',seed:261127044,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const tests=runCoreTests();
  console.log(JSON.stringify({tests,pilot:runPilot()},null,2));
}
