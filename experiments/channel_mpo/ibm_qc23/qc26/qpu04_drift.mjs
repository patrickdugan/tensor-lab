// PGC-QPU04: independent two-epoch calibration model. No device access.
export const VERSION='pgc-qpu04-v1';
export const BLOCKS=['qc26a','q5','q23'];
export const SX=[3,5,21,23];
export const EDGES=[[4,5],[5,6],[16,23],[22,23]];
export const DRIFT_RULE={sx_sigma_per_sqrt_hour:0.003,cz_sigma_per_sqrt_hour:0.01,
  sx_se_floor:0.003,cz_se_floor:0.005,max_fresh_age_hours:4,
  model:'independent Gaussian random walks on the small-error phase branch; plug-in posterior mean'};
const requireOK=(ok,msg)=>{if(!ok)throw Error('PGC-QPU04: '+msg);};
const timestamp=s=>{const t=Date.parse(s);requireOK(typeof s==='string'&&/(Z|[+-]\d\d:\d\d)$/.test(s)&&Number.isFinite(t),'offset-aware timestamp required');return t;};
const finite=x=>typeof x==='number'&&Number.isFinite(x);
export function parameterSpecs(){
  const out=SX.map(q=>({key:'sx.'+q,block:'qc26a',kind:'sx',site:q}));
  for(const [a,b] of EDGES){
    const edge=a+'_'+b,block=b<10?'q5':'q23';
    out.push({key:'cz.'+edge+'.delta',block,kind:'cz',edge});
    for(const q of [a,b])out.push({key:'cz.'+edge+'.alpha.'+q,block,kind:'cz',edge,site:q});
  }
  return out;
}
export function toChannel(values){
  const p={sx:{},cz:{}};
  for(const s of parameterSpecs()){
    const v=values[s.key];requireOK(finite(v),'missing value '+s.key);
    if(s.kind==='sx')p.sx[s.site]=v;
    else{
      const c=p.cz[s.edge]??={alpha:{}};
      if(s.site===undefined)c.delta=v;else c.alpha[s.site]=v;
    }
  }
  return p;
}
export function validatePack(pack){
  requireOK(pack?.version===VERSION&&pack.backend==='ibm_kingston','version/backend');
  requireOK(Array.isArray(pack.epochs)&&pack.epochs.length===2,'exactly two independent calibration rounds required');
  const seen=new Set(),specs=parameterSpecs();
  for(const e of pack.epochs){
    requireOK(typeof e.root==='string'&&e.root.length>0,'receipt directory missing');
    requireOK(Object.keys(e.observations??{}).sort().join()===specs.map(s=>s.key).sort().join(),'parameter set mismatch');
    for(const block of BLOCKS){
      const b=e.blocks?.[block];
      requireOK(b&&typeof b.job_id==='string'&&b.job_id.length>0&&!seen.has(b.job_id),'missing/reused calibration job');
      seen.add(b.job_id);
      requireOK(/^[0-9a-f]{64}$/.test(b.plan_sha256),'invalid plan hash');
      requireOK(timestamp(b.calibration_timestamp)<=timestamp(b.submitted_at),'job before calibration snapshot');
    }
    for(const s of specs){
      const v=e.observations[s.key],bound=s.kind==='sx'?.1:.4;
      requireOK(v.block===s.block&&finite(v.estimate)&&Math.abs(v.estimate)<=bound&&finite(v.se)&&v.se>0,'bad estimate/SE/branch '+s.key);
    }
    requireOK(Array.isArray(e.sources)&&e.sources.length>=12,'missing provenance files');
    const paths=new Set();
    for(const x of e.sources){
      requireOK(typeof x.path==='string'&&!x.path.startsWith('/')&&!x.path.includes('\\')&&!x.path.split('/').includes('..')&&!paths.has(x.path)&&/^[0-9a-f]{64}$/.test(x.sha256),'invalid provenance path/hash');
      paths.add(x.path);
    }
  }
  const [old,fresh]=pack.epochs;
  const latestOld=Math.max(...BLOCKS.map(k=>timestamp(old.blocks[k].submitted_at)));
  for(const k of BLOCKS){
    requireOK(timestamp(fresh.blocks[k].submitted_at)>latestOld,'rounds overlap');
    requireOK(timestamp(fresh.blocks[k].calibration_timestamp)>timestamp(old.blocks[k].calibration_timestamp),'fresh calibration cycle not established for '+k);
    requireOK(Number.isInteger(fresh.blocks[k].ordering_seed),'fresh block is not independently shuffled: '+k);
  }
  return pack;
}
export function kalmanTwo(old,newer,dtHours,forecastHours,q,seFloor){
  requireOK([dtHours,forecastHours,q,seFloor].every(x=>finite(x)&&x>=0)&&seFloor>0,'invalid drift variance/time');
  const p0=Math.max(old.se**2,seFloor**2),r=Math.max(newer.se**2,seFloor**2);
  const pred=p0+q*q*dtHours,gain=pred/(pred+r);
  const mean=old.estimate+gain*(newer.estimate-old.estimate),variance=pred*r/(pred+r);
  return {mean,variance,forecast_variance:variance+q*q*forecastHours,gain,
    standardized_innovation:(newer.estimate-old.estimate)/Math.sqrt(pred+r),
    process_variance:q*q*dtHours,latest_variance:r,elapsed_hours:dtHours,forecast_hours:forecastHours};
}
export function derive(pack,compiledAt,calibrationTimestamp){
  validatePack(pack);
  const [old,fresh]=pack.epochs,t=timestamp(compiledAt),ct=timestamp(calibrationTimestamp);
  const latest={},filtered={},posterior={};
  for(const s of parameterSpecs()){
    const a=old.blocks[s.block],b=fresh.blocks[s.block],prev=timestamp(a.submitted_at),now=timestamp(b.submitted_at);
    const dt=(now-prev)/3600000,f=(t-now)/3600000;
    requireOK(f>=0&&f<=DRIFT_RULE.max_fresh_age_hours,'fresh calibration stale or later than build: '+s.block);
    requireOK(ct>=timestamp(b.calibration_timestamp)&&ct<=t,'evaluation calibration timestamp precedes fresh source or follows build');
    const q=s.kind==='sx'?DRIFT_RULE.sx_sigma_per_sqrt_hour:DRIFT_RULE.cz_sigma_per_sqrt_hour;
    const floor=s.kind==='sx'?DRIFT_RULE.sx_se_floor:DRIFT_RULE.cz_se_floor;
    latest[s.key]=fresh.observations[s.key].estimate;
    posterior[s.key]=kalmanTwo(old.observations[s.key],fresh.observations[s.key],dt,f,q,floor);
    filtered[s.key]=posterior[s.key].mean;
  }
  return {rule:DRIFT_RULE,latest:toChannel(latest),drift:toChannel(filtered),posterior,
    limitations:['submission time is a proxy for acquisition time','process noise fixed before outcomes, not fitted to two epochs',
      'independent Gaussian approximation with SE floors; cross-parameter covariance omitted','plug-in prediction, not posterior integration',
      'no mid-evaluation adaptation; fixed mean across the evaluation block']};
}
