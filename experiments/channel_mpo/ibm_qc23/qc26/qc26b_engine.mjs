// QC-26B: heterogeneous native-gate channel engine (sparse Pauli Heisenberg propagation, exact ordering).
// Models C0..C3 on the original submitted native QASM:
//   C0 gate-local stochastic (twirled) noise        C1 = C0 with non-unital T1/T2 relaxation
//   C2 = C1 + coherent SX angle offsets (per physical qubit)
//   C3 = C2 + coherent CZ error exp(-i/2 (a_c Z_c + a_t Z_t + d Z_c Z_t)) applied right after every CZ
// Offline only. No network, no credentials. Exports pure functions; see qc26b_replay.mjs for the driver.
export function angle(a){
  if(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(a))return Number(a);
  const m=a.match(/^(-?)(?:(\d+(?:\.\d+)?)\*)?pi(?:\/(\d+))?$/);
  if(!m)throw Error("angle "+a);
  return (m[1]?-1:1)*Number(m[2]||1)*Math.PI/Number(m[3]||1);
}
export function parse(r,qasm){
  const path=r.physical_path,ops=[],meas={};
  for(let line of qasm.split(/\r?\n/)){
    line=line.trim();
    if(!line||/^(OPENQASM|include |bit\[)/.test(line))continue;
    const mm=line.match(/^meas\[(\d+)\] = measure \$(\d+);$/);
    if(mm){meas[mm[1]]=+mm[2];continue;}
    const m=line.match(/^(\w+)(?:\(([^)]+)\))? (.+);$/);
    if(!m)throw Error("line "+line);
    const p=Array.from(m[3].matchAll(/\$(\d+)/g),x=>+x[1]);
    if(p.some(x=>!path.includes(x)))throw Error("path");
    ops.push({name:m[1],angle:m[2]===undefined?null:angle(m[2]),p,q:p.map(x=>path.indexOf(x))});
  }
  for(let i=0;i<path.length;i++)if(meas[i]!==path[i])throw Error("measurement map");
  return ops;
}
// ASAP schedule with the QC-23 calibrated fault placement. `coherentCz` inserts a marker op after each CZ.
export function schedule(r,ops,snap){
  const clocks=Array(r.n).fill(0),seq=[];
  function rel(q,dur){
    const v=snap.qubits[r.physical_path[q]];
    if(dur<=0||!v.t1||!v.t2)return {p:0,e:[1,1,1,1]};
    const pxy=(1-Math.exp(-dur/v.t1))/4,pz=Math.max(0,(1-Math.exp(-dur/Math.min(v.t2,2*v.t1)))/2-pxy);
    return {p:2*pxy+pz,e:[1,1-2*(pxy+pz),1-2*(pxy+pz),1-4*pxy]};
  }
  const addrel=(q,d)=>{const v=rel(q,d);if(v.p>0)seq.push({name:"relax",q:[q],e:v.e});};
  for(const o of ops){
    const {name,p,q}=o;
    if(name==="barrier"){const t=Math.max(...q.map(i=>clocks[i]));for(const i of q){addrel(i,t-clocks[i]);clocks[i]=t;}continue;}
    if(!["rz","sx","x","cz","id"].includes(name))throw Error("unsupported "+name);
    const t=Math.max(...q.map(i=>clocks[i]));let dur=0,err=0;
    if(name==="sx"||name==="x"){dur=snap.qubits[p[0]][name+"_duration"]||0;err=snap.qubits[p[0]][name+"_error"]||0;}
    if(name==="cz"){
      const e=snap.edges.find(x=>x.qargs[0]===p[0]&&x.qargs[1]===p[1])||snap.edges.find(x=>x.qargs[0]===p[1]&&x.qargs[1]===p[0]);
      if(!e)throw Error("edge");dur=e.duration||0;err=e.error||0;
    }
    for(const i of q){addrel(i,t-clocks[i]);clocks[i]=t+dur;}
    seq.push(o);
    if(name==="cz")seq.push({name:"czerr",p,q});               // coherent error sits inside the gate, before stochastic faults
    if(name==="sx"||name==="x"){
      const v=rel(q[0],dur),dp=Math.min(.75,Math.max(0,1.5*err-v.p));
      if(dp)seq.push({name:"diag",q,e:[1,1-4*dp/3,1-4*dp/3,1-4*dp/3]});
      addrel(q[0],dur);
    }
    if(name==="cz"){
      const a=rel(q[0],dur),b=rel(q[1],dur),dp=Math.min(15/16,Math.max(0,1.25*err-a.p-b.p));
      if(dp)seq.push({name:"diag2",q,e:1-16*dp/15});
      for(const i of q)addrel(i,dur);
    }
  }
  return seq;
}
export const czKey=(a,b)=>a<b?`${a}_${b}`:`${b}_${a}`;
// cfg: {nonunital:bool, sx:{physQubit:rad}, cz:{"a_b":{alpha:{phys:rad}, delta:rad}}}
// Returns {z, discard (sum |discarded coefficients|), ledger:{events, l1, l2sq, peak}}.
export function evaluate(n,seq,target,ro,cfg={},tol=1e-15){
  const nonunital=!!cfg.nonunital,sx=cfg.sx||{},czc=cfg.cz||{};
  let terms=new Map([[(3<<(2*target))>>>0,1-2*ro]]),l1=0,l2sq=0,events=0,peak=1,etaSum=0,stepSq=0;
  for(let j=seq.length-1;j>=0;j--){
    const o=seq[j],a=2*o.q[0],b=o.q.length===2?2*o.q[1]:0,out=new Map();stepSq=0;
    const add=(k,v)=>{if(Math.abs(v)<=tol){l1+=Math.abs(v);l2sq+=v*v;stepSq+=v*v;events++;return;}out.set(k,(out.get(k)||0)+v);};
    const cs=o.name==="rz"?Math.cos(o.angle):0,ss=o.name==="rz"?Math.sin(o.angle):0;
    const off=o.name==="sx"?(sx[o.p[0]]||0):0,xc=off===0?0:-Math.sin(off),xs=off===0?1:Math.cos(off);
    let ce=null;
    if(o.name==="czerr"){const e=czc[czKey(o.p[0],o.p[1])];if(e)ce={da:e.alpha?(e.alpha[o.p[0]]||0):0,db:e.alpha?(e.alpha[o.p[1]]||0):0,d:e.delta||0};}
    if(o.name==="czerr"&&!ce)continue;                          // no coherent error configured on this edge
    for(const [k,v] of terms){
      const x=(k>>>a)&3,y=(k>>>b)&3;
      switch(o.name){
        case "relax":add(k,v*o.e[x]);if(nonunital&&x===3)add((k^(3<<a))>>>0,v*(1-o.e[3]));break;
        case "diag":add(k,v*o.e[x]);break;
        case "diag2":add(k,(x||y)?v*o.e:v);break;
        case "rz":if(x===1||x===2){add(k,v*cs);add((k^(3<<a))>>>0,v*ss*(x===1?-1:1));}else add(k,v);break;
        case "sx":if(x===2||x===3){if(xc!==0)add(k,v*xc);add((k^(1<<a))>>>0,v*xs*(x===2?-1:1));}else add(k,v);break;
        case "x":add(k,(x===2||x===3)?-v:v);break;
        case "cz":{
          const ax=x===1||x===2,ay=y===1||y===2;let xx=x,yy=y,sign=1;
          if(ax&&ay){xx=3-x;yy=3-y;sign=x===y?1:-1;}else if(ax){yy=3-y;}else if(ay){xx=3-x;}
          add((k^((x^xx)<<a)^((y^yy)<<b))>>>0,v*sign);break;
        }
        case "czerr":{
          // U = exp(-i/2 (da Z_a + db Z_b + d Z_a Z_b)); Heisenberg step is U^dag P U. The three generators commute.
          // Expand sequentially: local Z rotations (as rz), then the ZZ rotation.
          let list=[[k,v]];
          const rzStep=(arr,pos,th)=>{
            if(th===0)return arr;const c=Math.cos(th),s=Math.sin(th),res=[];
            for(const [kk,vv] of arr){const p=(kk>>>pos)&3;
              if(p===1||p===2){res.push([kk,vv*c]);res.push([(kk^(3<<pos))>>>0,vv*s*(p===1?-1:1)]);}else res.push([kk,vv]);}
            return res;
          };
          list=rzStep(list,a,ce.da);list=rzStep(list,b,ce.db);
          if(ce.d!==0){
            const c=Math.cos(ce.d),s=Math.sin(ce.d),res=[];
            for(const [kk,vv] of list){
              const pa=(kk>>>a)&3,pb=(kk>>>b)&3,xa=pa===1||pa===2,xb=pb===1||pb===2;
              if(xa===xb){res.push([kk,vv]);continue;}              // commutes with Z_a Z_b
              res.push([kk,vv*c]);
              const flip=(p)=>p===0?3:p===3?0:p===1?2:1;             // multiplication by Z: I<->Z, X<->Y
              const k2=((kk&~(3<<a)&~(3<<b))|(flip(pa)<<a)|(flip(pb)<<b))>>>0;
              const xp=xa?pa:pb;                                      // the X/Y factor: P*ZZ = -sin(d)*[X] or +sin(d)*[Y]
              res.push([k2,vv*s*(xp===1?-1:1)]);
            }
            list=res;
          }
          for(const [kk,vv] of list)add(kk,vv);
          break;
        }
        case "id":add(k,v);break;
        default:throw Error(o.name);
      }
    }
    etaSum+=Math.sqrt(stepSq);terms=out;peak=Math.max(peak,terms.size);if(peak>2000000)throw Error("term budget");
  }
  let z=0;
  for(const [k,v] of terms){let ok=true;for(let q=0;q<n;q++){const p=(k>>>(2*q))&3;if(p===1||p===2){ok=false;break;}}if(ok)z+=v;}
  return {z,discard:l1,ledger:{events,l1,l2sq,peak,old_bound:Math.pow(2,n/2)*etaSum,l1_bound:l1,min_bound:Math.min(Math.pow(2,n/2)*etaSum,l1)}};
}
