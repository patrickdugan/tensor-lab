"""PGC-QPU04: assemble an immutable two-round calibration parameter pack.

Run from any directory. Input is committed receipt directories, never evaluation counts.
The historical Ramsey reports predate count-hash binding; their exact archival job IDs
and plan hashes are checked. New Ramsey fits must carry canonical count hashes.
This tool never connects to IBM or creates authorization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PGC=HERE.parent/"pgc_qpu"
OLD=PGC/"receipts/live_20261127"
VERSION="pgc-qpu04-v1"
BLOCKS=("qc26a","q5","q23")
SEEDS={"qc26a":261127041,"q5":261127042,"q23":261127043}
OLD_JOBS={"qc26a":"dc4n9ads2omc738fm84g","q5":"dc4ne0thdvas73bq1hq0","q23":"dc4ngn2nj2vs73dkdrdg"}
SX=(3,5,21,23)
EDGES={"q5":((4,5),(5,6)),"q23":((16,23),(22,23))}

def need(ok,message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def lf(path):
    return path.read_bytes().replace(b"\r\n",b"\n")

def matches_bytes(path,digest):
    raw=lf(path)
    return digest in {sha(raw),sha(raw.replace(b"\n",b"\r\n"))}

def rel(path):
    return path.resolve().relative_to(ROOT).as_posix()

def read(path,sources):
    key=rel(path)
    raw=lf(path)
    sources[key]=sha(raw)
    return json.loads(raw)

def observation(estimate,se,block,**extra):
    need(type(estimate) in (int,float) and math.isfinite(estimate),"nonfinite estimate")
    need(type(se) in (int,float) and math.isfinite(se) and se>0,"invalid standard error")
    return {"estimate":estimate,"se":se,"block":block,**extra}

def read_epoch(directory,historical=False):
    directory=directory.resolve()
    need(directory.is_relative_to(PGC/"receipts"),"calibration receipts must be archived under pgc_qpu/receipts")
    if historical:
        need(directory==OLD,"historical round is pinned")
    sources={}
    proposal=read(HERE/"qc26a_plan.json",sources)
    campaign=read(PGC/"ramsey_campaign.json",sources)
    blocks={}
    obs={}
    for block in BLOCKS:
        base=directory/block
        compiled=read(base/"compiled.json",sources)
        job=read(base/"job.json",sources)
        counts=read(base/"counts.json",sources)
        cal=read(base/"calibration_build.json",sources)
        sub=read(base/"calibration_submission.json",sources)
        need(compiled["backend"]==cal["backend"]==sub["backend"]=="ibm_kingston","wrong backend")
        need(compiled["plan_sha256"]==job["plan_sha256"]==sha(canon({k:v for k,v in compiled.items() if k!="plan_sha256"})),"manifest/job hash mismatch")
        need(matches_bytes(base/"calibration_build.json",compiled["calibration_sha256"]),"calibration bytes mismatch")
        for k in ("timestamp","qubits","edges","basis_gates","two_qubit_gate"):
            need(cal[k]==sub[k],"calibration changed: "+block+" "+k)
        need(cal["two_qubit_gate"]=="cz","non-CZ backend")
        if block=="qc26a":
            specs=proposal["rows"]
            proposal_file=HERE/"qc26a_plan.json"
            report_file=directory/"analysis/identify_qc26a_live.json"
        else:
            specs=next(b["rows"] for b in campaign["blocks"] if b["id"]==block)
            proposal_file=PGC/"ramsey_campaign.json"
            report_file=directory/("analysis/ramsey_"+block+"_identify.json")
        need(matches_bytes(proposal_file,compiled["proposal_sha256"]),"proposal hash mismatch: "+block)
        expected=list(specs)
        if historical:
            need(job["job_id"]==OLD_JOBS[block],"historical job ID mismatch")
        else:
            need(compiled.get("ordering_seed")==SEEDS[block],"unexpected replication ordering seed")
            random.Random(SEEDS[block]).shuffle(expected)
            need([r["id"] for r in expected]!=[r["id"] for r in specs],"replication order unchanged")
            execution=read(base/"execution_receipt.json",sources)
            need(execution["job_id"]==job["job_id"] and execution["plan_sha256"]==job["plan_sha256"],"execution receipt mismatch")
            need(execution["circuits"]==len(expected) and execution["shots_per_circuit"]==384,"execution shape")
            need(execution["calibration_timestamp_build"]==cal["timestamp"]==execution["calibration_timestamp_submission"],"execution calibration mismatch")
        rows=compiled["circuits"]
        need(len(rows)==len(expected) and len({r["id"] for r in rows})==len(rows),"compiled row shape")
        for spec,row in zip(expected,rows):
            need(all(row.get(k)==v for k,v in spec.items()),"row/plan mismatch: "+spec["id"])
        need(set(counts)=={r["id"] for r in rows},"counts ID set mismatch")
        for row in rows:
            c=counts[row["id"]]
            need(all(len(k)==len(row["path"]) and not set(k)-set("01") and type(n) is int and n>=0 for k,n in c.items()),"bad bitstring/count")
            need(sum(c.values())==row["shots"]==384,"shot mismatch")
        count_hash=sha(canon(counts))
        report=read(report_file,sources)
        blocks[block]={"job_id":job["job_id"],"plan_sha256":job["plan_sha256"],"submitted_at":job["submitted_at"],
                       "calibration_timestamp":cal["timestamp"],"ordering_seed":compiled.get("ordering_seed"),
                       "counts_sha256":count_hash,"fit_path":rel(report_file)}
        if block=="qc26a":
            inp=report["inputs"]
            need(inp["counts_sha256"]==count_hash and inp["plan_sha256"]==sha(canon(proposal)),"QC26A fit provenance mismatch")
            need(inp["bootstraps"]>=200,"QC26A bootstrap count below 200")
            for q in SX:
                key=f"sx.{q}.offset_rad"
                obs[f"sx.{q}"]=observation(report["fit_parameters"][key],report["bootstrap"]["parameters"][key]["se"],block,
                                          diagnostic_status=report["decision"]["per_parameter"][key]["status"])
        else:
            fits=[x for x in report["blocks"] if x["block"]==block]
            need(len(fits)==1,"ambiguous Ramsey fit")
            fit=fits[0]
            need(fit["plan_sha256"]==compiled["plan_sha256"],"Ramsey fit plan mismatch")
            inp=fit.get("inputs")
            if not historical or inp is not None:
                need(inp is not None and inp["counts_sha256"]==count_hash and inp["bootstraps"]>=200,"Ramsey fit provenance/bootstrap mismatch")
            blocks[block]["fit_count_hash_bound"]=inp is not None
            for a,b in EDGES[block]:
                edge=f"{a}_{b}"
                f=fit["edges"][edge]
                p=f["point"]; boot=f["bootstrap"]
                meta={"forward_reverse_agree_95":f["forward_reverse_agree_95"],"chi2":p["chi2"]}
                obs[f"cz.{edge}.delta"]=observation(p["delta"],boot["delta"]["se"],block,**meta)
                for q in (a,b):
                    obs[f"cz.{edge}.alpha.{q}"]=observation(p["alpha"][str(q)],boot[f"alpha_{q}"]["se"],block,**meta)
    return {"root":rel(directory),"blocks":blocks,"observations":obs,
            "sources":[{"path":k,"sha256":v} for k,v in sorted(sources.items())]}

def pack_for(fresh):
    old=read_epoch(OLD,True)
    new=read_epoch(fresh)
    return {"version":VERSION,"backend":"ibm_kingston","epochs":[old,new]}

def latest_params(pack):
    obs=pack["epochs"][1]["observations"]
    sx={str(q):obs[f"sx.{q}"]["estimate"] for q in SX}
    cz={}
    for pairs in EDGES.values():
        for a,b in pairs:
            edge=f"{a}_{b}"
            cz[edge]={"delta":obs[f"cz.{edge}.delta"]["estimate"],
                      "alpha":{str(q):obs[f"cz.{edge}.alpha.{q}"]["estimate"] for q in (a,b)}}
    return {"sx":sx,"cz":cz,"meta":{"pgc_qpu04":pack}}

def verify(params):
    saved=json.loads(Path(params).read_text(encoding="utf-8"))
    pack=saved["meta"]["pgc_qpu04"]
    fresh=(ROOT/pack["epochs"][1]["root"]).resolve()
    need(latest_params(pack_for(fresh))==saved,"parameter pack/source files changed")
    return saved

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--fresh",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    v=sub.add_parser("verify")
    v.add_argument("--params",required=True,type=Path)
    a=ap.parse_args()
    if a.command=="prepare":
        params=latest_params(pack_for(a.fresh))
        with a.out.open("x",encoding="utf-8") as f:
            json.dump(params,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
        print("Wrote independent calibration pack",a.out)
    else:
        verify(a.params)
        print("PGC-QPU04 calibration source pack verified")

if __name__=="__main__":
    main()
