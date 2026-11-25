"""Offline integration smoke test against QC-23's exact Clifford Pauli oracle."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qc23_analysis import load_set
from qc23_stabilizer import extract_native_ops,pauli_flip_expectations
from qc23_noise import model_a_faults,restrict_snapshot
from qc25_channels import native_channels,mpo_z,dense_z

def main():
    root=Path(__file__).resolve().parents[1]/'results'/'hw_root'
    snap=json.loads((root/'compiled'/'calibration_snapshot.json').read_text())
    _,items=load_set(root,'eval_echo')
    ids=sorted(cid for cid,(rec,_) in items.items() if rec['n']==8)
    assert ids, 'missing committed width-8 echo'
    rec,isa=items[ids[0]]
    ops=extract_native_ops(isa,rec['physical_path'])
    local=restrict_snapshot(snap,rec['physical_path'])
    faults,readout=model_a_faults(ops,local)
    channels,ro=native_channels(ops,local)
    n=rec['n']
    bits=tuple(rec['bits'])
    exact=pauli_flip_expectations(ops,n,faults,readout,bits)
    for b in bits:
        r=mpo_z(n,channels,b,cap=16,readout=ro[b])
        assert abs(r['z']-exact[b])<1e-9,(b,r,exact[b])
    print('QC-25 native Clifford bridge:',ids[0],len(bits),'observables passed')
if __name__=='__main__': main()
