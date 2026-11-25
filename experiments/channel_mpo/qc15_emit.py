"""QC-15 full-witness and conditional-ledger emitters for QC-14.

The full witness captures only the changed MPO cores at each audited operation,
plus the gate/Kraus coefficients and final claimed error. This keeps a single
continuously linked trace without dumping a full many-body state each time.

Use qc15_checker.py in a distinct process to verify the resulting artifact.
The ledger-only export provides compact deep-circuit arithmetic replay but is
explicitly conditional on QC-14's locally supplied residual bounds.
"""
from __future__ import annotations
import argparse
import base64
from fractions import Fraction as F
from pathlib import Path
import numpy as np

from qc15_checker import SCHEMA,LEDGER_SCHEMA,write_proof
from qc14_certified import VerifiedOperatorMPO,certify_observable
from qc13_dyadic import _upper_float
from qc10_cone import backward_cone


def pack_tensor(a):
    a=np.asarray(a)
    if a.dtype not in (np.dtype('float64'),np.dtype('complex128')) or not np.isfinite(a).all():
        raise ValueError('only finite binary64 real/complex arrays')
    b=np.ascontiguousarray(a,dtype='<c16')
    return {'shape':list(b.shape),'binary64_le':base64.b64encode(b.tobytes()).decode('ascii')}


def hx(value):
    return float(value).hex()


class WitnessOperatorMPO(VerifiedOperatorMPO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shadow=[a.copy() for a in self.cores]
        self.trace=[]

    def grow(self,first,last):
        old_first=self.first;old_last=self.last
        super().grow(first,last)
        for _ in range(old_first-first):
            self.shadow.insert(0,np.eye(2,dtype=complex).reshape(1,2,2,1).copy())
        for _ in range(last-old_last):
            self.shadow.append(np.eye(2,dtype=complex).reshape(1,2,2,1).copy())
        if any(not np.array_equal(a,b) for a,b in zip(self.shadow,self.cores)):
            raise ArithmeticError('shadow state diverged while growing')
        self.trace.append({'kind':'grow','layer':self._layer,'first':first,'last':last})

    def _record(self,kind,layer,site,local,excluded,alpha=0.):
        super()._record(kind,layer,site,local,excluded,alpha)
        indices=sorted(excluded)
        if len(self.shadow)!=len(self.cores):
            raise ArithmeticError('shadow state length mismatch')
        for i in range(len(self.cores)):
            if i not in excluded and not np.array_equal(self.shadow[i],self.cores[i]):
                raise ArithmeticError('unlogged tensor changed')
        event=self.events[-1]
        self.trace.append({
            'kind':kind,'layer':layer,'site':site,'indices':indices,
            'new_cores':[pack_tensor(self.cores[i]) for i in indices],
            'residual_upper':hx(event.local_residual_upper),
            'context_upper':hx(event.external_norm_upper),
            'polar_upper':hx(event.polar_correction_upper)
        })
        for i in indices:self.shadow[i]=self.cores[i].copy()


def export_witness(n,layers,operators,targets,label,*,bond_cap=8,
                   max_cone=64,max_entries=1_000_000,max_events=4000):
    targets=tuple(targets)
    cone=backward_cone(n,layers,targets)
    if len(cone.sites)>max_cone:raise MemoryError('cone cap')
    ks=tuple(np.asarray(k) for k in operators)
    chain=WitnessOperatorMPO(targets,label,bond_cap,operators=ks,max_events=max_events)
    if chain.max_entries>max_entries:raise MemoryError('initial entries exceed cap')
    for t in range(len(layers)-1,-1,-1):
        chain._layer=t
        active_after=cone.active_before[t+1]
        for site in range(chain.first,chain.last+1):
            chain.channel_adjoint(site,ks)
        active_before=cone.active_before[t]
        if tuple(range(active_before[0],active_before[-1]+1))!=active_before:
            raise ValueError('noncontiguous backward cone')
        chain.grow(active_before[0],active_before[-1])
        if chain.max_entries>max_entries:raise MemoryError('retained MPO cap')
        for site,u in layers[t]:
            if site in active_after or site+1 in active_after:
                chain.gate_adjoint(site,np.asarray(u),t)
                if chain.max_entries>max_entries:raise MemoryError('retained MPO cap')
    expectation=chain.input_zero_expectation()
    rounding=chain.readout_cert.maximum_real_error(expectation)
    total=_upper_float(F(chain.e)+F(rounding))
    kinds=('channel','left_qr','right_qr','gate_svd')
    obj={
        'format':SCHEMA,
        'circuit':{'qubits':n,'targets':list(targets),'pauli':label,
                   'layers':[[{'site':int(j),'gate':pack_tensor(u)} for j,u in layer] for layer in layers],
                   'kraus':[pack_tensor(k) for k in ks]},
        'events':chain.trace,
        'result':{'expectation':hx(expectation),'error_upper':hx(total),
                  'cone_width':len(cone.sites),
                  'event_counts':{k:sum(e.kind==k for e in chain.events) for k in kinds}},
    }
    return obj


def export_ledger(n,layers,operators,targets,label,*,bond_cap=8):
    r=certify_observable(n,layers,operators,targets,label,bond_cap=bond_cap)
    return {'format':LEDGER_SCHEMA,'events':[
        {'kind':e.kind,'residual_upper':hx(e.local_residual_upper),
         'context_upper':hx(e.external_norm_upper),
         'polar_upper':hx(e.polar_correction_upper)} for e in r.events],
         'result':{'event_count':len(r.events),'expectation':hx(r.reported_expectation),
                   'error_upper':hx(r.reported_error_upper),
                   'readout_real_numerator_hex':hex(r.exact_readout.exact_real_numerator),
                   'readout_denominator_exp':r.exact_readout.denominator_exp}}


def main():
    from channel_mpo import gate_layers,kraus
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qubits',type=int,default=3)
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--noise',choices=('dephasing','depolarizing','amplitude_damping'),default='dephasing')
    p.add_argument('--strength',type=float,default=.5)
    p.add_argument('--bond-cap',type=int,default=4)
    p.add_argument('--target',type=int,default=0)
    p.add_argument('--tier',choices=('witness','ledger'),default='witness')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    layers=gate_layers(a.qubits,a.depth,a.seed)
    ks=kraus(a.noise,a.strength)
    args=(a.qubits,layers,ks,(a.target,),'Z')
    if a.tier=='witness':
        proof=export_witness(*args,bond_cap=a.bond_cap)
    else:
        proof=export_ledger(*args,bond_cap=a.bond_cap)
    write_proof(proof,a.output)
    print(a.output,a.output.stat().st_size,'bytes',len(proof['events']),'entries')

if __name__=='__main__':
    main()
