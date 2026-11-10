"""QC-21 packed checkpoint preflight with exact global boundary anchors.

Preserves QC-17 genesis/readout checks without decoding every stored binary64
coefficient into persistent Fraction tuples. Checkpoint hashing, shape, index
and tally validation remains the responsibility of the existing independent
QC-20 packed checkpoint parser.
"""
from __future__ import annotations

from qc15_checker import require, initial_core, realhex
from qc20_packed import StreamFastKernel, pack_tensor


def check_genesis(point, plan):
    require(point['event_index']==0 and
            (point['first'],point['last'])==(plan['targets'][0],plan['targets'][-1]),
            'checkpoint genesis support mismatch')
    expected=[pack_tensor(initial_core(ch)) for ch in plan['pauli']]
    observed=point['cores']
    require(len(observed)==len(expected) and
            all(a.shape==b.shape and a.raw==b.raw for a,b in zip(observed,expected)),
            'checkpoint genesis must be exact Pauli observable')
    require(point['accumulated']==0 and all(x==0 for x in point['tally'].values()),
            'checkpoint genesis has nonzero error or counts')


def check_final(point, plan, claimed):
    require(point['event_index']==len(plan['schedule']),
            'wrong final event index')
    require((point['first'],point['last'])==plan['final_support'],
            'final support differs from circuit backward cone')
    value=StreamFastKernel(cache_limit=6).exact_readout(point['cores'])
    reported=realhex(claimed.get('expectation'))
    error=realhex(claimed.get('error_upper'),nonnegative=True)
    require(error>=point['accumulated']+abs(value[0]-reported),
            'end-to-end error bound understated')
    require(abs(value[1])<=error,'imaginary readout exceeds bound')
    require(claimed.get('event_counts')==point['tally'],
            'final event tallies disagree')
    require(sum(point['tally'].values())==sum(s[0]!='grow' for s in plan['schedule']),
            'incomplete audited-event count')
    return {'proof_level':'checkpointed-parallel-full-witness',
            'reported_expectation':float(reported),
            'verified_bound':float(error),
            'arithmetic_lower_requirement':float(point['accumulated']),
            'event_counts':point['tally'],
            'events_checked':sum(point['tally'].values()),
            'cone_width':plan['cone_width'],
            'target':'polar-corrected binary64 gate/Kraus circuit'}
