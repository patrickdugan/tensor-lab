"""Offline verification of every selected control block and frozen layout."""
import json
from pathlib import Path
import numpy as np
import pytest
from math import pi
from recursive import simulate
from selective import relative_flip,selective_y,core,audit,bound,count
from route_dp import Router,verify,index
from qpu04_design import regenerate,EDGES,REGIONS
from qpu04 import logical_rows
HERE=Path(__file__).resolve().parent

@pytest.mark.parametrize('r',range(1,7))
def test_relative_flip(r):
    n=1<<(r+1);u=simulate(relative_flip(list(range(1,r+1)),0),r+1,np.eye(n));ix=np.arange(n);goal=ix.copy();goal[(ix&((1<<r)-1))==0]^=1<<r
    assert np.max(abs(abs(u)-np.eye(n)[:,goal]))<1e-11

@pytest.mark.parametrize('r',range(1,9))
def test_selective(r):
    n=1<<(r+1);L=1<<r
    for theta in (-pi/2,.382):
        ref=np.eye(n,dtype=complex);ref[0,0]=ref[L,L]=np.cos(theta/2);ref[L,0]=np.sin(theta/2);ref[0,L]=-np.sin(theta/2)
        assert np.max(abs(simulate(selective_y(list(range(1,r+1)),0,theta),r+1,np.eye(n))-ref))<1e-11

@pytest.mark.parametrize('b',[3,4,5,6,7,8])
def test_unitary(b):
    a=core(b,'direct');c=core(b,'commutator');u=simulate(a,b,np.eye(1<<b));v=simulate(c,b,np.eye(1<<b));assert np.max(abs(u-v))<1e-11
    audit(c,b,bound(b))

@pytest.mark.parametrize('region',[0,1])
def test_frozen(region):
    d=json.loads((HERE/f'qpu04_b7_r{region}.json').read_text());z,rows=logical_rows(d)
    assert len(rows)==38 and count(z['A'][1]['gates'])==276 and count(z['D'][1]['gates'])==199
    assert z['A'][1]['initial']==z['D'][1]['initial'] and z['A'][1]['final']==z['D'][1]['final']
    assert set(REGIONS[0]).isdisjoint(REGIONS[1]);assert all(z[k][2]['max_routing_error']<1e-11 for k in z)


def test_three_wire_routing():
    r=Router(3,[(0,1),(1,2)]);rng=np.random.default_rng(4704)
    for _ in range(12):
        g=[]
        for j in range(15):
            a,b=rng.choice(3,2,replace=False);g += [('cx',int(a),int(b)),('rz',int(a),float(rng.normal()))]
        z=r.route(g);assert verify(g,z,3)<1e-11


def test_native_optional():
    pytest.importorskip('qiskit');pytest.importorskip('qiskit_ibm_runtime')
    from qpu04 import build_rows
    d=json.loads((HERE/'qpu04_b7_r0.json').read_text());rows,info=build_rows(d);assert len(rows)==38 and info['D']['core_metrics']['cz']<200
