"""Radial QFT for Z/(p^n), restricted to states invariant under every unit.
The normalized valuation shells have dimension n+1. This is an exact
mathematical restriction; the emitted rotation angles are binary64 values.
"""
from math import atan2, sqrt
import numpy as np
from circuit import mux_ry
from orbit_qft import prime


def radial_matrix(n,p=2):
    assert n>=1 and prime(p)
    w = np.array([1.] + [(p-1)*float(p)**(j-1) for j in range(1,n+1)])
    a = np.sqrt(np.outer(w,w)/float(p)**n)
    i,j = np.indices((n+1,n+1))
    a[(i>0)&(j>0)&(i+j==n+1)] *= -1/(p-1)
    a[(i>0)&(j>0)&(i+j>n+1)] = 0
    return a


def radial_product(n,p=2):
    """Q R = G_0 ... G_{n-1}, cos(theta)=sqrt((p-1)/p), sin(theta)=1/sqrt(p)."""
    assert n>=1 and prime(p)
    a=np.eye(n+1)
    c,d=sqrt((p-1)/p),1/sqrt(p)
    for j in range(n):
        a[:,j:j+2]=a[:,j:j+2]@np.array([[c,d],[-d,c]])
    return a[:,::-1]


def gray(x):
    return x^(x>>1)


def compile_radial(n,p=2):
    """Use n+1=2^b shells; Gray encoding makes reversal a single X.
    Each adjacent shell rotation is a uniformly controlled RY with just one
    nonzero branch angle. No ancillas, all-to-all logical connections.
    """
    b=(n+1).bit_length()-1
    assert (1<<b)==n+1 and b>=1 and prime(p)
    gates=[('x',0)]
    theta=2*atan2(1,sqrt(p-1))
    for j in reversed(range(n)):
        x,y=gray(j),gray(j+1);bit=(x^y).bit_length()-1
        assert x^y==1<<bit
        t=b-1-bit
        controls=[i for i in range(b) if i!=t]
        pattern=0
        for c in controls:
            pattern=2*pattern+((x>>(b-1-c))&1)
        angles=[0.]*(1<<(b-1))
        angles[pattern]=theta if x&(1<<bit) else -theta
        gates+=mux_ry(angles,controls,t)
    return gates
