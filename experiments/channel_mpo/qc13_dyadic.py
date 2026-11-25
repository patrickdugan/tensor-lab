"""QC-13: exact dyadic audit of stored SVD factors and MPO readout.

This module uses Python arbitrary-precision integers to evaluate operations on
*the exact binary64 values supplied to it*. It never assumes SVD or BLAS is
backward-stable. The output residuals are rigorous for the locally supplied
floating-point matrix and the *stored* floating-point factors, including the
rounded product s * vh; they do not yet certify an entire QC-12 circuit.

There is no dependency on an interval library or a directed-rounding FPU mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, isqrt, nextafter, inf
from typing import Sequence
import numpy as np


def _parts(z:complex | float) -> tuple[float,float]:
    # Accept numpy scalar numeric types and fail closed on invalid inputs.
    w = complex(z)
    if not (isfinite(w.real) and isfinite(w.imag)):
        raise ValueError('nonfinite binary floating-point input')
    return w.real,w.imag


def _exponent(x:float) -> int:
    n,d=x.as_integer_ratio()
    return d.bit_length()-1


def _scale(x:float,p:int) -> int:
    n,d=x.as_integer_ratio()
    return n << (p-(d.bit_length()-1))


def _int_complex_array(a:np.ndarray,p:int) -> list:
    """Convert exact binary64 values to integer pairs scaled by 2**p."""
    return [[(_scale(re,p),_scale(im,p)) for re,im in map(_parts,row)]
            for row in np.asarray(a)]


def _upper_float(q:Fraction) -> float:
    if q==0:return 0.
    f=float(q)
    if not isfinite(f):raise OverflowError('bound outside finite binary64 range')
    if Fraction(f)<q:
        f=nextafter(f,inf)
    return f


def _lower_float(q:Fraction) -> float:
    f=float(q)
    if not isfinite(f):raise OverflowError('bound outside finite binary64 range')
    if Fraction(f)>q:
        f=nextafter(f,-inf)
    return f


@dataclass(frozen=True)
class DyadicResidual:
    upper: float
    squared_numerator: int
    squared_denominator_exp: int
    entries: int
    inner_dimension: int
    def squared_exact(self)->Fraction:
        return Fraction(self.squared_numerator, 1<<self.squared_denominator_exp)


def residual_certificate(m:np.ndarray,left:np.ndarray,right:np.ndarray) -> DyadicResidual:
    """Rigorous ||M-LR||_F upper for the exact values of binary64 input arrays.

    All complex products and sums are evaluated using Python integers. The
    final sqrt is rounded upward via isqrt, then the binary64 result is rounded
    upward once more. In particular, it covers NumPy's multiplication roundoff
    when `right` is the stored rounded s*vh rather than an abstract SVD factor.
    """
    m=np.asarray(m);left=np.asarray(left);right=np.asarray(right)
    if any(a.dtype not in (np.dtype('float64'),np.dtype('complex128'))
           for a in (m,left,right)):
        raise TypeError('certified residual requires binary64 real/complex arrays')
    if m.ndim!=2 or left.ndim!=2 or right.ndim!=2:
        raise ValueError('expected three matrices')
    rows,cols=m.shape; k=left.shape[1]
    if not k or left.shape[0]!=rows or right.shape!=(k,cols):
        raise ValueError('incompatible factor dimensions')
    values=np.concatenate([m.reshape(-1),left.reshape(-1),right.reshape(-1)])
    p=max(_exponent(v) for z in values for v in _parts(z))
    mi=_int_complex_array(m,p);li=_int_complex_array(left,p);ri=_int_complex_array(right,p)
    two_p=1<<p
    sq=0
    for i in range(rows):
        for j in range(cols):
            er=mi[i][j][0]*two_p
            ei=mi[i][j][1]*two_p
            for v in range(k):
                a,b=li[i][v];c,d=ri[v][j]
                er-=a*c-b*d
                ei-=a*d+b*c
            sq+=er*er+ei*ei
    root=isqrt(sq)
    if root*root<sq:root+=1
    upper=_upper_float(Fraction(root,1<<(2*p)))
    return DyadicResidual(upper,sq,4*p,rows*cols,k)


def svd_stored_certificate(matrix:np.ndarray,u:np.ndarray,s:np.ndarray,
                           vh:np.ndarray,keep:int)->DyadicResidual:
    """Upper for exact matrix minus reconstructed *stored* MPO factors."""
    if not 1<=keep<=len(s):raise ValueError('invalid kept rank')
    # Match the array assigned to QC-12's right MPO core.
    right=s[:keep,None]*vh[:keep]
    return residual_certificate(matrix,u[:,:keep],right)


@dataclass(frozen=True)
class DyadicReadout:
    real_lo:float
    real_hi:float
    imaginary_lo:float
    imaginary_hi:float
    exact_real_numerator:int
    exact_imag_numerator:int
    denominator_exp:int
    def maximum_real_error(self,computed:float)->float:
        if not isfinite(computed):raise ValueError('nonfinite readout')
        # Bounds from outward-rounded exact dyadic enclosure; final absolute
        # differences are also rounded outwards (via Fraction).
        value=Fraction(computed)
        truth=Fraction(self.exact_real_numerator,1<<self.denominator_exp)
        return _upper_float(abs(truth-value))


def mpo_zero_readout_certificate(cores:Sequence[np.ndarray])->DyadicReadout:
    """Rigorous enclosure of <0^n|MPO|0^n> for stored binary64 MPO cores.

    Contracted as a boundary vector with exact integer arithmetic, so the
    enclosure is independent of NumPy einsum and its summation order.
    """
    cores=list(cores)
    if not cores:raise ValueError('empty MPO')
    cuts=[]
    p=0
    right=1
    for a in cores:
        a=np.asarray(a)
        if a.dtype not in (np.dtype('float64'),np.dtype('complex128')):
            raise TypeError('certified MPO readout requires binary64 arrays')
        if a.ndim!=4 or a.shape[1:3]!=(2,2) or a.shape[0]!=right:
            raise ValueError('invalid MPO bond/physical shape')
        b=a[:,0,0,:]
        p=max(p,*( _exponent(v) for z in b.flat for v in _parts(z)))
        cuts.append(b)
        right=a.shape[-1]
    if right!=1 or cuts[0].shape[0]!=1:raise ValueError('unclosed MPO')
    vector=[(1,0)]
    for b in cuts:
        new=[]
        for j in range(b.shape[1]):
            er=ei=0
            for i in range(b.shape[0]):
                a,c=vector[i]
                re,im=_parts(b[i,j])
                br,bi=_scale(re,p),_scale(im,p)
                er+=a*br-c*bi
                ei+=a*bi+c*br
            new.append((er,ei))
        vector=new
    real,imag=vector[0]
    denom=1<<(p*len(cores))
    return DyadicReadout(_lower_float(Fraction(real,denom)),
                         _upper_float(Fraction(real,denom)),
                         _lower_float(Fraction(imag,denom)),
                         _upper_float(Fraction(imag,denom)),
                         real,imag,p*len(cores))


def kraus_completeness_certificate(operators:Sequence[np.ndarray])->float:
    """Exact-dyadic Frobenius upper for I - sum K^dag K.

    This is a rigorous defect of the literal binary64 Kraus coefficients,
    which can differ from exactly trace-preserving abstract CPTP channels.
    """
    ops=[np.asarray(a) for a in operators]
    if not ops or any(a.shape!=(2,2) or
                      a.dtype not in (np.dtype('float64'),np.dtype('complex128')) for a in ops):
        raise ValueError('finite 2x2 Kraus matrices required')
    p=max(_exponent(v) for a in ops for z in a.flat for v in _parts(z))
    ar=[_int_complex_array(a,p) for a in ops]
    sq=0
    for i in range(2):
        for j in range(2):
            re=(int(i==j)<<(2*p));im=0
            for a in ar:
                for k in range(2):
                    x,y=a[k][i];z,w=a[k][j]
                    re-=x*z+y*w
                    im-=x*w-y*z
            sq+=re*re+im*im
    root=isqrt(sq)
    if root*root<sq:root+=1
    return _upper_float(Fraction(root,1<<(2*p)))
