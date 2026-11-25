"""QC-16: exact dyadic integer kernels for the independent QC-15 checker.

All binary64 inputs are *exact* rationals with powers-of-two denominators.
Each tensor is converted once to a common fixed denominator (2**exp), and
matrix contractions accumulate integer real/imaginary pairs. Fractions are
constructed only at the boundary of each squared norm or final readout.

No NumPy, floating-point arithmetic, or simulator imports. A bounded cache
keeps converted cores for recurring context checks without retaining the full
verification transcript. The reference checker remains available as an
independent arithmetic implementation for differential testing.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction as F
from qc15_checker import sqrt_up, require

I0=(0,0)
I1=(1,0)


def mul(a,b):
    return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]


def plus(a,b):
    return a[0]+b[0], a[1]+b[1]


def conj(a):
    return a[0],-a[1]


def sqnorm(a):
    return a[0]*a[0]+a[1]*a[1]


def shift(a,e):
    return a[0]<<e,a[1]<<e


@dataclass(frozen=True)
class DyadTensor:
    shape:tuple[int,...]
    exponent:int
    vals:tuple[tuple[int,int],...]

    def get4(self,l,a,b,r):
        return self.vals[((l*2+a)*2+b)*self.shape[3]+r]

    def get2(self,i,j):
        return self.vals[i*self.shape[1]+j]


class Kernel:
    def __init__(self,cache_limit=96):
        self.cache=OrderedDict()
        self.cache_limit=cache_limit

    def tensor(self,t):
        key=id(t)
        found=self.cache.get(key)
        if found is not None and found[0] is t:
            self.cache.move_to_end(key)
            return found[1]
        exponent=max((q.denominator.bit_length()-1
                     for pair in t.vals for q in pair),default=0)
        entries=tuple((a.numerator<<(exponent-(a.denominator.bit_length()-1)),
                       b.numerator<<(exponent-(b.denominator.bit_length()-1)))
                      for a,b in t.vals)
        new=DyadTensor(t.shape,exponent,entries)
        self.cache[key]=(t,new)
        if len(self.cache)>self.cache_limit:
            self.cache.popitem(last=False)
        return new

    def _pair(self,a,b):
        require(a.shape[-1]==b.shape[0],"pair bond mismatch")
        dl,dr,inner=a.shape[0],b.shape[-1],a.shape[-1]
        out=[]
        for l in range(dl):
            for r in range(dr):
                op=[]
                for x in range(4):
                    ax,bx=divmod(x,2)
                    row=[]
                    for y in range(4):
                        ay,by=divmod(y,2)
                        sr=si=0
                        for k in range(inner):
                            p=mul(a.get4(l,ax,ay,k),b.get4(k,bx,by,r))
                            sr+=p[0];si+=p[1]
                        row.append((sr,si))
                    op.append(row)
                out.append(op)
        return out

    def pair_error2(self,a,b,c,d,gate=None):
        require(a.shape[0]==c.shape[0] and b.shape[-1]==d.shape[-1],
                "pair outer bonds changed")
        a,b,c,d=[self.tensor(t) for t in (a,b,c,d)]
        left=self._pair(a,b)
        right=self._pair(c,d)
        e_left=a.exponent+b.exponent
        e_right=c.exponent+d.exponent
        if gate is not None:
            g=self.tensor(gate)
            require(g.shape==(4,4),"invalid gate matrix")
            transformed=[]
            for original in left:
                matrix=[]
                for i in range(4):
                    row=[]
                    for j in range(4):
                        sr=si=0
                        for x in range(4):
                            for y in range(4):
                                v=mul(mul(conj(g.get2(x,i)),original[x][y]),g.get2(y,j))
                                sr+=v[0];si+=v[1]
                        row.append((sr,si))
                    matrix.append(row)
                transformed.append(matrix)
            left=transformed
            e_left+=2*g.exponent
        exp=max(e_left,e_right)
        acc=0
        for old, new in zip(left,right):
            for i in range(4):
                for j in range(4):
                    v=shift(old[i][j],exp-e_left)
                    w=shift(new[i][j],exp-e_right)
                    acc+=(v[0]-w[0])**2+(v[1]-w[1])**2
        return F(acc,1<<(2*exp))

    def channel_error2(self,a,b,operators):
        require(a.shape==b.shape,"channel changed bond shape")
        a,b=[self.tensor(t) for t in (a,b)]
        ks=[self.tensor(k) for k in operators]
        require(ks and all(k.shape==(2,2) for k in ks),"bad Kraus list")
        common=max(k.exponent for k in ks)
        kk=[(k,common-k.exponent) for k in ks]
        exp_before=a.exponent+2*common
        exp=max(exp_before,b.exponent)
        acc=0
        for l in range(a.shape[0]):
            for r in range(a.shape[-1]):
                for i in range(2):
                    for j in range(2):
                        sr=si=0
                        for k, delta in kk:
                            for x in range(2):
                                for y in range(2):
                                    v=mul(mul(conj(k.get2(x,i)),a.get4(l,x,y,r)),k.get2(y,j))
                                    v=shift(v,2*delta)
                                    sr+=v[0];si+=v[1]
                        old=shift((sr,si),exp-exp_before)
                        new=shift(b.get4(l,i,j,r),exp-b.exponent)
                        acc+=(old[0]-new[0])**2+(old[1]-new[1])**2
        return F(acc,1<<(2*exp))

    def gram_defect2(self,t,orientation):
        t=self.tensor(t)
        require(len(t.shape)==4,"Gram needs an MPO core")
        dl,dr=t.shape[0],t.shape[-1]
        require(orientation in ("L","R"),"bad isometry orientation")
        dim=dr if orientation=="L" else dl
        acc=0
        for i in range(dim):
            for j in range(dim):
                sr=si=0
                if orientation=="L":
                    for l in range(dl):
                        for ao in range(2):
                            for ai in range(2):
                                v=mul(conj(t.get4(l,ao,ai,i)),t.get4(l,ao,ai,j))
                                sr+=v[0];si+=v[1]
                else:
                    for r in range(dr):
                        for ao in range(2):
                            for ai in range(2):
                                v=mul(t.get4(i,ao,ai,r),conj(t.get4(j,ao,ai,r)))
                                sr+=v[0];si+=v[1]
                if i==j:sr-=1<<(2*t.exponent)
                acc+=sr*sr+si*si
        return F(acc,1<<(4*t.exponent))

    def _stacked_defect2(self,mats,dim):
        mats=[self.tensor(t) for t in mats]
        require(mats and all(t.shape==(dim,dim) for t in mats),"bad matrix list")
        exp=max(t.exponent for t in mats)
        acc=0
        for i in range(dim):
            for j in range(dim):
                sr=si=0
                for k in mats:
                    for x in range(dim):
                        v=mul(conj(k.get2(x,i)),k.get2(x,j))
                        v=shift(v,2*(exp-k.exponent))
                        sr+=v[0];si+=v[1]
                if i==j:sr-=1<<(2*exp)
                acc+=sr*sr+si*si
        return F(acc,1<<(4*exp))

    def gate_defect2(self,t):
        return self._stacked_defect2((t,),4)

    def kraus_defect2(self,ks):
        return self._stacked_defect2(ks,2)

    def frob_norm2(self,t):
        x=self.tensor(t)
        return F(sum(sqnorm(v) for v in x.vals),1<<(2*x.exponent))

    def context_bound(self,cores,excluded,*,gate=False):
        lo=min(excluded);hi=max(excluded)
        product=F(1)
        for idx, core in enumerate(cores):
            if idx in excluded:continue
            if gate and idx<lo:
                product*=1+sqrt_up(self.gram_defect2(core,"L"))
            elif gate and idx>hi:
                product*=1+sqrt_up(self.gram_defect2(core,"R"))
            else:
                product*=sqrt_up(self.frob_norm2(core))
        return product

    def exact_readout(self,cores):
        vector=[I1]
        exponent=0
        for core in cores:
            x=self.tensor(core)
            out=[]
            for j in range(x.shape[-1]):
                sr=si=0
                for i in range(x.shape[0]):
                    v=mul(vector[i],x.get4(i,0,0,j))
                    sr+=v[0];si+=v[1]
                out.append((sr,si))
            vector=out
            exponent+=x.exponent
        require(len(vector)==1,"unclosed readout")
        return F(vector[0][0],1<<exponent),F(vector[0][1],1<<exponent)
