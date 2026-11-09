"""QC-08: one-canonical-sweep spectra and low-workspace joint compression.

This experiment keeps QC-07's fixed-layer entry budgets and its greedy
rank heuristic.  A single canonical scan collects spatial and local-environment
spectra.  Its environment scores use Hermitian Gram matrices; near numerical
rank deficiencies are covered by a direct-SVD fallback.  The default fused
compression visits each tensor once, truncating environment then outgoing
bond.  This changes truncation order, so fidelity must be measured separately.

The alternate 'classic' compression reproduces QC-07's order with faster
scoring.  All compression maintains a local purification and hence positivity.
Physical trace-distance claims require the independent small-n oracle.
Neither the explicit-workspace estimate nor retained-entry accounting bounds
NumPy's hidden LAPACK allocations or process RSS.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from purified_mpo import Truncation, select_rank, stable_svd
from qc07_joint import JointPurifiedChain, JointPlan, allocate_joint, one_site_densities, pauli_transfer_norm


def _gram_singular_values(m: np.ndarray, *, stable: bool = True) -> np.ndarray:
    """Singular values from smaller Hermitian Gram matrix, descending.

    For a near-zero eigenvalue the squaring loses relative precision, so
    a direct SVD is used if Gram suggests conditioning beyond 1e8 or an
    appreciably negative eigenvalue. This is a *scoring* operation only.
    """
    if m.size == 0 or m.ndim != 2 or not np.isfinite(m).all():
        raise ValueError('finite nonempty matrix required')
    gram = m.conj().T @ m if m.shape[1] <= m.shape[0] else m @ m.conj().T
    gram = (gram + gram.conj().T) * .5
    eig = np.linalg.eigvalsh(gram)[::-1]
    if stable and (eig[-1] <= 1.e-16 * max(eig[0],1.e-100)
                   or eig[-1] < -1e-10 * max(eig[0],1.e-100)):
        return np.linalg.svd(m,compute_uv=False)
    return np.sqrt(np.maximum(eig,0.))


@dataclass(frozen=True)
class ScanStats:
    right_canonical_sweeps: int
    forward_qrs: int
    bond_score_svds: int
    environment_gram_scores: int
    environment_score_svd_fallbacks: int
    seconds: float


def scan_joint_spectra(chain: JointPurifiedChain, *, gram: bool=True):
    """Extract both classes of rank spectrum in one left-to-right sweep.

    At each site, the prefix is left-orthonormal and the suffix right-
    orthonormal. This allows independent physical-bond and environment Schmidt
    spectra from the same center tensor before propagating its QR R factor.
    The operation changes only purification gauge, not physical density.
    """
    start=perf_counter()
    chain._right_qr(1)
    bond=[]; env=[]
    fallback=0
    for i in range(chain.n):
        a=chain.cores[i]
        l,p,e,r=a.shape
        if p!=2: raise ValueError('physical qubit dimension required')
        env_mat=a.transpose(0,1,3,2).reshape(l*p*r,e)
        if gram:
            gram_s=_gram_singular_values(env_mat,stable=False)
            # Optional robust fallback: only for numerically meaningful near-zero
            # spectrum where it might change the rank selection.
            if gram_s.size>1 and gram_s[-1] < gram_s[0]*1e-6:
                env_s=np.linalg.svd(env_mat,compute_uv=False)
                fallback+=1
            else:
                env_s=gram_s
        else:
            env_s=np.linalg.svd(env_mat,compute_uv=False)
            fallback+=1
        env.append(env_s)
        if i+1<chain.n:
            m=a.reshape(l*p*e,r)
            q,rr=np.linalg.qr(m,mode='reduced')
            bond.append(np.linalg.svd(rr,compute_uv=False))
            chain.cores[i]=q.reshape(l,p,e,q.shape[1])
            chain.cores[i+1]=np.tensordot(rr,chain.cores[i+1],axes=(1,0))
    return bond,env,ScanStats(1,chain.n-1,chain.n-1,
                               chain.n if gram else 0,fallback,perf_counter()-start)


@dataclass(frozen=True)
class StreamResult:
    plan: JointPlan
    retained_entries: int
    modeled_work_peak_entries: int
    discarded_squared_weight: float
    error_bound_increment: float
    min_local_retention: float
    compression_svds: int
    compression_right_qrs: int
    scan: ScanStats
    seconds: float
    marginal_changes: tuple[float,...]


class StreamingJointChain(JointPurifiedChain):
    """Planned QC-07 ranks, with a one-pass (env then bond) execution option."""
    def compress_layer_stream(self, budget_entries:int,
                              operators:tuple[np.ndarray,...],
                              *, bond_cap:int|None=None,env_cap:int|None=None,
                              bond_weight_override:float|None=None,
                              order:str='fused',gram:bool=True,
                              track_marginals:bool=False)->StreamResult:
        if order not in ('fused','classic'):
            raise ValueError('compression order must be fused or classic')
        if bond_cap is not None and bond_cap<1 or env_cap is not None and env_cap<1:
            raise ValueError('rank caps must be positive')
        if not isinstance(budget_entries,(int,np.integer)) or budget_entries<2*self.n:
            raise ValueError('entry budget infeasible')
        start=perf_counter()
        self._preflight_sweeps()
        bond_s,env_s,stats=scan_joint_spectra(self,gram=gram)
        weight=pauli_transfer_norm(operators) if bond_weight_override is None else float(bond_weight_override)
        plan=allocate_joint(bond_s,env_s,int(budget_entries),bond_cap or self.max_bond,
                            env_cap or self.max_env,channel_bond_weight=weight)
        before=one_site_densities(self) if track_marginals else None
        loss_sum=0.; error_prior=self.error_bound_sum; retention=1.; svds=0
        self._right_qr(1)
        sweep_count=1

        def env_compress(i:int,target:int)->None:
            nonlocal loss_sum,retention,svds
            a=self.cores[i]
            l,p,e,r=a.shape
            mat=a.transpose(0,1,3,2).reshape(l*p*r,e)
            self._admit(3*mat.size+(mat.shape[0]+mat.shape[1])*min(mat.shape))
            u,s,_=stable_svd(mat)
            svds+=1
            rank=min(int(target),select_rank(s,None,self.local_tol))
            dropped=float(np.sum(s[rank:]**2));full=float(np.sum(s**2))
            if full<=0: raise ArithmeticError('zero norm under compression')
            self.error_bound_sum+=float(np.sqrt(min(1.,dropped/full)))
            loss_sum+=dropped;retention=min(retention,1.-dropped/full)
            self.truncations.append(Truncation('environment',i,rank,len(s),dropped))
            self.cores[i]=(u[:,:rank]*s[:rank]).reshape(l,p,r,rank).transpose(0,1,3,2)

        def bond_compress(i:int,target:int)->None:
            nonlocal loss_sum,retention,svds
            a=self.cores[i]
            l,p,e,r=a.shape
            mat=a.reshape(l*p*e,r)
            self._admit(3*mat.size+(mat.shape[0]+mat.shape[1])*min(mat.shape))
            u,s,vh=stable_svd(mat)
            svds+=1
            rank=min(int(target),select_rank(s,None,self.local_tol))
            dropped=float(np.sum(s[rank:]**2));full=float(np.sum(s**2))
            if full<=0: raise ArithmeticError('zero norm under compression')
            self.error_bound_sum+=float(np.sqrt(min(1.,dropped/full)))
            loss_sum+=dropped;retention=min(retention,1.-dropped/full)
            self.truncations.append(Truncation('bond',i,rank,len(s),dropped))
            self.cores[i]=u[:,:rank].reshape(l,p,e,rank)
            self.cores[i+1]=np.tensordot(s[:rank,None]*vh[:rank,:],self.cores[i+1],axes=(1,0))

        if order=='fused':
            for i in range(self.n):
                env_compress(i,plan.environment_ranks[i])
                if i+1<self.n:bond_compress(i,plan.bond_ranks[i])
        else:
            for i,target in enumerate(plan.bond_ranks):
                bond_compress(i,target)
            self._right_qr(1)
            sweep_count+=1
            for i,target in enumerate(plan.environment_ranks):
                env_compress(i,target)
                if i+1<self.n:
                    a=self.cores[i];l,p,e,r=a.shape
                    q,rr=np.linalg.qr(a.reshape(l*p*e,r),mode='reduced')
                    self.cores[i]=q.reshape(l,p,e,q.shape[1])
                    self.cores[i+1]=np.tensordot(rr,self.cores[i+1],axes=(1,0))
        retained=self.dimensions()['purified_entries']
        if retained>budget_entries:raise AssertionError(f'{retained} > {budget_entries} retained entries')
        changes=()
        if before is not None:
            after=one_site_densities(self)
            changes=tuple(float(np.linalg.norm(x-y,ord='nuc')/2) for x,y in zip(before,after))
        return StreamResult(plan,retained,self.predicted_working_peak_entries,
                            loss_sum,self.error_bound_sum-error_prior,retention,
                            svds,sweep_count,stats,perf_counter()-start,changes)

    def gate_layer(self, layer) -> int:
        """Apply a sorted, non-overlapping gate layer with one canonical sweep.

        Compared with calling gate() on each pair, the right-orthogonal suffix
        is prepared once and the left orthogonal gauge is advanced monotonically.
        Post-SVD truncation error uses the center's pretruncation Schmidt norm.
        """
        pairs=list(layer)
        if not pairs:
            return 0
        last=-2
        for i,u in pairs:
            if not isinstance(i,int) or i<=last+1 or not 0<=i<self.n-1 or u.shape!=(4,4):
                raise ValueError('sorted, disjoint nearest-neighbor pairs required')
            last=i
        self._right_qr(1)
        cursor=0
        for i,u in pairs:
            # Carry the left-orthonormal prefix to the next gate.
            for j in range(cursor,i):
                a=self.cores[j]
                l,p,e,r=a.shape
                q,rr=np.linalg.qr(a.reshape(l*p*e,r),mode='reduced')
                self.cores[j]=q.reshape(l,p,e,q.shape[1])
                self.cores[j+1]=np.tensordot(rr,self.cores[j+1],axes=(1,0))
            a,b=self.cores[i],self.cores[i+1]
            l,_,ea,_=a.shape
            _,_,eb,r=b.shape
            rows=l*2*ea;cols=2*eb*r;mrank=min(rows,cols)
            self._admit(3*rows*cols+(rows+cols+1)*mrank)
            theta=np.tensordot(a,b,axes=(3,0))
            theta=np.einsum('xyuv,luavbr->lxaybr',u.reshape(2,2,2,2),theta,optimize=True)
            uu,s,vh=stable_svd(theta.reshape(rows,cols))
            rank=select_rank(s,self.max_bond,self.local_tol)
            lost=float(np.sum(s[rank:]**2)); total=float(np.sum(s**2))
            self.truncations.append(Truncation('bond',i,rank,len(s),lost))
            self.cores[i]=uu[:,:rank].reshape(l,2,ea,rank)
            self.cores[i+1]=(s[:rank,None]*vh[:rank,:]).reshape(rank,2,eb,r)
            if lost>0:
                self.error_bound_sum+=float(np.sqrt(min(1.,lost/total)))
            cursor=i+1
        return len(pairs)
