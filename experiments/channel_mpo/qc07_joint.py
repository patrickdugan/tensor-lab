"""QC-07: jointly allocate spatial bonds and local Kraus ranks.

A finite-memory, locally purified, channel-sensitive heuristic.  The layer-
boundary entry budget is exact.  ``work_cap_entries`` is an admission limit on a
conservative explicit-array workload model; it is *not* a guarantee on NumPy's
hidden LAPACK allocations, process RSS, or OS virtual memory.  Compare RSS in
an isolated process before making actual peak-memory claims.

Rank selection uses singular spectra of the current purification.  Its local
importance proxy is squared Schmidt weight; the single-qubit channel's
traceless Pauli transfer norm downweights cross-cut bond improvements after
strongly contracting noise.  These are surrogates for physical trace error,
not optimized global trace distance.  No dense oracle is used by the policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from purified_mpo import PurifiedChain, Truncation, select_rank, stable_svd
from qc06_adaptive import scan_ancilla_spectra
from channel_mpo import X, Y, Z


def pauli_transfer_norm(operators: tuple[np.ndarray, ...]) -> float:
    """Largest singular value of the traceless Pauli transfer 3x3 block.

    E is assumed CPTP.  For amplitude damping this uses the linear Bloch
    block (translation is immaterial to differences of density operators).
    """
    if not operators or any(k.shape != (2, 2) for k in operators):
        raise ValueError('qubit Kraus operators required')
    ident = sum((k.conj().T @ k for k in operators), np.zeros((2, 2), complex))
    if not np.allclose(ident, np.eye(2), atol=1e-10):
        raise ValueError('the channel must be trace-preserving')
    p = (X, Y, Z)
    mat = np.array([[0.5 * sum(np.trace(pi @ k @ pj @ k.conj().T)
                                    for k in operators).real
                     for pj in p] for pi in p], dtype=float)
    return float(np.linalg.svd(mat, compute_uv=False)[0])


def spatial_spectra(chain: PurifiedChain) -> list[np.ndarray]:
    """Schmidt spectra for every purification bond, using only QR gauges.

    This transforms core storage but preserves the purified physical state.
    """
    chain._right_qr(1)
    out = []
    for i in range(chain.n - 1):
        a = chain.cores[i]
        l, d, e, r = a.shape
        m = a.reshape(l * d * e, r)
        out.append(np.linalg.svd(m, full_matrices=False, compute_uv=False))
        q, rr = np.linalg.qr(m, mode='reduced')
        chain.cores[i] = q.reshape(l, d, e, q.shape[1])
        chain.cores[i+1] = np.tensordot(rr, chain.cores[i+1], axes=(1, 0))
    return out


def projected_entries(bonds: tuple[int, ...] | list[int],
                      envs: tuple[int, ...] | list[int]) -> int:
    if len(envs) != len(bonds)+1 or any(x < 1 for x in bonds) or any(x < 1 for x in envs):
        raise ValueError('invalid bond/environment ranks')
    full = (1, *bonds, 1)
    return int(sum(2 * full[i] * envs[i] * full[i+1] for i in range(len(envs))))


@dataclass(frozen=True)
class JointPlan:
    bond_ranks: tuple[int, ...]
    environment_ranks: tuple[int, ...]
    predicted_entries: int
    budget_entries: int
    predicted_squared_tail: float
    channel_bond_weight: float
    num_rank_additions: int


def allocate_joint(s_bonds: list[np.ndarray], s_envs: list[np.ndarray],
                   budget_entries: int, bond_cap: int, env_cap: int,
                   channel_bond_weight: float = 1.0) -> JointPlan:
    """Greedy multi-choice allocator with exact nonlinear storage accounting.

    Each extra rank changes neighboring entry costs, recomputed at every step.
    A local singular-value squared gain is divided by its incremental entry
    cost; this is deliberately a heuristic rather than a global optimality
    claim or an error bound. Tie-breaking is deterministic.
    """
    n = len(s_envs)
    if n < 2 or len(s_bonds) != n-1 or bond_cap < 1 or env_cap < 1:
        raise ValueError('invalid spectra or rank limits')
    if not isinstance(budget_entries, (int, np.integer)) or budget_entries < 2*n:
        raise ValueError('entry budget is infeasible for minimal tensors')
    if not np.isfinite(channel_bond_weight) or not 0 <= channel_bond_weight <= 1+1e-9:
        raise ValueError('invalid channel contraction weight')
    spectra = s_bonds + s_envs
    if any(len(s)==0 or np.any(~np.isfinite(s)) or np.any(s<0) for s in spectra):
        raise ValueError('singular spectra must be nonempty, finite, nonnegative')
    bc = [min(bond_cap,select_rank(s,None,0)) for s in s_bonds]
    ec = [min(env_cap,select_rank(s,None,0)) for s in s_envs]
    bs = [1]*(n-1)
    es = [1]*n
    used = 2*n
    additions = 0
    while True:
        all_b = [1, *bs, 1]
        candidates=[]
        for i,s in enumerate(s_bonds):
            extra=2*(all_b[i]*es[i]+all_b[i+2]*es[i+1])
            if bs[i] < bc[i] and used+extra <= budget_entries:
                benefit=channel_bond_weight*float(s[bs[i]]**2)
                candidates.append((benefit/extra,benefit,-i,'b',i,extra))
        for i,s in enumerate(s_envs):
            extra=2*all_b[i]*all_b[i+1]
            if es[i] < ec[i] and used+extra <= budget_entries:
                benefit=float(s[es[i]]**2)
                candidates.append((benefit/extra,benefit,-i,'e',i,extra))
        if not candidates: break
        score,benefit,_,kind,i,cost=max(candidates)
        if benefit <= 1e-28: break
        (bs if kind=='b' else es)[i]+=1
        used+=cost
        additions+=1
    assert projected_entries(bs,es)==used
    squared_tail=(sum(channel_bond_weight*float(np.sum(s[b:]**2))
                      for s,b in zip(s_bonds,bs))+
                  sum(float(np.sum(s[e:]**2)) for s,e in zip(s_envs,es)))
    return JointPlan(tuple(bs),tuple(es),used,int(budget_entries),
                     squared_tail,float(channel_bond_weight),additions)


@dataclass(frozen=True)
class JointResult:
    plan: JointPlan
    retained_entries: int
    predicted_working_peak_entries: int
    discarded_squared_weight: float
    layer_error_bound_increment: float
    min_local_retention: float
    marginal_changes: tuple[float,...]
    seconds: float


def one_site_densities(chain: PurifiedChain) -> list[np.ndarray]:
    """Exact one-site marginals of the *current approximation* via TN contractions.

    Used to quantify physical changes caused by compression, rather than
    reconstruct a full density matrix. These marginals do not upper-bound
    global trace distance.
    """
    right=[None]*(chain.n+1)
    right[-1]=np.ones((1,1),complex)
    for j in range(chain.n-1,-1,-1):
        a=chain.cores[j]
        right[j]=np.einsum('lpar,mpas,rs->lm',a,a.conj(),right[j+1],optimize=True)
    left=np.ones((1,1),complex)
    out=[]
    tr=chain.trace()
    for j,a in enumerate(chain.cores):
        rho=np.einsum('lm,liar,mjas,rs->ij',left,a,a.conj(),right[j+1],optimize=True)
        out.append((rho+rho.conj().T)/(2*tr))
        left=np.einsum('lm,lpar,mpas->rs',left,a,a.conj(),optimize=True)
    return out


class JointPurifiedChain(PurifiedChain):
    def __init__(self, n:int, *, max_bond:int=16, max_env:int=12,
                 work_cap_entries:int|None=None, local_tol:float=0.):
        super().__init__(n,max_bond=max_bond,max_env=max_env,local_tol=local_tol)
        if work_cap_entries is not None and (not isinstance(work_cap_entries,int) or work_cap_entries<1):
            raise ValueError('work cap must be a positive integer')
        self.work_cap_entries=work_cap_entries
        self.predicted_working_peak_entries=self.dimensions()['purified_entries']
        self.error_bound_sum=0.

    def _admit(self, extra: int) -> None:
        estimate=self.dimensions()['purified_entries'] + int(extra)
        if self.work_cap_entries is not None and estimate>self.work_cap_entries:
            raise MemoryError(f'estimated temporary entry demand {estimate} exceeds cap {self.work_cap_entries}')
        self.predicted_working_peak_entries=max(self.predicted_working_peak_entries,estimate)

    def gate(self,site:int,unitary:np.ndarray)->None:
        if not 0<=site<self.n-1 or unitary.shape!=(4,4):
            raise ValueError('invalid gate')
        a,b=self.cores[site],self.cores[site+1]
        l,_,ea,_=a.shape; _,_,eb,r=b.shape
        rows=l*2*ea; cols=2*eb*r; min_dim=min(rows,cols)
        # theta, matricization, thin-SVD U, Vh, plus a copy of the input.
        self._admit(3*rows*cols + (rows+cols+1)*min_dim)
        super().gate(site,unitary)
        loss=self.truncations[-1].discarded_weight
        if loss>0:
            # Orthogonal SVD truncation before further operations. The
            # post-truncation norm is exactly (pre-truncation norm - loss).
            after=self.trace()
            self.error_bound_sum+=float(np.sqrt(min(1.,loss/(after+loss))))

    def channel(self,site:int,operators:tuple[np.ndarray,...])->None:
        if not 0<=site<self.n:
            raise ValueError('invalid channel site')
        after=self.cores[site].size*len(operators)
        self._admit(2*after)
        super().channel(site,operators)

    def _preflight_sweeps(self):
        # QR/SVD intermediate arrays are not all simultaneously live. This
        # conservative modeled demand is enforced before changing gauges.
        total=self.dimensions()['purified_entries']
        max_mat=max(a.size for a in self.cores)
        for a in self.cores:
            l,p,e,r=a.shape
            max_mat=max(max_mat,l*p*e*r,l*p*r*e)
        self._admit(4*max_mat)

    def compress_layer_joint(self,budget_entries:int,
                             operators:tuple[np.ndarray,...],
                             *, bond_cap:int|None=None,
                             env_cap:int|None=None,
                             bond_weight_override:float|None=None)->JointResult:
        if bond_cap is not None and bond_cap < 1:
            raise ValueError('bond_cap must be positive')
        if env_cap is not None and env_cap < 1:
            raise ValueError('env_cap must be positive')
        start=perf_counter()
        self._preflight_sweeps()
        bond_s=spatial_spectra(self)
        env_s,_=scan_ancilla_spectra(self)
        weight=(pauli_transfer_norm(operators) if bond_weight_override is None
                else float(bond_weight_override))
        plan=allocate_joint(bond_s,env_s,budget_entries,
                            bond_cap or self.max_bond,env_cap or self.max_env,
                            channel_bond_weight=weight)
        before_marginals=one_site_densities(self)
        loss_sum=0.
        error_before=self.error_bound_sum
        minimum=1.
        # Each SVD acts across a spatial cut. A right-orthogonal gauge means
        # its singular values are those of the complete purification.
        self._right_qr(1)
        for i,target in enumerate(plan.bond_ranks):
            a=self.cores[i]
            l,p,e,r=a.shape
            mat=a.reshape(l*p*e,r)
            self._admit(3*mat.size+(mat.shape[0]+mat.shape[1])*min(mat.shape))
            u,s,vh=stable_svd(mat)
            rank=min(target,select_rank(s,None,self.local_tol))
            dropped=float(np.sum(s[rank:]**2))
            full=float(np.sum(s**2))
            if full<=0:raise ArithmeticError('zero purification norm')
            self.error_bound_sum+=np.sqrt(min(1.,dropped/full))
            loss_sum+=dropped
            minimum=min(minimum,1.-dropped/full)
            self.truncations.append(Truncation('bond',i,rank,len(s),dropped))
            self.cores[i]=u[:,:rank].reshape(l,p,e,rank)
            self.cores[i+1]=np.tensordot(s[:rank,None]*vh[:rank,:],
                                          self.cores[i+1],axes=(1,0))
        # Spectra are pretruncation. Subsequent env SVDs measure updated loss.
        self._right_qr(1)
        for i,target in enumerate(plan.environment_ranks):
            a=self.cores[i]
            l,p,e,r=a.shape
            mat=a.transpose(0,1,3,2).reshape(l*p*r,e)
            self._admit(3*mat.size+(mat.shape[0]+mat.shape[1])*min(mat.shape))
            u,s,_=stable_svd(mat)
            rank=min(target,select_rank(s,None,self.local_tol))
            dropped=float(np.sum(s[rank:]**2))
            full=float(np.sum(s**2))
            if full<=0:raise ArithmeticError('zero purification norm')
            self.error_bound_sum+=np.sqrt(min(1.,dropped/full))
            loss_sum+=dropped
            minimum=min(minimum,1.-dropped/full)
            self.truncations.append(Truncation('environment',i,rank,len(s),dropped))
            self.cores[i]=(u[:,:rank]*s[:rank]).reshape(l,p,r,rank).transpose(0,1,3,2)
            if i+1<self.n:
                updated=self.cores[i]
                q,rr=np.linalg.qr(updated.reshape(l*p*rank,r),mode='reduced')
                self.cores[i]=q.reshape(l,p,rank,q.shape[1])
                self.cores[i+1]=np.tensordot(rr,self.cores[i+1],axes=(1,0))
        retained=self.dimensions()['purified_entries']
        if retained>budget_entries:
            raise AssertionError(f'joint allocation exceeded boundary budget {retained}>{budget_entries}')
        after_marginals=one_site_densities(self)
        changes=tuple(float(np.linalg.norm(after-before,ord='nuc')/2)
                      for before,after in zip(before_marginals,after_marginals))
        return JointResult(plan,retained,self.predicted_working_peak_entries,
                           loss_sum,self.error_bound_sum-error_before,minimum,
                           changes,perf_counter()-start)
