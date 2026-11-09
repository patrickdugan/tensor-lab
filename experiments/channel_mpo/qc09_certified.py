"""QC-09: residual-certified gate factors and peak-aware rank scheduling.

A numerical *a posteriori* certificate computes the Frobenius residual in
row blocks. In exact arithmetic the induced normalized-purification trace
distance is at most 2*relative_residual. This is an analytical inequality
applied to measured floating-point residuals, not an interval-arithmetic proof.

The peak model counts explicit NumPy complex-array entries, including resident
cores, gate theta and factor outputs; it does not include LAPACK-internal
buffers, Python allocator caches, or the process RSS. Actual RSS is separately
benchmarked in isolated subprocesses. The rank scheduler enforces a prospective
next-gate peak model and a strict layer-boundary retained-entry budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from channel_mpo import gate_layers
from purified_mpo import Truncation, select_rank, stable_svd
from qc07_joint import JointPlan, allocate_joint, projected_entries
from qc08_streaming import StreamingJointChain


def gate_live_model(rows: int, cols: int, resident_entries: int,
                    rank_cap: int, *, method: str = 'sketch',
                    oversample: int = 6, residual_block: int = 16) -> int:
    """Conservative explicit-array entry model for one two-site gate.

    Covers multiple transformed-theta temporaries, input and output factors,
    sketch/QR/SVD arrays, and chunked residual. Complex and real entries are
    treated at equal cost, so this is a *model*, not a byte-accurate ceiling.
    """
    if min(rows,cols,resident_entries,rank_cap,residual_block) < 1 or oversample < 0:
        raise ValueError('positive matrix shape/resident/rank/block required')
    if method not in ('full', 'sketch', 'auto'):
        raise ValueError('unknown gate decomposition')
    d=min(rows,cols)
    k=min(d,rank_cap+oversample)
    use_sketch = method=='sketch' or (method=='auto' and d >= 2*k and rows*cols >= 512)
    theta=rows*cols
    # The input M is a reshape/view of theta; einsum tends to allocate another
    # dense theta and intermediate contraction storage. Leave room for 3 copies.
    if use_sketch:
        workspace = 3*theta + 4*rows*k + 3*k*cols + 3*k*k + min(rows,residual_block)*cols
    else:
        workspace = 3*theta + 2*rows*d + 2*d*cols + d*d
    outputs = rank_cap*(rows+cols)
    return int(resident_entries + workspace + outputs)


def projected_gate_peak(bonds: tuple[int,...] | list[int],
                        envs: tuple[int,...] | list[int], *,
                        rank_cap:int,method:str='sketch',oversample:int=6)->int:
    """Worst adjacent two-qubit gate if *planned* ranks are the inputs.

    Includes an explicit model of peak simultaneous live entries and the
    retained resident cores. Channel growth before the next boundary sweep
    and hidden BLAS workspace are excluded and disclosed separately.
    """
    full=(1,*bonds,1)
    if len(full)!=len(envs)+1: raise ValueError('incorrect rank vectors')
    resident=projected_entries(bonds,envs)
    return max(gate_live_model(2*full[i]*envs[i],2*full[i+2]*envs[i+1],
                               resident,rank_cap,method=method,oversample=oversample)
               for i in range(len(envs)-1))


def projected_next_channel_peak(bonds:tuple[int,...]|list[int],
                                envs:tuple[int,...]|list[int],*,
                                rank_cap:int,kraus_count:int,
                                workspace_factor:int=6)->int:
    """Conservative next-layer local-noise workspace *model*.

    Gate evolution may restore each spatial bond up to ``rank_cap`` before
    applying Kraus operators. We therefore inflate all prospective bonds to
    rank_cap and account for a full sweep of local environment enlargement,
    plus a workspace allowance for the largest enlarged core. The requested
    gate count is deliberately not used as a predictor of the next bonds.
    """
    if rank_cap<1 or kraus_count<1 or workspace_factor<0:
        raise ValueError('invalid channel workspace model')
    inflated=(rank_cap,)*(len(envs)-1)
    resident=projected_entries(inflated,envs)
    full=(1,*inflated,1)
    biggest=max(2*full[i]*envs[i]*full[i+1] for i in range(len(envs)))
    return int(kraus_count*(resident+workspace_factor*biggest))


@dataclass(frozen=True)
class PeakPlan:
    joint:JointPlan
    modeled_next_gate_peak_entries:int
    cap_entries:int|None
    blocked_increments:int
    modeled_next_channel_peak_entries:int=0
    modeled_next_layer_peak_entries:int=0


def allocate_peak_safe(s_bonds:list[np.ndarray],s_envs:list[np.ndarray],
                       budget_entries:int,bond_cap:int,env_cap:int,
                       *,channel_bond_weight:float=1.,
                       peak_cap_entries:int|None=None,
                       gate_method:str='sketch',oversample:int=6,
                       safety_factor:float=1.,kraus_count:int=1,
                       account_channel_peak:bool=False)->PeakPlan:
    """Greedy joint-rank allocator with prospective live-buffer admission.

    The nonbinding case delegates directly to QC-07's allocator to preserve
    the exact tie-breaking and rank plan. With a cap, each candidate rank
    increment is rejected if its modelled largest next-gate footprint exceeds
    the cap. Both budget constraints are checked after selection.
    """
    if not np.isfinite(safety_factor) or safety_factor<1. or kraus_count<1:
        raise ValueError('safety factor must be finite and >= 1')
    standard=allocate_joint(s_bonds,s_envs,budget_entries,bond_cap,env_cap,
                            channel_bond_weight=channel_bond_weight)
    standard_peak=projected_gate_peak(standard.bond_ranks,standard.environment_ranks,
                                     rank_cap=bond_cap,method=gate_method,oversample=oversample)
    def full_peak(bs,es):
        gate=projected_gate_peak(bs,es,rank_cap=bond_cap,
                                  method=gate_method,oversample=oversample)
        noise=(projected_next_channel_peak(bs,es,rank_cap=bond_cap,
                                           kraus_count=kraus_count)
               if account_channel_peak else 0)
        return gate,noise,max(gate,noise)
    _,std_channel,std_max=full_peak(standard.bond_ranks,standard.environment_ranks)
    if peak_cap_entries is None or std_max*safety_factor <= peak_cap_entries:
        return PeakPlan(standard,standard_peak,peak_cap_entries,0,std_channel,std_max)
    if not isinstance(peak_cap_entries,(int,np.integer)) or peak_cap_entries<1:
        raise ValueError('positive peak cap required')
    n=len(s_envs)
    bs=[1]*(n-1);es=[1]*n;used=2*n;num_add=0;blocked=0
    permanently_blocked=set()
    min_peak=full_peak(bs,es)[2]
    if min_peak*safety_factor>peak_cap_entries:
        raise MemoryError(f'minimum feasible gate model {min_peak} exceeds cap {peak_cap_entries}')
    bc=[min(bond_cap, int(np.count_nonzero(s>1e-13*s[0])) or 1) for s in s_bonds]
    ec=[min(env_cap, int(np.count_nonzero(s>1e-13*s[0])) or 1) for s in s_envs]
    while True:
        all_b=[1,*bs,1]; candidates=[]
        for i,s in enumerate(s_bonds):
            extra=2*(all_b[i]*es[i]+all_b[i+2]*es[i+1])
            if ('b',i) not in permanently_blocked and bs[i]<bc[i] and used+extra<=budget_entries:
                benefit=channel_bond_weight*float(s[bs[i]]**2)
                candidates.append((benefit/extra,benefit,-i,'b',i,extra))
        for i,s in enumerate(s_envs):
            extra=2*all_b[i]*all_b[i+1]
            if ('e',i) not in permanently_blocked and es[i]<ec[i] and used+extra<=budget_entries:
                benefit=float(s[es[i]]**2)
                candidates.append((benefit/extra,benefit,-i,'e',i,extra))
        if not candidates: break
        # Skip candidates blocked on peak rather than terminating the search.
        selected=None
        for candidate in sorted(candidates,reverse=True):
            if candidate[1]<=1e-28:break
            _,_,_,kind,i,cost=candidate
            target=bs if kind=='b' else es
            target[i]+=1
            peak=full_peak(bs,es)[2]
            if peak*safety_factor<=peak_cap_entries:
                selected=candidate
                used+=cost;num_add+=1
                break
            target[i]-=1;blocked+=1
            # Gate/workspace models are monotone in all ranks. A rejected
            # increment stays infeasible as other ranks only grow.
            permanently_blocked.add((kind,i))
        if selected is None:break
    if projected_entries(bs,es)!=used:raise AssertionError('rank costs disagree')
    squared_tail=(sum(channel_bond_weight*float(np.sum(s[b:]**2)) for s,b in zip(s_bonds,bs))+
                  sum(float(np.sum(s[e:]**2)) for s,e in zip(s_envs,es)))
    joint=JointPlan(tuple(bs),tuple(es),used,int(budget_entries),squared_tail,
                    float(channel_bond_weight),num_add)
    gate,channel,peak=full_peak(bs,es)
    if peak*safety_factor>peak_cap_entries:raise AssertionError('gate-cap violated')
    return PeakPlan(joint,gate,int(peak_cap_entries),blocked,channel,peak)


@dataclass(frozen=True)
class ResidualCertificate:
    method:str
    kept_rank:int
    matrix_shape:tuple[int,int]
    input_fro_norm:float
    residual_fro_norm:float
    relative_residual:float
    trace_distance_bound:float
    projection_defect:float
    chunks_checked:int
    seconds:float


def _chunked_residual(m:np.ndarray,left:np.ndarray,right:np.ndarray,
                      chunk_rows:int)->tuple[float,float,int]:
    """Direct residual check, never creating a full-size residual matrix."""
    if chunk_rows<1:raise ValueError('chunk_rows must be positive')
    residual_sq=0.; norm_sq=0.; blocks=0
    for lo in range(0,m.shape[0],chunk_rows):
        hi=min(m.shape[0],lo+chunk_rows)
        mm=m[lo:hi]
        difference=mm-left[lo:hi]@right
        residual_sq+=float(np.vdot(difference,difference).real)
        norm_sq+=float(np.vdot(mm,mm).real)
        blocks+=1
    return float(np.sqrt(max(0,residual_sq))),float(np.sqrt(max(0,norm_sq))),blocks


def certified_gate_factor(m:np.ndarray,rank_cap:int,*,method:str='auto',
                          oversample:int=6,power_iters:int=2,
                          seed:int=0,local_tol:float=0.,
                          chunk_rows:int=16)->tuple[np.ndarray,np.ndarray,ResidualCertificate]:
    """Compute rank-k factors and independently measure their residual.

    Sketch uses a deterministic Gaussian range finder followed by exact thin
    SVD of Q* M; full uses thin SVD directly. The residual is independently
    computed from M and the final factors. The analytical inequality for
    normalized purifications is D <= min(1,2 ||M-LR||_F/||M||_F), assuming M
    is the canonical center and the other tensors are isometries.
    """
    t0=perf_counter()
    if m.ndim!=2 or min(m.shape)<1 or not np.issubdtype(m.dtype,np.number):
        raise ValueError('nonempty numeric matrix required')
    if rank_cap<1 or oversample<0 or power_iters<0 or not np.isfinite(local_tol) or local_tol<0:
        raise ValueError('invalid factor parameters')
    if not np.isfinite(m).all():raise FloatingPointError('nonfinite matrix')
    rows,cols=m.shape;d=min(rows,cols);width=min(d,rank_cap+oversample)
    if method not in ('auto','full','sketch'):raise ValueError('unknown factorization')
    chosen=('sketch' if (method=='sketch' or
                         method=='auto' and d>=2*width and m.size>=512)
            else 'full')
    if chosen=='sketch':
        rng=np.random.default_rng(seed)
        omega=(rng.standard_normal((cols,width))+
               1j*rng.standard_normal((cols,width)))/np.sqrt(2)
        q,_=np.linalg.qr(m@omega,mode='reduced')
        for _ in range(power_iters):
            z=m.conj().T@q
            q,_=np.linalg.qr(m@z,mode='reduced')
        b=q.conj().T@m
        ub,s,vh=stable_svd(b)
        kept=select_rank(s,rank_cap,local_tol)
        left=q@ub[:,:kept]
    else:
        u,s,vh=stable_svd(m)
        kept=select_rank(s,rank_cap,local_tol)
        left=u[:,:kept]
    right=s[:kept,None]*vh[:kept]
    residual,mnorm,blocks=_chunked_residual(m,left,right,chunk_rows)
    if mnorm<=0:raise ArithmeticError('zero gate input')
    ratio=residual/mnorm
    # Projection defect is the discrepancy from ideal orthogonal projection.
    # It is recorded explicitly; the generic 2*ratio bound remains valid.
    projected_norm_sq=float(np.vdot(right,right).real)
    # (left@right) is a projection if left columns orthonormal and B uses Q* M.
    cross=complex(np.vdot(right,left.conj().T@m))
    defect=float(abs(cross-projected_norm_sq)/(mnorm*mnorm))
    if not np.isfinite(ratio+defect):raise FloatingPointError('invalid residual certificate')
    cert=ResidualCertificate(chosen,kept,(rows,cols),mnorm,residual,ratio,
                              min(1.,2*ratio),defect,blocks,perf_counter()-t0)
    return left,right,cert


@dataclass(frozen=True)
class PeakLayerResult:
    plan:PeakPlan
    retained_entries:int
    normalized_trace_before:float
    modeled_live_peak_entries:int
    gate_bound_sum:float
    layer_error_bound_sum:float
    seconds:float


class CertifiedPeakChain(StreamingJointChain):
    """QC-08 compression plus independently checked two-site gate factors.

    Use `compress_layer_peak` to ensure rank allocation respects a predeclared
    prospective next-gate peak cap. `gate` checks its own live-buffer model
    before allocating theta; it records a *modeled* peak, not actual RSS.
    """
    def __init__(self,n:int,*,max_bond:int=12,max_env:int=12,
                 work_cap_entries:int|None=None,
                 peak_gate_cap_entries:int|None=None,
                 gate_method:str='auto',oversample:int=6,power_iters:int=2,
                 residual_chunk_rows:int=16,
                 planner_safety_factor:float=1.08):
        super().__init__(n,max_bond=max_bond,max_env=max_env,
                         work_cap_entries=work_cap_entries)
        if peak_gate_cap_entries is not None and peak_gate_cap_entries<1:
            raise ValueError('gate cap must be positive')
        self.peak_gate_cap_entries=peak_gate_cap_entries
        self.gate_method=gate_method
        self.oversample=oversample
        self.power_iters=power_iters
        self.residual_chunk_rows=residual_chunk_rows
        self.planner_safety_factor=planner_safety_factor
        self.gate_certificates:list[ResidualCertificate]=[]
        self.modeled_live_peak_entries=0
        self.peak_rank_plans:list[PeakPlan]=[]

    def _admit(self,extra:int)->None:
        """Audit QC-08 channel/compression preflight as part of live budget."""
        current=self.dimensions()['purified_entries']
        proposal=current+int(extra)
        if self.peak_gate_cap_entries is not None and proposal>self.peak_gate_cap_entries:
            raise MemoryError(f'live-array model {proposal} exceeds cap {self.peak_gate_cap_entries}')
        super()._admit(extra)
        self.modeled_live_peak_entries=max(self.modeled_live_peak_entries,proposal)

    def gate(self,site:int,unitary:np.ndarray)->None:
        if not isinstance(site,int) or not 0<=site<self.n-1 or unitary.shape!=(4,4):
            raise ValueError('invalid nearest-neighbor gate')
        # Preflight with input cores before canonicalization. QR can alter
        # intermediate ranks, so a second exact-shape preflight follows.
        a,b=self.cores[site:site+2]
        rows=a.shape[0]*2*a.shape[2];cols=2*b.shape[2]*b.shape[-1]
        rough=gate_live_model(rows,cols,sum(x.size for x in self.cores),
                              self.max_bond,method=self.gate_method,
                              oversample=self.oversample,residual_block=self.residual_chunk_rows)
        if self.peak_gate_cap_entries is not None and rough>self.peak_gate_cap_entries:
            raise MemoryError(f'preflight peak model {rough} exceeds cap {self.peak_gate_cap_entries}')
        self._canonicalize(site,site+1)
        a,b=self.cores[site:site+2]
        l,_,ea,_=a.shape;_,_,eb,r=b.shape
        rows=l*2*ea;cols=2*eb*r
        model=gate_live_model(rows,cols,sum(x.size for x in self.cores),
                              self.max_bond,method=self.gate_method,
                              oversample=self.oversample,residual_block=self.residual_chunk_rows)
        if self.peak_gate_cap_entries is not None and model>self.peak_gate_cap_entries:
            raise MemoryError(f'canonical gate peak model {model} exceeds cap {self.peak_gate_cap_entries}')
        theta=np.tensordot(a,b,axes=(3,0))
        theta=np.einsum('xyuv,luavbr->lxaybr',unitary.reshape(2,2,2,2),theta,optimize=True)
        matrix=theta.reshape(rows,cols)
        left,right,cert=certified_gate_factor(matrix,self.max_bond,
                 method=self.gate_method,oversample=self.oversample,
                 power_iters=self.power_iters,seed=1009+site+29*len(self.gate_certificates),
                 local_tol=self.local_tol,chunk_rows=self.residual_chunk_rows)
        kept=cert.kept_rank
        self.cores[site]=left.reshape(l,2,ea,kept)
        self.cores[site+1]=right.reshape(kept,2,eb,r)
        self.truncations.append(Truncation('bond',site,kept,min(rows,cols),cert.residual_fro_norm**2))
        self.error_bound_sum+=cert.trace_distance_bound
        self.gate_certificates.append(cert)
        self.modeled_live_peak_entries=max(self.modeled_live_peak_entries,model)
        self.predicted_working_peak_entries=max(self.predicted_working_peak_entries,model)

    def gate_layer(self,layer)->int:
        pairs=list(layer)
        last=-2
        for i,u in pairs:
            if not isinstance(i,int) or i<=last+1 or i<0 or i>=self.n-1 or u.shape!=(4,4):
                raise ValueError('sorted disjoint two-site gates required')
            last=i
        for i,u in pairs:self.gate(i,u)
        return len(pairs)

    def compress_layer_peak(self,budget_entries:int,operators:tuple[np.ndarray,...],
                            *,bond_cap:int|None=None,env_cap:int|None=None,
                            bond_weight_override:float|None=None,
                            track_marginals:bool=False)->PeakLayerResult:
        begin=perf_counter();before=self.error_bound_sum
        cap_b=bond_cap or self.max_bond
        cap_e=env_cap or self.max_env
        def peak_allocator(s_bonds,s_envs,budget_entries,bond_cap,env_cap,
                           *,channel_bond_weight):
            selected=allocate_peak_safe(s_bonds,s_envs,budget_entries,bond_cap,env_cap,
                    channel_bond_weight=channel_bond_weight,
                    peak_cap_entries=self.peak_gate_cap_entries,
                    gate_method=self.gate_method,oversample=self.oversample,
                    safety_factor=self.planner_safety_factor,
                    kraus_count=len(operators),account_channel_peak=True)
            self.peak_rank_plans.append(selected)
            return selected.joint
        result=self.compress_layer_stream(budget_entries,operators,
             bond_cap=cap_b,env_cap=cap_e,bond_weight_override=bond_weight_override,
             order='classic',track_marginals=track_marginals,rank_allocator=peak_allocator)
        # Normalization is deliberately after compression; the local certificate
        # applies to normalized purifications in a hypothetical equivalent path.
        retained=self.normalize()
        return PeakLayerResult(self.peak_rank_plans[-1],result.retained_entries,retained,
                               self.modeled_live_peak_entries,
                               sum(x.trace_distance_bound for x in self.gate_certificates),
                               self.error_bound_sum-before,perf_counter()-begin)
