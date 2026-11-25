"""QC-03: pure-state MPS quantum trajectories as a positive independent oracle.

Kraus outcome k is drawn with probability ||(K_k)_j |psi>||^2 and the
chosen pure state is normalized. The empirical mean of each local Pauli-Z
is unbiased for the exact mixed-state expectation when MPS bonds are kept
without truncation. A bounded bond can bias the estimator; recorded gate
truncation diagnostics must be checked separately.

Two *equivalent CPTP maps* are unraveled differently for depolarizing noise:
- pauli: a four-operator random-unitary representation;
- reset: 5 Kraus operators (sqrt(1-p) I and sqrt(p/2) |r><s|).
The reset representation contains entanglement-cutting measurement outcomes.
This experiment tests representation-dependent tensor-network cost; results
cannot be interpreted as different physical noise channels.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from channel_mpo import I2, apply_channel, apply_operator, gate_layers, kraus
from purified_mpo import PurifiedChain


def depolarizing_reset_kraus(p: float) -> tuple[np.ndarray, ...]:
    if not 0 <= p <= 1:
        raise ValueError("p outside [0,1]")
    ks = [np.sqrt(1-p)*I2]
    for r in range(2):
        for s in range(2):
            k = np.zeros((2,2), dtype=complex)
            k[r,s] = np.sqrt(p/2)
            ks.append(k)
    return tuple(ks)


def unravelling(name: str, p: float, scheme: str,
                reset_mix: float = 0.5) -> tuple[np.ndarray,...]:
    if not 0 <= reset_mix <= 1:
        raise ValueError('reset_mix must lie in [0,1]')
    if name == 'depolarizing' and scheme == 'hybrid':
        pauli = kraus(name,p)
        reset = depolarizing_reset_kraus(p)
        return tuple(np.sqrt(1-reset_mix)*k for k in pauli) + tuple(
            np.sqrt(reset_mix)*k for k in reset)
    if name == "depolarizing" and scheme == "reset":
        return depolarizing_reset_kraus(p)
    if scheme in ("canonical", "pauli"):
        return kraus(name, p)
    raise ValueError(f"unsupported unravelling {name}/{scheme}")


class TrajectoryChain(PurifiedChain):
    def __init__(self, n: int, max_bond: int | None, seed: int):
        super().__init__(n, max_bond, max_env=1)
        self.rng = np.random.default_rng(seed)
        self.total_gate_discarded_weight = 0.0
        self.min_gate_retention = 1.0
        self.num_sampled_jumps = 0

    def apply_gate(self, site: int, unitary: np.ndarray) -> None:
        self.gate(site, unitary)
        loss = self.truncations[-1].discarded_weight
        self.total_gate_discarded_weight += loss
        retained = self.normalize()
        self.min_gate_retention = min(self.min_gate_retention, retained)

    def sample_noise_layer(self, operators: tuple[np.ndarray,...]) -> None:
        # One right-canonical sweep and a left-moving orthogonality center.
        # This avoids an O(n)-site sweep for each sampled local channel.
        self._right_qr(1)
        ks = np.asarray(operators)
        for site in range(self.n):
            a = self.cores[site]
            if a.shape[2] != 1:
                raise AssertionError("pure trajectory unexpectedly has an environment")
            candidates = np.einsum('moi,liar->mloar',ks,a,optimize=True)
            probabilities = np.sum(np.abs(candidates)**2,axis=(1,2,3,4))
            total = float(probabilities.sum())
            if total <= 0 or abs(total-1) > 1e-7:
                raise ArithmeticError(f"invalid Kraus probabilities, total={total}")
            probabilities = probabilities/total
            picked = int(self.rng.choice(len(ks),p=probabilities))
            self.cores[site] = candidates[picked]/np.sqrt(float(probabilities[picked])*total)
            self.num_sampled_jumps += int(picked > 0)
            if site < self.n-1:
                left,_,env,right = self.cores[site].shape
                q,r = np.linalg.qr(self.cores[site].reshape(left*2*env,right),mode="reduced")
                self.cores[site] = q.reshape(left,2,env,q.shape[1])
                self.cores[site+1] = np.tensordot(r,self.cores[site+1],axes=(1,0))


def sample_ensemble(n: int, depth: int, seed: int, channel: str, p: float,
                    scheme: str, shots: int, max_bond: int | None,
                    *, exact_observables: bool = False, reset_mix: float = 0.5) -> dict:
    if n < 2 or depth < 1 or shots < 2 or (max_bond is not None and max_bond < 1):
        raise ValueError("n>=2, depth>=1, shots>=2, bond cap>=1")
    gates = gate_layers(n,depth,seed)
    ks = unravelling(channel,p,scheme,reset_mix)
    data = np.empty((shots,n),dtype=float)
    largest_bond = 1
    largest_entries = 0
    stored_entries = []
    stored_bonds = []
    max_discarded_weight = 0.0
    min_retention = 1.0
    t0 = perf_counter()
    for trial in range(shots):
        # Stable per-trial random stream independent of floating CPU scheduling.
        sample_seed = seed*1000009+trial*65537+12345
        sim = TrajectoryChain(n,max_bond,sample_seed)
        for layer in gates:
            for i,u in layer:
                sim.apply_gate(i,u)
            sim.sample_noise_layer(ks)
        data[trial,:] = sim.z_expectations()
        dims = sim.dimensions()
        largest_bond = max(largest_bond,dims["max_bond"])
        largest_entries = max(largest_entries,dims["purified_entries"])
        stored_entries.append(dims["purified_entries"])
        stored_bonds.append(dims["max_bond"])
        min_retention = min(min_retention,sim.min_gate_retention)
        max_discarded_weight = max(max_discarded_weight,sim.total_gate_discarded_weight)
    seconds=perf_counter()-t0
    means=data.mean(axis=0)
    ses=data.std(axis=0,ddof=1)/np.sqrt(shots)
    result = {
        "n":n,"depth":depth,"seed":seed,"channel":channel,"p":p,
        "unravelling":scheme,"reset_mix":reset_mix if scheme=="hybrid" else "",
        "shots":shots,
        "bond_cap":max_bond if max_bond is not None else "none",
        "seconds":seconds,"seconds_per_shot":seconds/shots,
        "max_observed_bond":largest_bond,"max_stored_complex_entries":largest_entries,
        "mean_final_bond":float(np.mean(stored_bonds)),
        "mean_stored_complex_entries":float(np.mean(stored_entries)),
        "p95_stored_complex_entries":float(np.quantile(stored_entries,.95)),
        "max_cumulative_gate_discarded":max_discarded_weight,
        "min_gate_retention":min_retention,
        "z0_mean":float(means[0]),"z0_se":float(ses[0]),
        "zmid_mean":float(means[n//2]),"zmid_se":float(ses[n//2]),
    }
    if exact_observables:
        if n>10:
            raise ValueError("dense reference limited to 10 qubits")
        rho=np.zeros((2**n,2**n),complex)
        rho[0,0]=1
        for layer in gates:
            for i,u in layer:
                rho=apply_operator(rho,u,(i,i+1),n)
            for i in range(n):
                rho=apply_channel(rho,kraus(channel,p),i,n)
        diag=np.diag(rho).real
        for index,label in [(0,"z0"),(n//2,"zmid")]:
            pattern = np.array([1 if (q >> (n-1-index)) % 2 == 0 else -1
                                for q in range(2**n)])
            target=float(diag@pattern)
            result[f"{label}_exact"]=target
            result[f"{label}_abs_error"]=float(abs(result[f"{label}_mean"]-target))
            result[f"{label}_zscore"] = float((result[f"{label}_mean"]-target)/
                                               max(result[f"{label}_se"],1e-12))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits",default="6,8,20,48")
    ap.add_argument("--depth",type=int,default=6)
    ap.add_argument("--seeds",type=int,default=2)
    ap.add_argument("--channel",default="depolarizing")
    ap.add_argument("--strengths",default="0.2,0.5")
    ap.add_argument("--unravellings",default="pauli,reset")
    ap.add_argument("--shots",type=int,default=128)
    ap.add_argument("--max-bond",type=int,default=16)
    ap.add_argument("--reset-mix",type=float,default=.5)
    ap.add_argument("--validate-small",action="store_true")
    ap.add_argument("--output",type=Path,default=Path("qc03_trajectories.csv"))
    args=ap.parse_args()
    rows=[]
    for n in map(int,args.qubits.split(',')):
        for seed in range(args.seeds):
            for p in map(float,args.strengths.split(',')):
                for scheme in args.unravellings.split(','):
                    result=sample_ensemble(n,args.depth,seed,args.channel,p,scheme,
                                           args.shots,args.max_bond,
                                           exact_observables=args.validate_small and n<=10,
                                           reset_mix=args.reset_mix)
                    rows.append(result)
                    print(f"n={n} seed={seed} p={p} {scheme} "
                          f"z0={result['z0_mean']:+.4f}±{1.96*result['z0_se']:.4f} "
                          f"bond={result['max_observed_bond']} "
                          f"ret={result['min_gate_retention']:.3g} "
                          f"time={result['seconds']:.2f}s",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} records to {args.output}")

if __name__=='__main__':main()
