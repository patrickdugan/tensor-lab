"""Limit-system search on the coarse pattern: r rank-one + t tangential terms.
Usage (from tensor-lab/):  python scripts/asymptotic/limit_run.py N r t starts iters seed"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import limit_system_search, coarse_pattern
N, r, t, starts, iters, seed = (int(v) for v in sys.argv[1:7])
t0 = time.time()
print(f"coarse N={N}: {r} rank-one + {t} tangential ({r + 2 * t} copies), starts={starts} iters={iters} seed={seed}", flush=True)
out = limit_system_search(coarse_pattern(N), r, t, starts=starts, iters=iters, seed=seed, verbose=True)
print(f"BEST residual {out['residual']:.3e} norm {out['path'][-1][2]:.2f} [{time.time() - t0:.0f}s]", flush=True)
np.save(f"logs/limit_N{N}_r{r}_t{t}_s{seed}.npy", out["theta"])
print("DONE", flush=True)
