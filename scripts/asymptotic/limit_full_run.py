"""Limit-system search on the full p = 2 block form of C(3, 2n) (path boundary kept).
Usage (from tensor-lab/):  python scripts/asymptotic/limit_full_run.py n N r t starts iters seed"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import limit_system_search, _setup_pattern
n, N, r, t, starts, iters, seed = (int(v) for v in sys.argv[1:8])
t0 = time.time()
pat = _setup_pattern(2, n, N)[:3]
print(f"full n={n} N={N}: {r} rank-one + {t} tangential ({r + 2 * t} copies), support {len(pat[1])}, "
      f"dims {pat[2]}, starts={starts} iters={iters} seed={seed}", flush=True)
out = limit_system_search(pat, r, t, starts=starts, iters=iters, seed=seed, verbose=True)
print(f"BEST residual {out['residual']:.3e} norm {out['path'][-1][2]:.2f} [{time.time() - t0:.0f}s]", flush=True)
np.save(f"logs/limitfull_n{n}_N{N}_r{r}_t{t}_s{seed}.npy", out["theta"])
print("DONE", flush=True)
