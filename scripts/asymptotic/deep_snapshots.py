"""Reproduce the deep five-copy coarse N=3 path (seed 11) and save the parameters
after every regularisation stage, for eps-exponent analysis.  Run from tensor-lab/."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import copy_cost_continuation, copy_cost_lm, coarse_pattern
pat = coarse_pattern(3)
mus = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
t0 = time.time()
r = copy_cost_continuation(2, 0, 3, 5, starts=1, iters=300, mus=mus, seed=11, pattern=pat)
th = r['theta']
k = 0
np.save(f'logs/snap_N3_c5_{k}.npy', th)
print(f'stage {k} (mu=1e-10): residual {r["residual"]:.3e} norm {r["norm"]:.1f} [{time.time()-t0:.0f}s]', flush=True)
for mu in (1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16):
    k += 1
    rr = copy_cost_lm(2, 0, 3, 5, starts=1, iters=1500, mu=mu, init=th, pattern=pat)
    th = rr['theta']
    np.save(f'logs/snap_N3_c5_{k}.npy', th)
    print(f'stage {k} (mu={mu:.0e}): residual {rr["residual"]:.3e} norm {rr["norm"]:.1f} [{time.time()-t0:.0f}s]', flush=True)
print('DONE', flush=True)
