"""Continue the deep five-copy coarse N=3 solution with unregularised LM in chunks,
saving a snapshot after each chunk, to measure how each coordinate scales as the
solution approaches the border (eps -> 0).  Run from tensor-lab/."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import copy_cost_lm, coarse_pattern
pat = coarse_pattern(3)
th = np.load('logs/deep_N3_c5_theta.npy')
np.save('logs/snap_N3_c5_0.npy', th)
t0 = time.time()
for k in range(1, 7):
    r = copy_cost_lm(2, 0, 3, 5, starts=1, iters=600, mu=0.0, init=th, pattern=pat, tol=1e-30)
    th = r['theta']
    np.save(f'logs/snap_N3_c5_{k}.npy', th)
    print(f'chunk {k}: residual {r["residual"]:.3e} norm {r["norm"]:.1f} [{time.time() - t0:.0f}s]', flush=True)
print('DONE', flush=True)
