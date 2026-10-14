import sys, time
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import copy_cost_continuation, copy_cost_lm, coarse_pattern
N, c, seed = (int(v) for v in sys.argv[1:4])
pat = coarse_pattern(N)
mus = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
t0 = time.time()
r = copy_cost_continuation(2, 0, N, c, starts=1, iters=300, mus=mus, seed=seed, pattern=pat)
th = r['theta']
print(f'coarse N={N} c={c} seed={seed} reproduced start 0: ' + ' '.join(f'{res:.1e}' for _, res, _ in r['path']) + f' [{time.time()-t0:.0f}s]', flush=True)
for mu in (1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 0.0):
    rr = copy_cost_lm(2, 0, N, c, starts=1, iters=1500, mu=mu, init=th, pattern=pat)
    th = rr['theta']
    print(f'  mu={mu:.0e}: residual {rr["residual"]:.3e}  norm {rr["norm"]:.1f}  [{time.time()-t0:.0f}s]', flush=True)
np.save(f'logs/deep_N{N}_c{c}_theta.npy', th)
print('DONE', flush=True)
