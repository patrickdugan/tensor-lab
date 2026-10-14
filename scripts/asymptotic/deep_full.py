"""Reproduce one start of many_starts.py on the full block form and push the
continuation further, saving snapshots.  Usage (from tensor-lab/):
  python scripts/asymptotic/deep_full.py n N c seed"""
import sys
import time

sys.path.insert(0, '.')
import numpy as np

from tensorlab.numeric_restriction import _setup_pattern, border_path, copy_cost_lm

n, N, c, seed = (int(v) for v in sys.argv[1:5])
MUS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10)
pat = _setup_pattern(2, n, N)[:3]
t0 = time.time()
out = border_path(2, 0, N, c, mus=MUS, starts=1, iters=250, seed=seed, pattern=pat)
th = out['theta']
print(f'full n={n} N={N} c={c} seed={seed} reproduced: ' + ' '.join(f'{r:.1e}' for _, r, _ in out['path']) +
      f' | norm {out["path"][-1][2]:.1f} [{time.time() - t0:.0f}s]', flush=True)
np.save(f'logs/deepfull_n{n}_N{N}_c{c}_s{seed}_0.npy', th)
for k, mu in enumerate((1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 0.0), start=1):
    rr = copy_cost_lm(2, n, N, c, starts=1, iters=1500, mu=mu, init=th, pattern=pat)
    th = rr['theta']
    np.save(f'logs/deepfull_n{n}_N{N}_c{c}_s{seed}_{k}.npy', th)
    print(f'  mu={mu:.0e}: residual {rr["residual"]:.3e} norm {rr["norm"]:.1f} [{time.time() - t0:.0f}s]', flush=True)
print('DONE', flush=True)
