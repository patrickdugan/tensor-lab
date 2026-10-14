"""Two-copy screen on the full block form of C(3,2n) at N factors with the bounded
rank-2 parametrisation (limit_system_search with r = 2, t = 0): many short starts,
then the best ones continued with a long mu schedule.  A residual that keeps
falling over many decades while the norm grows is a border (degeneration)
solution from two copies.
Usage (from tensor-lab/): python scripts/asymptotic/rank2_screen.py n N screen_starts keep seed0"""
import sys
import time

sys.path.insert(0, '.')
import numpy as np

from tensorlab.numeric_restriction import _setup_pattern, limit_system_search

n, N, starts, keep, seed0 = (int(v) for v in sys.argv[1:6])
pat = _setup_pattern(2, n, N)[:3]
LONG = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 0.0)
t0 = time.time()
print(f'full n={n} N={N}: support {len(pat[1])}, dims {pat[2]}; screening {starts} starts', flush=True)
res = []
for s in range(starts):
    out = limit_system_search(pat, 2, 0, starts=1, iters=200, seed=seed0 + s)
    res.append((out['residual'], seed0 + s))
    print(f'  seed {seed0 + s}: {out["residual"]:.3e} norm {out["path"][-1][2]:.0f} [{time.time() - t0:.0f}s]', flush=True)
res.sort()
print('SCREEN best', [(f'{r:.2e}', s) for r, s in res[:keep]], flush=True)
for r0, s in res[:keep]:
    out = limit_system_search(pat, 2, 0, starts=1, iters=3000, seed=s, mus=LONG)
    print(f'LONG seed {s}: ' + ' '.join(f'{q:.2e}/{nrm:.0f}' for _, q, nrm in out['path']) + f' [{time.time() - t0:.0f}s]', flush=True)
    np.save(f'logs/rank2_n{n}_N{N}_seed{s}.npy', out['theta'])
print('DONE', flush=True)
