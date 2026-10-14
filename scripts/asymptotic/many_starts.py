"""Many-start two-copy (or c-copy) continuation on one problem, logging every start.

Usage (from tensor-lab/, OMP_NUM_THREADS=1):
  python scripts/asymptotic/many_starts.py full n N c starts seed0
  python scripts/asymptotic/many_starts.py coarse N c starts seed0
  python scripts/asymptotic/many_starts.py box n c starts seed0      (logs/minimal_box_n{n}.json, N = 2)
Each start: regularised continuation mu = 1e-1 ... 1e-10 (border_path), one start per
seed.  Reports final residual and the tail ratio residual(1e-10)/residual(1e-6):
a ratio near 1 is a plateau, a ratio << 1 a descending (exact or border) path."""
import json
import sys
import time

sys.path.insert(0, '.')
import numpy as np

from tensorlab.numeric_restriction import _setup_pattern, border_path, coarse_pattern

MUS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10)
kind = sys.argv[1]
if kind == 'full':
    n, N, c, starts, seed0 = (int(v) for v in sys.argv[2:7])
    pat = _setup_pattern(2, n, N)[:3]
    name = f'full n={n} N={N} c={c}'
elif kind == 'coarse':
    N, c, starts, seed0 = (int(v) for v in sys.argv[2:6])
    pat = coarse_pattern(N)
    name = f'coarse N={N} c={c}'
else:
    n, c, starts, seed0 = (int(v) for v in sys.argv[2:6])
    keep = json.load(open(f'logs/minimal_box_n{n}.json'))['keep']
    rows, target, dims, _, _ = _setup_pattern(2, n, 2)
    mask = np.isin(rows[:, 0], keep[0]) & np.isin(rows[:, 1], keep[1]) & np.isin(rows[:, 2], keep[2])
    maps = [{v: i for i, v in enumerate(sorted(k))} for k in keep]
    rr = np.array([[maps[0][a], maps[1][b], maps[2][d]] for a, b, d in rows[mask]])
    pat = (rr, target[mask], tuple(len(k) for k in keep))
    N = 2
    name = f'box n={n} {pat[2]} c={c}'
t0 = time.time()
print(f'{name}: support {len(pat[1])}, starts {starts}', flush=True)
finals = []
for s in range(starts):
    out = border_path(2, 0, N, c, mus=MUS, starts=1, iters=250, seed=seed0 + s, pattern=pat)
    p = out['path']
    final, mid = p[-1][1], p[-3][1]
    finals.append(final)
    tag = 'DESCENT' if final < 0.5 * mid and final < 1e-3 else ''
    print(f'  seed {seed0 + s}: final {final:.3e}  tail ratio {final / max(mid, 1e-300):.2f}  norm {p[-1][2]:.1f} {tag} [{time.time() - t0:.0f}s]', flush=True)
finals = np.array(finals)
print(f'SUMMARY {name}: best {finals.min():.3e}, starts below 1e-3: {(finals < 1e-3).sum()}/{starts}, '
      f'plateau values {np.unique(np.round(finals[finals > 1e-3], 2))[:8]}', flush=True)
print('DONE', flush=True)
