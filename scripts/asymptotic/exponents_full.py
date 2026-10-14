"""eps-exponent structure of a deep_full.py run on the full block form.
Usage (from tensor-lab/): python scripts/asymptotic/exponents_full.py n N c seed"""
import glob
import sys
from collections import Counter

sys.path.insert(0, '.')
import numpy as np

from tensorlab.dense import block_form
from tensorlab.numeric_restriction import _setup_pattern

n, N, c, seed = (int(v) for v in sys.argv[1:5])
rows, target, (nx, ny, nz), terms, coords = _setup_pattern(2, n, N)
files = sorted(glob.glob(f'logs/deepfull_n{n}_N{N}_c{c}_s{seed}_*.npy'), key=lambda s: int(s.split('_')[-1].split('.')[0]))[1:]
def unpack(th):
    return th[:c * nx].reshape(c, nx), th[c * nx:c * (nx + ny)].reshape(c, ny), th[c * (nx + ny):].reshape(c, nz)
P = [unpack(np.load(f)) for f in files]
V = [a[:, rows[:, 0]] * d[:, rows[:, 1]] * e[:, rows[:, 2]] for a, d, e in P]
res = [float(((v.sum(0) - target) ** 2).sum()) for v in V]
big = int(np.argmax(np.linalg.norm(V[-1], axis=1)))
le = np.array([-np.log(np.linalg.norm(v[big])) for v in V])
A = np.vstack([le, np.ones_like(le)]).T
print(f'{len(V)} snapshots, residual {res[0]:.2e} -> {res[-1]:.2e}, log eps moved {le[-1] - le[0]:.3f}, '
      f'residual exponent {np.polyfit(le, np.log(res), 1)[0]:.2f}')
S = np.array([np.linalg.lstsq(A, np.log(np.abs(np.array([v[r] for v in V])) + 1e-300), rcond=None)[0][0] for r in range(c)])
S1, deltas = block_form(2, n)
sup1 = sorted(S1.terms)
X1, Y1, Z1 = S1.dims
def split(t):
    x, y, z = rows[t]
    xs, ys, zs = [], [], []
    for _ in range(N):
        xs.append(x % X1); x //= X1; ys.append(y % Y1); y //= Y1; zs.append(z % Z1); z //= Z1
    return list(zip(xs[::-1], ys[::-1], zs[::-1]))
def lab(t1):
    x, j, k = t1
    d = deltas[(x, j, k)]
    return f"{x}{j}{k}{'+' if d > 0 else '-' if d < 0 else ''}"
for r in range(c):
    print(f'copy {r}: |support| {np.linalg.norm(V[-1][r]):.1f}; exponent counts (1/2):',
          sorted(Counter(np.round(S[r] * 2) / 2).items())[:10])
special = np.where(np.all(np.round(S * 2) / 2 <= -0.5, axis=0))[0]
print(f'entries where every copy diverges: {len(special)} (retained {int(target[special].sum())})')
neg = {r: np.where(np.round(S[r] * 2) / 2 < 0)[0] for r in range(c)}
for r in range(c):
    cnt = Counter(tuple(lab(t) for t in split(i)) for i in neg[r])
    print(f'  copy {r} diverging entries {len(neg[r])}:', list(cnt.items())[:12])

# x-pair and retained/cross pattern of diverging entries, by exponent class
for cls in (-1.0, -0.5):
    idx = np.where(np.round(S[0] * 2) / 2 == cls)[0]
    cnt = Counter()
    for i in idx:
        f = split(i)
        cnt[(tuple(t[0] for t in f), 'R' if target[i] else 'X')] += 1
    print(f'class {cls}: ' + ', '.join(f'x={k[0]} {k[1]}:{v}' for k, v in sorted(cnt.items())))
# total value of the pair on diverging entries vs target there
tot = V[-1].sum(0)
idx = np.where(np.round(S[0] * 2) / 2 < 0)[0]
print('max |sum - target| on diverging entries', float(np.abs(tot[idx] - target[idx]).max()),
      '; elsewhere', float(np.abs(np.delete(tot - target, idx)).max()))
