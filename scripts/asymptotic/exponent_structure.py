"""Per-copy eps-exponents on the support from the snapshots, the special
(cancelling) entries of the diverging pair, and an additivity test of each
exponent function (u(x) + v(y) + w(z))."""
import sys, glob
sys.path.insert(0, '.')
import numpy as np
from collections import Counter
from tensorlab.numeric_restriction import coarse_pattern, coarse_types
rows, target, (nx, ny, nz) = coarse_pattern(3)
c = 5
types = coarse_types()
tindex = {(t[0], t[1], t[2]): i for i, t in enumerate(types)}
snaps = sorted(glob.glob('logs/snap_N3_c5_*.npy'), key=lambda s: int(s.split('_')[-1].split('.')[0]))[1:]
def unpack(th):
    return th[:c * nx].reshape(c, nx), th[c * nx:c * (nx + ny)].reshape(c, ny), th[c * (nx + ny):].reshape(c, nz)
Vs, P = [], []
for f in snaps:
    a, d, e = unpack(np.load(f)); P.append((a, d, e))
    Vs.append(a[:, rows[:, 0]] * d[:, rows[:, 1]] * e[:, rows[:, 2]])
big = int(np.argmax(np.linalg.norm(Vs[-1], axis=1)))
le = np.array([-np.log(np.linalg.norm(V[big])) for V in Vs]); le -= le[-1]
A = np.vstack([le, np.ones_like(le)]).T
def slopes(series):
    coef, *_ = np.linalg.lstsq(A, series, rcond=None)
    return coef[0]
def decode(idx):
    x, y, z = rows[idx]
    xs = [(x // 9) % 3, (x // 3) % 3, x % 3]; ys = [(y // 16) % 4, (y // 4) % 4, y % 4]; zs = [(z // 16) % 4, (z // 4) % 4, z % 4]
    return tuple(tindex[(xs[i], ys[i], zs[i])] for i in range(3))
lab = [f"{t[0]}{t[1]}{t[2]}{'r' if t[3] else 'x'}" for t in types]
S = np.array([slopes(np.log(np.abs(np.array([V[r] for V in Vs])) + 1e-300)) for r in range(c)])  # (c, terms)
for r in range(c):
    print(f'copy {r}: slope counts (rounded 1/2):', sorted(Counter(np.round(S[r] * 2) / 2).items())[:12])
special = np.where((np.round(S[2] * 2) / 2 == -1) & (np.round(S[3] * 2) / 2 == -1))[0]
print(f'\nspecial entries (both diverging copies ~ 1/eps): {len(special)}; retained among them: {int(target[special].sum())}')
cnt = Counter()
for idx in special:
    cnt[tuple(sorted(lab[k] for k in decode(idx)))] += 1
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:12]:
    print('   ', v, k)
# per-factor type sets of special entries
for f in range(3):
    print(f'  factor {f + 1} types in special entries:', sorted(Counter(lab[decode(i)[f]] for i in special).items()))
# additivity of each copy's exponent function on entries with moderate slopes
print()
for r in range(c):
    s = S[r]
    ok = np.abs(s) < 3
    m = ok.sum()
    M = np.zeros((m, nx + ny + nz)); idx = np.where(ok)[0]
    M[np.arange(m), rows[idx, 0]] = 1; M[np.arange(m), nx + rows[idx, 1]] = 1; M[np.arange(m), nx + ny + rows[idx, 2]] = 1
    sol, *_ = np.linalg.lstsq(M, s[idx], rcond=None)
    res = M @ sol - s[idx]
    print(f'copy {r}: additive fit of exponents on {m} entries: max error {np.abs(res).max():.3f}, rms {np.sqrt((res**2).mean()):.3f}')
