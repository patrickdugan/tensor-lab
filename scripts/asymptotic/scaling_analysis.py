"""Estimate eps-exponents of each copy's values on the support from the snapshots
written by snapshot_run.py (V_r(tau) ~ eps^{e_r(tau)}, eps ~ 1/|v_2|)."""
import sys, glob
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import coarse_pattern, coarse_types
rows, target, (nx, ny, nz) = coarse_pattern(3)
c = 5
snaps = sorted(glob.glob('logs/snap_N3_c5_*.npy'), key=lambda s: int(s.split('_')[-1].split('.')[0]))
def vals(th):
    a = th[:c * nx].reshape(c, nx); d = th[c * nx:c * (nx + ny)].reshape(c, ny); e = th[c * (nx + ny):].reshape(c, nz)
    return a[:, rows[:, 0]] * d[:, rows[:, 1]] * e[:, rows[:, 2]]
V0, V1 = vals(np.load(snaps[0])), vals(np.load(snaps[-1]))
big = int(np.argmax(np.linalg.norm(V1, axis=1)))
leps0, leps1 = -np.log(np.linalg.norm(V0[big])), -np.log(np.linalg.norm(V1[big]))
dl = leps1 - leps0
print(f'{len(snaps)} snapshots; diverging copy {big}; log eps moved by {dl:.3f} (support norm {np.linalg.norm(V0[big]):.0f} -> {np.linalg.norm(V1[big]):.0f})')
res0 = V0.sum(0) - target; res1 = V1.sum(0) - target
print(f'residual {res0 @ res0:.3e} -> {res1 @ res1:.3e}; residual exponent {np.log((res1 @ res1) / (res0 @ res0)) / dl:.2f} (in eps)')
for r in range(c):
    with np.errstate(divide='ignore', invalid='ignore'):
        ex = (np.log(np.abs(V1[r]) + 1e-300) - np.log(np.abs(V0[r]) + 1e-300)) / dl
    mag = np.abs(V1[r])
    sig = mag > 1e-6 * mag.max()
    q = np.round(ex[sig] * 4) / 4
    vals_, counts = np.unique(q, return_counts=True)
    top = sorted(zip(counts, vals_), reverse=True)[:6]
    print(f'copy {r}: |support|={np.linalg.norm(V1[r]):.1f}, significant entries {sig.sum()}, exponent histogram (rounded to 1/4): ' +
          ', '.join(f'{v:+.2f}x{n}' for n, v in top))


# ---- regression over all snapshots (skips the first, pre-asymptotic one) ----
Vs = [vals(np.load(f)) for f in snaps[1:]]
if len(Vs) >= 3:
    big = int(np.argmax(np.linalg.norm(Vs[-1], axis=1)))
    le = np.array([-np.log(np.linalg.norm(V[big])) for V in Vs])   # log eps (up to a constant)
    print(f'regression over {len(Vs)} snapshots, log eps range {le.min():.2f}..{le.max():.2f}')
    for r in range(c):
        L = np.array([np.log(np.abs(V[r]) + 1e-300) for V in Vs])     # (snap, terms)
        A = np.vstack([le, np.ones_like(le)]).T
        coef, *_ = np.linalg.lstsq(A, L, rcond=None)
        slope = coef[0]
        mag = np.abs(Vs[-1][r]); sig = mag > 1e-6 * mag.max()
        q = np.round(slope[sig] * 2) / 2
        vv, cc = np.unique(q, return_counts=True)
        top = sorted(zip(cc, vv), reverse=True)[:6]
        print(f'  copy {r}: exponent histogram (rounded to 1/2) ' + ', '.join(f'{v:+.1f}x{n}' for n, v in top))
