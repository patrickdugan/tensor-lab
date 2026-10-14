import sys
sys.path.insert(0, '.')
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
from tensorlab.numeric_restriction import coarse_pattern, coarse_types
N, c = 3, 5
rows, target, (nx, ny, nz) = coarse_pattern(N)
th = np.load('logs/deep_N3_c5_theta.npy')
a = th[:c * nx].reshape(c, nx); d = th[c * nx:c * (nx + ny)].reshape(c, ny); e = th[c * (nx + ny):].reshape(c, nz)
V = a[:, rows[:, 0]] * d[:, rows[:, 1]] * e[:, rows[:, 2]]
types = coarse_types()
lab = [f"{t[0]}{t[1]}{t[2]}{'r' if t[3] else 'x'}" for t in types]
F = V[2] + V[3]
def unfold_ranks(v):
    T = v.reshape(12, 12, 12)
    out = []
    for ax in range(3):
        M = np.moveaxis(T, ax, 0).reshape(12, -1)
        s = np.linalg.svd(M, compute_uv=False)
        out.append(int((s > 1e-6 * s[0]).sum()))
    return out, s[:4]
for name, v in [('copy0', V[0]), ('copy1', V[1]), ('copy4', V[4]), ('pair2+3', F), ('target', target)]:
    rk, _ = unfold_ranks(v)
    print(f'{name}: factor-unfolding ranks {rk}, max|v| {np.abs(v).max():.3f}, #nonzero {(np.abs(v) > 1e-6).sum()}')
print('labels', lab)
for name, v in [('copy0', V[0]), ('copy1', V[1]), ('copy4', V[4])]:
    T = v.reshape(12, 12, 12)
    # if rank one across factors, print factor vectors
    U = [np.linalg.svd(np.moveaxis(T, ax, 0).reshape(12, -1))[0][:, 0] for ax in range(3)]
    print(name, 'leading factor vectors (normalised by max):')
    for ax in range(3):
        u = U[ax] / U[ax][np.argmax(np.abs(U[ax]))]
        print('   f', ax + 1, u)
