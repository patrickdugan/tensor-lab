import sys
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import coarse_pattern
N, c, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rows, target, (nx, ny, nz) = coarse_pattern(N)
th = np.load(path)
a = th[:c * nx].reshape(c, nx); d = th[c * nx:c * (nx + ny)].reshape(c, ny); e = th[c * (nx + ny):].reshape(c, nz)
V = a[:, rows[:, 0]] * d[:, rows[:, 1]] * e[:, rows[:, 2]]
res = V.sum(0) - target
print('residual', float(res @ res), 'param norm', float(np.linalg.norm(th)))
print('copy support-norms', np.round(np.linalg.norm(V, axis=1), 3))
for i in range(c):
    for j in range(i + 1, c):
        s = np.linalg.norm(V[i] + V[j]); print(f'  |v{i}+v{j}| = {s:.3f}', end='')
    print()
# group sums: subsets whose sum is bounded
import itertools
for k in (2, 3):
    for S in itertools.combinations(range(c), k):
        print(f'  |sum{S}| = {np.linalg.norm(V[list(S)].sum(0)):.3f}', end='')
    print()
