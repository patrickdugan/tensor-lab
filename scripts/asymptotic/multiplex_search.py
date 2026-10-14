"""Structured search: R^{(x)3} = sum_h A_h (x) F_h  (one multiplexed rank-one term)
+ sum_h A_h (x) E_h (E_h rank 2), on the coarse support.  Unknowns on the
two-factor coarse support Phi^2 (factors 1, 2):
  F_h = a * b_h * g_h  (common a), with a b_0 g_1 = a b_1 g_0 = 0 on Phi^2,
  E_h = sum_{r<2} u_hr v_hr w_hr,  F_h + E_h = R (x) R on Phi^2.
A solution gives c_3 <= 5 for the coarse problem (hence for every n)."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from tensorlab.numeric_restriction import coarse_pattern

rows, target, (nx, ny, nz) = coarse_pattern(2)
m = len(rows)
# parameter layout
off = {}
P = 0
def alloc(name, size):
    global P
    off[name] = (P, size); P += size
alloc('a', nx)
for h in (0, 1):
    alloc(f'b{h}', ny); alloc(f'g{h}', nz)
    for r in range(2):
        alloc(f'u{h}{r}', nx); alloc(f'v{h}{r}', ny); alloc(f'w{h}{r}', nz)
def idx(name, i):
    return off[name][0] + i
monos = []  # (eq, i, j, k)
tgt = []
eq = 0
for t in range(m):
    x, y, z = rows[t]
    # orthogonality equations
    monos.append((eq, idx('a', x), idx('b0', y), idx('g1', z))); tgt.append(0.0); eq += 1
    monos.append((eq, idx('a', x), idx('b1', y), idx('g0', z))); tgt.append(0.0); eq += 1
    for h in (0, 1):
        monos.append((eq, idx('a', x), idx(f'b{h}', y), idx(f'g{h}', z)))
        for r in range(2):
            monos.append((eq, idx(f'u{h}{r}', x), idx(f'v{h}{r}', y), idx(f'w{h}{r}', z)))
        tgt.append(target[t]); eq += 1
monos = np.array(monos); tgt = np.array(tgt); neq = eq
E, I, J, K = monos.T

def resid(th):
    return np.bincount(E, weights=th[I] * th[J] * th[K], minlength=neq) - tgt

def jac(th):
    Jm = np.zeros((neq, P))
    np.add.at(Jm, (E, I), th[J] * th[K])
    np.add.at(Jm, (E, J), th[I] * th[K])
    np.add.at(Jm, (E, K), th[I] * th[J])
    return Jm

def run(seed, mus=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0.0), iters=400):
    rng = np.random.default_rng(seed)
    th = rng.standard_normal(P)
    path = []
    for mu in mus:
        lam = 1e-3
        r = resid(th); f = r @ r + mu * th @ th
        for it in range(iters):
            Jm = jac(th); g = Jm.T @ r + mu * th; H = Jm.T @ Jm + mu * np.eye(P)
            ok = False
            while lam < 1e14:
                th2 = th - np.linalg.solve(H + lam * np.eye(P), g)
                r2 = resid(th2); f2 = r2 @ r2 + mu * th2 @ th2
                if f2 < f:
                    th, r, f = th2, r2, f2; lam = max(lam / 3, 1e-15); ok = True; break
                lam *= 4
            if not ok or r @ r < 1e-26:
                break
        path.append((mu, float(r @ r), float(np.linalg.norm(th))))
    return path, th

if __name__ == '__main__':
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f'unknowns {P}, equations {neq}', flush=True)
    best = None
    for s in range(starts):
        t0 = time.time()
        path, th = run(1000 + s)
        print(f'start {s}: ' + ' '.join(f'{res:.1e}' for _, res, _ in path) + f' | norm {path[-1][2]:.1f} [{time.time()-t0:.0f}s]', flush=True)
        if best is None or path[-1][1] < best[0]:
            best = (path[-1][1], th)
            np.save('logs/multiplex_N3_theta.npy', th)
        if best[0] < 1e-24:
            break
    print('BEST', best[0], flush=True)
