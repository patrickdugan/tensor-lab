import sys, time
sys.path.insert(0, '.')
from tensorlab.numeric_restriction import copy_cost_continuation
n, N, c, starts, iters, seed = (int(v) for v in sys.argv[1:7])
mus = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0.0)
t0 = time.time()
print(f"full n={n} N={N} c={c} starts={starts} iters={iters} seed={seed}", flush=True)
r = copy_cost_continuation(2, n, N, c, starts=starts, iters=iters, mus=mus, seed=seed, verbose=True)
print("BEST", " ".join(f"{res:.2e}" for _, res, _ in r["path"]), f"| norm {r['path'][-1][2]:.2f}",
      f"[{time.time() - t0:.0f}s]", flush=True)
print("DONE", flush=True)
