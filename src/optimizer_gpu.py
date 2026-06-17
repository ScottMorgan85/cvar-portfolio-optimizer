import time
import numpy as np

try:
    import cupy as cp
    cp.cuda.Device(0).use()
    _CUPY_OK = True
except Exception as _e:
    _CUPY_OK = False
    _cupy_err = str(_e)


def _simplex_project(v: "cp.ndarray") -> "cp.ndarray":
    """Project v onto the probability simplex (sum=1, all>=0)."""
    n = len(v)
    u = cp.sort(v)[::-1]
    cssv = cp.cumsum(u) - 1.0
    rho = cp.where(u > cssv / cp.arange(1, n + 1, dtype=cp.float64))[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return cp.maximum(v - theta, 0.0)


def run_cvar_optimization(
    returns: np.ndarray,
    n_simulations: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    if not _CUPY_OK:
        raise RuntimeError(f"CuPy not available: {_cupy_err}")

    t0 = time.perf_counter()
    n_assets = returns.shape[1]

    returns_gpu = cp.asarray(returns)
    mu = returns_gpu.mean(axis=0)
    cov = cp.cov(returns_gpu.T)

    L = cp.linalg.cholesky(cov + 1e-8 * cp.eye(n_assets))
    z = cp.random.randn(n_simulations, n_assets)
    sim = (L @ z.T).T + mu

    # projected gradient descent on the simplex
    w = cp.ones(n_assets, dtype=cp.float64) / n_assets
    lr = 0.01
    n_iter = 300
    cutoff_idx = int((1 - confidence) * n_simulations)

    for _ in range(n_iter):
        port_ret = sim @ w
        sorted_idx = cp.argsort(port_ret)
        tail_idx = sorted_idx[:cutoff_idx]
        tail_ret = port_ret[tail_idx]
        cvar_val = -tail_ret.mean()

        # gradient of CVaR w.r.t. w
        grad = -sim[tail_idx].mean(axis=0)
        w = w - lr * grad
        w = _simplex_project(w)

    port_ret_np = (sim @ w).get()
    weights_np = w.get()

    cutoff = np.quantile(port_ret_np, 1 - confidence)
    tail = port_ret_np[port_ret_np <= cutoff]
    cvar = float(-tail.mean())
    var = float(-cutoff)

    return {
        "weights": weights_np,
        "cvar": cvar,
        "var": var,
        "simulations": port_ret_np,
        "runtime_seconds": time.perf_counter() - t0,
        "mode": "GPU",
    }
