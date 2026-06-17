import time
import numpy as np
from scipy.optimize import minimize


def run_cvar_optimization(
    returns: np.ndarray,
    n_simulations: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """Minimize CVaR via Monte Carlo + SLSQP.

    Args:
        returns: (T, N) array of historical log returns
        n_simulations: number of Monte Carlo paths
        confidence: tail confidence level (e.g. 0.95)

    Returns:
        dict with keys: weights, cvar, var, simulations, runtime_seconds, mode
    """
    t0 = time.perf_counter()
    n_assets = returns.shape[1]
    mu = returns.mean(axis=0)
    cov = np.cov(returns.T)

    sim = np.random.multivariate_normal(mu, cov, size=n_simulations)

    def portfolio_cvar(w):
        port_ret = sim @ w
        cutoff = np.quantile(port_ret, 1 - confidence)
        tail = port_ret[port_ret <= cutoff]
        return -tail.mean() if len(tail) > 0 else 0.0

    w0 = np.ones(n_assets) / n_assets
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n_assets

    result = minimize(
        portfolio_cvar,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    weights = result.x
    port_ret = sim @ weights
    cutoff = np.quantile(port_ret, 1 - confidence)
    tail = port_ret[port_ret <= cutoff]
    cvar = float(-tail.mean())
    var = float(-cutoff)

    return {
        "weights": weights,
        "cvar": cvar,
        "var": var,
        "simulations": port_ret,
        "runtime_seconds": time.perf_counter() - t0,
        "mode": "CPU",
    }
