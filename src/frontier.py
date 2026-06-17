import numpy as np
import pandas as pd
import streamlit as st
from .optimizer import GPU_AVAILABLE


def _cvar_for_weights(weights: np.ndarray, sim: np.ndarray, confidence: float) -> float:
    port_ret = sim @ weights
    cutoff = np.quantile(port_ret, 1 - confidence)
    tail = port_ret[port_ret <= cutoff]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def compute_frontier(
    returns_tuple: tuple,
    tickers: tuple[str, ...],
    n_simulations: int = 10_000,
    confidence: float = 0.95,
    n_points: int = 50,
) -> pd.DataFrame:
    import scipy.optimize as sco

    returns = np.array(returns_tuple)
    n_assets = returns.shape[1]
    mu = returns.mean(axis=0)
    cov = np.cov(returns.T)
    sim = np.random.multivariate_normal(mu, cov, size=n_simulations)

    min_ret = mu.min() * 252
    max_ret = mu.max() * 252
    target_returns = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for target in target_returns:
        daily_target = target / 252

        def objective(w):
            return _cvar_for_weights(w, sim, confidence)

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w, t=daily_target: (sim @ w).mean() - t},
        ]
        bounds = [(0.0, 1.0)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        try:
            res = sco.minimize(
                objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-7},
            )
            w = res.x
        except Exception:
            w = w0

        port_ret = sim @ w
        mean_ret = float(port_ret.mean()) * 252
        cvar = _cvar_for_weights(w, sim, confidence)
        sharpe = mean_ret / (cvar + 1e-9)

        rows.append({
            "target_return": target,
            "mean_return": mean_ret,
            "min_cvar": cvar,
            "sharpe_proxy": sharpe,
            "weights": {tickers[i]: float(w[i]) for i in range(n_assets)},
        })

    return pd.DataFrame(rows)
