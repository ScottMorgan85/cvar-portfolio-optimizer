import time
import numpy as np
import pandas as pd
from .config import SCENARIOS


def run_scenario_cvar(
    full_returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """Run CVaR for all defined scenarios using historical return slices."""
    results = {}

    for name, cfg in SCENARIOS.items():
        t0 = time.perf_counter()
        start = cfg["start"]
        end = cfg["end"]

        mask = full_returns.index >= start
        if end:
            mask &= full_returns.index <= end

        slice_df = full_returns.loc[mask]
        if len(slice_df) < 10:
            results[name] = {"error": "insufficient data", "description": cfg["description"]}
            continue

        mu = slice_df.values.mean(axis=0)
        cov = np.cov(slice_df.values.T)
        sim = np.random.multivariate_normal(mu, cov, size=n_simulations)

        port_ret = sim @ weights
        cutoff = np.quantile(port_ret, 1 - confidence)
        tail = port_ret[port_ret <= cutoff]

        cvar = float(-tail.mean()) if len(tail) > 0 else 0.0
        var = float(-cutoff)
        mean_ret = float(port_ret.mean())
        worst_day = float((slice_df.values @ weights).min())

        results[name] = {
            "cvar": cvar,
            "var": var,
            "mean_return": mean_ret,
            "worst_day": worst_day,
            "n_obs": len(slice_df),
            "runtime_seconds": time.perf_counter() - t0,
            "description": cfg["description"],
        }

    return results
