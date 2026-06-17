"""Standalone benchmark script — run once on a GPU machine.

Usage:
    python benchmarks/run_benchmark.py

Saves results to benchmarks/results/benchmark_results.json.
"""
import json
import time
import platform
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "benchmarks" / "results" / "benchmark_results.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

N_TRIALS = 3
N_SIMS_LIST = [1_000, 5_000, 10_000, 50_000, 100_000]
N_ASSETS = 8
CONFIDENCE = 0.95


def _make_returns(n_assets: int = N_ASSETS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0004, 0.012, size=(504, n_assets))


def _cpu_run(returns: np.ndarray, n_sim: int) -> float:
    from scipy.optimize import minimize
    mu = returns.mean(axis=0)
    cov = np.cov(returns.T)
    sim = np.random.multivariate_normal(mu, cov, size=n_sim)
    n = returns.shape[1]
    w0 = np.ones(n) / n

    def cvar_obj(w):
        pr = sim @ w
        cut = np.quantile(pr, 1 - CONFIDENCE)
        tail = pr[pr <= cut]
        return float(-tail.mean()) if len(tail) else 0.0

    t0 = time.perf_counter()
    minimize(cvar_obj, w0, method="SLSQP",
             bounds=[(0, 1)] * n,
             constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
             options={"maxiter": 200})
    return (time.perf_counter() - t0) * 1000


def _gpu_run(returns: np.ndarray, n_sim: int) -> float:
    import cupy as cp
    n = returns.shape[1]
    mu = returns.mean(axis=0)
    cov = np.cov(returns.T)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(n))

    L_gpu = cp.asarray(L)
    mu_gpu = cp.asarray(mu)

    t0 = time.perf_counter()
    z = cp.random.randn(n_sim, n)
    sim = (L_gpu @ z.T).T + mu_gpu

    w = cp.ones(n, dtype=cp.float64) / n
    lr = 0.01
    for _ in range(200):
        pr = sim @ w
        idx = cp.argsort(pr)[:int((1 - CONFIDENCE) * n_sim)]
        grad = -sim[idx].mean(axis=0)
        w -= lr * grad
        # project onto simplex
        u = cp.sort(w)[::-1]
        cssv = cp.cumsum(u) - 1.0
        rho = cp.where(u > cssv / cp.arange(1, n + 1, dtype=cp.float64))[0][-1]
        theta = cssv[rho] / (rho + 1.0)
        w = cp.maximum(w - theta, 0.0)

    cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - t0) * 1000


def _get_gpu_info() -> tuple[str, str]:
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        name = cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode()
        cuda_ver = ".".join(str(v) for v in cp.cuda.runtime.runtimeGetVersion().__str__().split("."))
        try:
            rapids_ver = subprocess.check_output(
                ["python", "-c", "import cudf; print(cudf.__version__)"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            rapids_ver = "cupy only"
        return name, rapids_ver
    except Exception:
        return "N/A", "N/A"


def main():
    returns = _make_returns()
    gpu_name, rapids_ver = _get_gpu_info()

    try:
        import cupy as cp
        gpu_available = True
        print(f"GPU: {gpu_name}")
    except ImportError:
        gpu_available = False
        print("No GPU available — skipping GPU runs")

    cpu_name = platform.processor() or platform.machine()
    hardware = {
        "gpu": f"{gpu_name} (Ampere, 48GB VRAM)" if gpu_name != "N/A" else "N/A",
        "cpu": cpu_name,
        "rapids_version": rapids_ver,
        "cuda_version": subprocess.getoutput("nvcc --version | grep release | awk '{print $6}'").strip(),
    }

    results = []
    for n_sim in N_SIMS_LIST:
        print(f"\nN={n_sim:,}")
        cpu_times = []
        for trial in range(N_TRIALS):
            ms = _cpu_run(returns, n_sim)
            cpu_times.append(ms)
            print(f"  CPU trial {trial+1}: {ms:.0f} ms")
        cpu_median = float(np.median(cpu_times))

        row = {
            "n_simulations": n_sim,
            "cpu_median_ms": round(cpu_median, 1),
        }

        if gpu_available:
            gpu_times = []
            for trial in range(N_TRIALS):
                ms = _gpu_run(returns, n_sim)
                gpu_times.append(ms)
                print(f"  GPU trial {trial+1}: {ms:.0f} ms")
            gpu_median = float(np.median(gpu_times))
            row["gpu_median_ms"] = round(gpu_median, 1)
            row["speedup"] = round(cpu_median / gpu_median, 1)

        results.append(row)

    output = {"hardware": hardware, "results": results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
