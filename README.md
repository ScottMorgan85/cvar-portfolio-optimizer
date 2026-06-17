# CVaR Portfolio Optimizer — GPU-Accelerated Risk Optimization

**Extends Regenstein (2026) — adds efficient frontier, stress scenarios, and local A6000 vs A10G benchmark**

A four-page Streamlit app that minimizes Conditional Value-at-Risk (CVaR) for multi-asset portfolios using Monte Carlo simulation, with an automatic CPU/GPU backend router. Built as an extension of the Regenstein (2026) article:

> Regenstein, J. (2026). *Accelerating Portfolio Optimization on the Snowflake AI Data Cloud with NVIDIA Blackwell Compute.*
> [Read on Medium](https://medium.com/snowflake/accelerating-portfolio-optimization-on-the-snowflake-ai-data-cloud-with-nvidia-blackwell-compute-66f733765b92)

---

## Hardware note

The Regenstein (2026) article title references "Blackwell" compute, but the benchmark hardware used in the article is an **NVIDIA A10G (Ampere, 24GB VRAM)** inside Snowflake Container Runtime. The article notes that Blackwell on Snowflake is "coming soon."

This repo benchmarks an **NVIDIA RTX A6000 (Ampere, 48GB VRAM)** locally against the CPU baseline, making this an Ampere vs Ampere comparison with more VRAM rather than a Blackwell vs Ampere comparison.

**Blackwell context:** Blackwell GPUs achieve roughly 2× the throughput of Ampere on FP16 tensor operations. CVaR Monte Carlo optimization is dominated by FP64 matrix ops (Cholesky decomposition, large matrix multiplications), so the real-world speedup gap between Blackwell and Ampere is narrower — typically 1.2–1.5× rather than 2× for this workload.

---

## Quick start — CPU mode

No GPU required. Works on any machine with Python 3.11+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs at `http://localhost:8501`.

---

## GPU mode setup

Requires CUDA 12.x and a compatible NVIDIA GPU.

```bash
# Install CuPy for CUDA 12
pip install -r requirements-gpu.txt

# Optional: install full RAPIDS suite (cuDF, cuML, cuOpt)
# See https://docs.rapids.ai/install for conda/pip instructions
# Version-pin to your exact CUDA version (e.g. cudf-cu12==24.x.x)

streamlit run app.py
```

The sidebar will show **GPU Mode (RAPIDS)** when CuPy initializes successfully. If CuPy is unavailable or the GPU is not found, the app falls back to CPU mode automatically — no code changes needed.

---

## Running the benchmark

Run this once on a machine with a GPU to generate `benchmarks/results/benchmark_results.json`:

```bash
python benchmarks/run_benchmark.py
```

The script runs 3 trials each of CPU (NumPy/SciPy SLSQP) and GPU (CuPy projected gradient) at 5 simulation counts: 1K, 5K, 10K, 50K, 100K paths. Results are saved as JSON and displayed on the **GPU Benchmark** page in the app.

---

## Architecture — CPU/GPU router

```
src/optimizer.py          ← public API: run_cvar_optimization(), GPU_AVAILABLE
├── src/optimizer_cpu.py  ← NumPy + SciPy SLSQP (always available)
└── src/optimizer_gpu.py  ← CuPy projected gradient (requires CUDA)
```

`src/optimizer.py` tries to import CuPy at module load time. If successful it re-exports the GPU implementation; otherwise it silently falls back to the CPU implementation. All four pages import only from `src/optimizer` — no page-level GPU checks needed.

---

## Data

All price data is fetched from **Yahoo Finance via yfinance** — no API key required. Prices are cached as Parquet files in `data/` (keyed by ticker list + date range hash) to avoid redundant downloads within a session.

Default tickers: `SPY, QQQ, IEF, GLD, VNQ, EEM, HYG, LQD`

Returns are computed as log returns: `ln(P_t / P_{t-1})`.

---

## Deployment — DigitalOcean App Platform

1. Push this repo to GitHub.
2. Create a new App on DigitalOcean App Platform, pointing to the repo.
3. Set the **Run Command** to:
   ```
   streamlit run app.py --server.port=8080 --server.headless=true
   ```
4. No environment variables required for CPU mode.
5. The `Dockerfile` is provided for container-based deployments.

---

## Pages

| # | Page | Description |
|---|------|-------------|
| 1 | **Portfolio Optimizer** | Run CVaR minimization; view optimal vs equal-weight allocation, simulated return distribution with VaR/CVaR lines |
| 2 | **Efficient Frontier** | Compute the CVaR-efficient frontier across a return target grid; plot optimal and equal-weight portfolios on the frontier |
| 3 | **Stress Scenarios** | Overlay optimal weights against historical stress periods (COVID Crash, 2022 Rates Shock, GFC Peak); per-asset contribution chart |
| 4 | **GPU Benchmark** | Display runtime comparison from `benchmark_results.json`; log-scale runtime chart and speedup bar chart |

---

## Benchmark results (placeholder — A6000 vs CPU baseline)

| Monte Carlo Paths | CPU Median (ms) | GPU Median (ms) | Speedup |
|------------------:|----------------:|----------------:|--------:|
| 1,000             | 120             | 45              | 2.7×    |
| 5,000             | 480             | 62              | 7.7×    |
| 10,000            | 950             | 85              | 11.2×   |
| 50,000            | 4,800           | 280             | 17.1×   |
| 100,000           | 9,600           | 520             | 18.5×   |

Run `python benchmarks/run_benchmark.py` on your GPU machine to replace these with real measurements.
