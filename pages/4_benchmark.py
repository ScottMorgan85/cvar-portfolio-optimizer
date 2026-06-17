import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.title("GPU vs CPU Benchmark")

results_path = Path(__file__).parent.parent / "benchmarks" / "results" / "benchmark_results.json"

if not results_path.exists():
    st.error("benchmark_results.json not found. Run `python benchmarks/run_benchmark.py` on a GPU machine to generate it.")
    st.stop()

with open(results_path) as f:
    data = json.load(f)

hw = data.get("hardware", {})
results = data.get("results", [])

st.subheader("Hardware")
col1, col2 = st.columns(2)
col1.info(f"**GPU:** {hw.get('gpu', 'N/A')}\n\n**RAPIDS:** {hw.get('rapids_version', 'N/A')}\n\n**CUDA:** {hw.get('cuda_version', 'N/A')}")
col2.info(f"**CPU:** {hw.get('cpu', 'N/A')}")

st.caption(
    "Results captured on RTX A6000 (Ampere, 48GB VRAM). "
    "The original Regenstein (2026) article used an A10G (Ampere, 24GB VRAM) inside Snowflake Container Runtime — "
    "not a Blackwell GPU as the title suggests. Blackwell on Snowflake is listed as 'coming soon' in that article. "
    "This benchmark is A6000 vs A10G (Ampere vs Ampere, more VRAM vs less VRAM)."
)

st.info(
    "**Blackwell vs Ampere context:** Blackwell GPUs are ~2× faster than Ampere on FP16 tensor ops. "
    "CVaR Monte Carlo is dominated by FP64 matrix operations, so the speedup gap between Blackwell and Ampere "
    "is narrower in practice — typically 1.2–1.5× rather than 2×."
)

df = pd.DataFrame(results)
if df.empty:
    st.warning("No benchmark results found in JSON.")
    st.stop()

fig = go.Figure()
if "cpu_median_ms" in df.columns:
    fig.add_scatter(
        x=df["n_simulations"],
        y=df["cpu_median_ms"],
        mode="lines+markers",
        name="CPU (NumPy/SciPy)",
        line=dict(color="#4a90d9", width=2),
        marker=dict(size=8),
    )
if "gpu_median_ms" in df.columns:
    fig.add_scatter(
        x=df["n_simulations"],
        y=df["gpu_median_ms"],
        mode="lines+markers",
        name="GPU (RAPIDS/CuPy)",
        line=dict(color="#76b900", width=2),
        marker=dict(size=8),
    )

fig.update_layout(
    xaxis_title="Monte Carlo Paths (N)",
    yaxis_title="Median Runtime (ms)",
    yaxis_type="log",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=40),
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

if "speedup" in df.columns:
    st.subheader("Speedup Ratio (CPU / GPU)")
    fig2 = go.Figure(go.Bar(
        x=df["n_simulations"],
        y=df["speedup"],
        marker_color="#76b900",
        text=[f"{v:.1f}×" for v in df["speedup"]],
        textposition="outside",
    ))
    fig2.update_layout(
        xaxis_title="Monte Carlo Paths (N)",
        yaxis_title="Speedup (×)",
        template="plotly_dark",
        margin=dict(t=20, b=40),
        height=320,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Raw Results")
st.dataframe(df, use_container_width=True)
