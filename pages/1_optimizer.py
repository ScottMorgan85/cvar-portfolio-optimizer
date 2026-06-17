import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import DEFAULT_TICKERS, DEFAULT_CONFIDENCE, DEFAULT_N_SIMULATIONS
from src.data import get_return_data
from src.optimizer import run_cvar_optimization, GPU_AVAILABLE

st.title("Portfolio CVaR Optimizer")

if GPU_AVAILABLE:
    st.success("🟢 GPU Mode active — RAPIDS/CuPy backend")
else:
    st.info("🟡 CPU Mode active — NumPy/SciPy backend")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
    start = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end = st.date_input("End date", value=pd.Timestamp("today"))
    n_sim = st.slider("Monte Carlo paths", 1_000, 50_000, DEFAULT_N_SIMULATIONS, step=1_000)
    conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x*100)}%")
    run_btn = st.button("Run Optimizer", type="primary", use_container_width=True)

if not tickers:
    st.warning("Select at least 2 tickers.")
    st.stop()

if run_btn or "opt_result" not in st.session_state:
    with st.spinner("Fetching data and running optimization…"):
        returns_df = get_return_data(tuple(tickers), str(start), str(end))
        if returns_df.empty or len(returns_df.columns) < 2:
            st.error("Not enough return data. Try different tickers or date range.")
            st.stop()

        active_tickers = list(returns_df.columns)
        returns_np = returns_df.values

        opt = run_cvar_optimization(returns_np, n_sim, conf)
        eq_weights = np.ones(len(active_tickers)) / len(active_tickers)

        # equal weight CVaR comparison
        from src.optimizer_cpu import run_cvar_optimization as cpu_opt
        import time
        mu = returns_np.mean(axis=0)
        cov = np.cov(returns_np.T)
        sim = np.random.multivariate_normal(mu, cov, size=n_sim)
        eq_port = sim @ eq_weights
        eq_cutoff = np.quantile(eq_port, 1 - conf)
        eq_tail = eq_port[eq_port <= eq_cutoff]
        eq_cvar = float(-eq_tail.mean())

        st.session_state["opt_result"] = opt
        st.session_state["eq_cvar"] = eq_cvar
        st.session_state["active_tickers"] = active_tickers
        st.session_state["eq_weights"] = eq_weights
        st.session_state["sim_returns"] = opt["simulations"]

result = st.session_state.get("opt_result")
if result is None:
    st.info("Configure settings and click **Run Optimizer**.")
    st.stop()

active_tickers = st.session_state["active_tickers"]
eq_cvar = st.session_state["eq_cvar"]
eq_weights = st.session_state["eq_weights"]
sim_returns = st.session_state["sim_returns"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("CVaR (Optimal)", f"{result['cvar']*100:.2f}%", f"{(result['cvar'] - eq_cvar)*100:+.2f}% vs EW")
col2.metric("VaR", f"{result['var']*100:.2f}%")
col3.metric("Runtime", f"{result['runtime_seconds']*1000:.0f} ms")
col4.metric("Backend", result["mode"])

st.subheader("Optimal vs Equal-Weight Allocation")
wt_df = pd.DataFrame({
    "Ticker": active_tickers,
    "Optimal": result["weights"] * 100,
    "Equal-Weight": eq_weights * 100,
})

fig_bar = go.Figure()
fig_bar.add_bar(x=wt_df["Ticker"], y=wt_df["Optimal"], name="Optimal", marker_color="#76b900")
fig_bar.add_bar(x=wt_df["Ticker"], y=wt_df["Equal-Weight"], name="Equal-Weight", marker_color="#4a90d9")
fig_bar.update_layout(
    barmode="group", yaxis_title="Weight (%)", template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(
    wt_df.set_index("Ticker").style.format("{:.1f}%"),
    use_container_width=True,
)

st.subheader("Simulated Return Distribution")
var_val = -result["var"]
cvar_val = -result["cvar"]

fig_hist = go.Figure()
fig_hist.add_histogram(x=sim_returns, nbinsx=80, name="Simulated Returns", marker_color="#4a90d9", opacity=0.8)
fig_hist.add_vline(x=var_val, line_dash="dash", line_color="orange", annotation_text=f"VaR {result['var']*100:.1f}%", annotation_position="top right")
fig_hist.add_vline(x=cvar_val, line_dash="dash", line_color="red", annotation_text=f"CVaR {result['cvar']*100:.1f}%", annotation_position="top left")
fig_hist.update_layout(
    xaxis_title="Daily Return",
    yaxis_title="Count",
    template="plotly_dark",
    showlegend=False,
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_hist, use_container_width=True)
