import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.config import DEFAULT_TICKERS, DEFAULT_N_SIMULATIONS, DEFAULT_CONFIDENCE
from src.data import get_return_data
from src.frontier import compute_frontier
from src.optimizer import run_cvar_optimization, GPU_AVAILABLE

st.title("Efficient Frontier")

if GPU_AVAILABLE:
    st.success("🟢 GPU Mode active")
else:
    st.info("🟡 CPU Mode active")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
    start = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end = st.date_input("End date", value=pd.Timestamp("today"))
    n_sim = st.slider("Monte Carlo paths", 1_000, 20_000, 5_000, step=1_000)
    conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x*100)}%")
    n_pts = st.slider("Frontier points", 20, 80, 40, step=5)
    run_btn = st.button("Compute Frontier", type="primary", use_container_width=True)

if not tickers or len(tickers) < 2:
    st.warning("Select at least 2 tickers.")
    st.stop()

if run_btn or "frontier_df" not in st.session_state:
    with st.spinner("Computing efficient frontier — this may take ~30s in CPU mode…"):
        returns_df = get_return_data(tuple(tickers), str(start), str(end))
        active_tickers = tuple(returns_df.columns)
        returns_np = returns_df.values

        frontier_df = compute_frontier(
            returns_tuple=tuple(map(tuple, returns_np)),
            tickers=active_tickers,
            n_simulations=n_sim,
            confidence=conf,
            n_points=n_pts,
        )
        opt = run_cvar_optimization(returns_np, n_sim, conf)
        mu = returns_np.mean(axis=0)

        st.session_state["frontier_df"] = frontier_df
        st.session_state["frontier_opt"] = opt
        st.session_state["frontier_tickers"] = active_tickers
        st.session_state["frontier_mu"] = mu

frontier_df = st.session_state.get("frontier_df")
if frontier_df is None:
    st.info("Configure settings and click **Compute Frontier**.")
    st.stop()

opt = st.session_state["frontier_opt"]
active_tickers = st.session_state["frontier_tickers"]
mu = st.session_state["frontier_mu"]

# build hover text showing weights for each frontier point
hover_texts = []
for _, row in frontier_df.iterrows():
    w = row["weights"]
    lines = [f"{t}: {w[t]*100:.1f}%" for t in active_tickers if t in w]
    hover_texts.append("<br>".join(lines))

eq_weights = np.ones(len(active_tickers)) / len(active_tickers)

eq_port_ret = (returns_df.values @ eq_weights)
eq_mean_ann = float(eq_port_ret.mean()) * 252

# quick CVaR for equal weight
cov = np.cov(returns_df.values.T)
sim_eq = np.random.multivariate_normal(mu, cov, size=n_sim)
eq_sim = sim_eq @ eq_weights
eq_cutoff = np.quantile(eq_sim, 1 - conf)
eq_tail = eq_sim[eq_sim <= eq_cutoff]
eq_cvar = float(-eq_tail.mean())

opt_mean_ann = float((returns_df.values @ opt["weights"]).mean()) * 252

fig = go.Figure()

fig.add_scatter(
    x=frontier_df["min_cvar"],
    y=frontier_df["mean_return"],
    mode="lines+markers",
    name="Efficient Frontier",
    line=dict(color="#76b900", width=2),
    marker=dict(size=6),
    text=hover_texts,
    hovertemplate="CVaR: %{x:.3f}<br>Return: %{y:.3f}<br>%{text}<extra></extra>",
)

fig.add_scatter(
    x=[opt["cvar"]],
    y=[opt_mean_ann],
    mode="markers+text",
    name="Optimal Portfolio",
    marker=dict(size=14, color="red", symbol="star"),
    text=["Optimal"],
    textposition="top right",
)

fig.add_scatter(
    x=[eq_cvar],
    y=[eq_mean_ann],
    mode="markers+text",
    name="Equal Weight",
    marker=dict(size=12, color="orange", symbol="diamond"),
    text=["Equal-Weight"],
    textposition="top left",
)

fig.update_layout(
    xaxis_title="CVaR (daily)",
    yaxis_title="Expected Annual Return",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=40),
    height=550,
)
st.plotly_chart(fig, use_container_width=True)
