import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.config import DEFAULT_TICKERS, DEFAULT_N_SIMULATIONS, DEFAULT_CONFIDENCE, SCENARIOS
from src.data import get_return_data
from src.optimizer import run_cvar_optimization, GPU_AVAILABLE
from src.scenarios import run_scenario_cvar

st.title("Stress Scenario Overlay")

if GPU_AVAILABLE:
    st.success("🟢 GPU Mode active")
else:
    st.info("🟡 CPU Mode active")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
    start = st.date_input("Start date", value=pd.Timestamp("2010-01-01"))
    end = st.date_input("End date", value=pd.Timestamp("today"))
    n_sim = st.slider("Monte Carlo paths", 1_000, 20_000, DEFAULT_N_SIMULATIONS, step=1_000)
    conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x*100)}%")
    scenario_name = st.radio("Active scenario", list(SCENARIOS.keys()))
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

if not tickers or len(tickers) < 2:
    st.warning("Select at least 2 tickers.")
    st.stop()

if run_btn or "stress_results" not in st.session_state:
    with st.spinner("Running stress scenario analysis…"):
        returns_df = get_return_data(tuple(tickers), str(start), str(end))
        active_tickers = list(returns_df.columns)
        returns_np = returns_df.values

        opt = run_cvar_optimization(returns_np, n_sim, conf)
        scenario_results = run_scenario_cvar(returns_df, opt["weights"], n_sim, conf)

        st.session_state["stress_results"] = scenario_results
        st.session_state["stress_opt"] = opt
        st.session_state["stress_tickers"] = active_tickers
        st.session_state["stress_returns"] = returns_df

results = st.session_state.get("stress_results")
if results is None:
    st.info("Click **Run Analysis** to begin.")
    st.stop()

opt = st.session_state["stress_opt"]
active_tickers = st.session_state["stress_tickers"]
returns_df = st.session_state["stress_returns"]
sel = results.get(scenario_name, {})

full = results.get("Full History", {})
full_cvar = full.get("cvar", 0)
sel_cvar = sel.get("cvar", 0)

col1, col2, col3 = st.columns(3)
col1.metric("Full History CVaR", f"{full_cvar*100:.2f}%")
col2.metric(f"{scenario_name} CVaR", f"{sel_cvar*100:.2f}%",
            delta=f"{(sel_cvar - full_cvar)*100:+.2f}%", delta_color="inverse")
col3.metric("Worst Day (Scenario)", f"{sel.get('worst_day', 0)*100:.2f}%" if sel.get("worst_day") else "N/A")

st.caption(SCENARIOS[scenario_name]["description"])

# asset contribution bar chart
st.subheader(f"Asset Contribution to Loss — {scenario_name}")
cfg = SCENARIOS[scenario_name]
scen_start = cfg["start"]
scen_end = cfg.get("end")

mask = returns_df.index >= scen_start
if scen_end:
    mask &= returns_df.index <= scen_end
scen_slice = returns_df.loc[mask]

if not scen_slice.empty:
    weighted_contrib = scen_slice.values * opt["weights"]
    mean_contrib = weighted_contrib.mean(axis=0)

    contrib_df = pd.DataFrame({
        "Ticker": active_tickers,
        "Contribution": mean_contrib,
    }).sort_values("Contribution")

    fig_contrib = go.Figure(go.Bar(
        x=contrib_df["Ticker"],
        y=contrib_df["Contribution"] * 100,
        marker_color=["#e74c3c" if v < 0 else "#76b900" for v in contrib_df["Contribution"]],
    ))
    fig_contrib.update_layout(
        yaxis_title="Avg Daily Contribution (%)",
        template="plotly_dark",
        margin=dict(t=10, b=20),
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

st.subheader("All Scenarios — CVaR Comparison")
all_rows = []
for name, r in results.items():
    if "error" in r:
        all_rows.append({"Scenario": name, "CVaR": "N/A", "VaR": "N/A", "Mean Return": "N/A", "N Obs": "N/A"})
    else:
        all_rows.append({
            "Scenario": name,
            "CVaR": f"{r['cvar']*100:.2f}%",
            "VaR": f"{r['var']*100:.2f}%",
            "Mean Return": f"{r['mean_return']*100:.3f}%",
            "N Obs": r["n_obs"],
        })
st.dataframe(pd.DataFrame(all_rows).set_index("Scenario"), use_container_width=True)
