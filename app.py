import streamlit as st

st.set_page_config(
    page_title="CVaR Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.optimizer import GPU_AVAILABLE

pages = [
    st.Page("pages/1_optimizer.py", title="Portfolio Optimizer", icon="⚙️"),
    st.Page("pages/2_frontier.py", title="Efficient Frontier", icon="📈"),
    st.Page("pages/3_stress.py", title="Stress Scenarios", icon="🔥"),
    st.Page("pages/4_benchmark.py", title="GPU Benchmark", icon="⚡"),
]

pg = st.navigation(pages)

with st.sidebar:
    if GPU_AVAILABLE:
        st.success("🟢 GPU Mode (RAPIDS)")
    else:
        st.warning("🟡 CPU Mode (NumPy/SciPy)")

pg.run()
