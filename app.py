# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ===============================
# App Config
# ===============================
APP_TITLE = "StockXplore: Big Data-Powered VIKOR System"
APP_TAGLINE = "Interactive VIKOR ranking for smarter stock selection."

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")
st.title(APP_TITLE)
st.markdown(APP_TAGLINE)

# ===============================
# Helpers
# ===============================
def load_sample_data(n=10, seed=42):
    np.random.seed(seed)
    data = {
        "Ticker": [f"S{str(i+1).zfill(3)}" for i in range(n)],
        "Name": [f"Company {i+1}" for i in range(n)],
        "EPS":  np.round(np.random.uniform(0.5, 6.0, n), 2),
        "DPS":  np.round(np.random.uniform(0.0, 3.0, n), 2),
        "NTA":  np.round(np.random.uniform(0.5, 8.0, n), 2),
        "PE":   np.round(np.random.uniform(6.0, 40.0, n), 2),
        "DY":   np.round(np.random.uniform(0.0, 12.0, n), 2),
        "ROE":  np.round(np.random.uniform(0.0, 30.0, n), 2),
        "PTBV": np.round(np.random.uniform(0.5, 5.0, n), 2),
    }
    return pd.DataFrame(data)

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def normalize_for_vikor(X, benefit_flags):
    m, n = X.shape
    D = np.zeros_like(X, dtype=float)
    for j in range(n):
        col = X[:, j].astype(float)
        if benefit_flags[j]:
            f_star = np.nanmax(col)
            f_minus = np.nanmin(col)
            denom = f_star - f_minus
            D[:, j] = 0 if denom == 0 else (f_star - col) / denom
        else:
            f_star = np.nanmin(col)
            f_minus = np.nanmax(col)
            denom = f_minus - f_star
            D[:, j] = 0 if denom == 0 else (col - f_star) / denom
    return D

def vikor(df, id_col, criteria, weights_dict, benefit_dict, v=0.5):
    X = df[criteria].astype(float).values
    benefit_flags = [benefit_dict[c] for c in criteria]
    D = normalize_for_vikor(X, benefit_flags)
    w = np.array([weights_dict[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)
    S = (D * w).sum(axis=1)
    R = (D * w).max(axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)
    out = df[[id_col]].copy()
    out["VIKOR_S"] = S
    out["VIKOR_R"] = R
    out["VIKOR_Q"] = Q
    out["VIKOR_Rank"] = out["VIKOR_Q"].rank(method="min").astype(int)
    return out.sort_values("VIKOR_Q")

def chart_vikor(df, id_col):
    df_plot = df.copy()
    df_plot[id_col] = df_plot[id_col].astype(str)  # Ensure categorical
    df_plot = df_plot.sort_values("VIKOR_Q", ascending=True)
    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("VIKOR_Q:Q", title="VIKOR Q (lower is better)"),
            y=alt.Y(f"{id_col}:N", sort='-x', title=id_col),
            tooltip=[alt.Tooltip(id_col), "VIKOR_S:Q", "VIKOR_R:Q", "VIKOR_Q:Q", "VIKOR_Rank:Q"]
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

def download_csv(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name=filename, mime="text/csv")

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("📈 StockXplore • VIKOR")
    st.markdown("Upload dataset, select ID & criteria, set weights, and run VIKOR.")

# ===============================
# Main App
# ===============================
# Step 1 — Load Data
st.header("Step 1 — Load Data")
uploaded = st.file_uploader("Upload CSV (rows=alternatives, columns=criteria)", type=["csv"])
if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    st.info("Using sample dataset.")
    df_raw = load_sample_data()

st.dataframe(df_raw.head(20), use_container_width=True)

# Step 2 — Select ID & Criteria
all_cols = list(df_raw.columns)
id_col = st.selectbox("Select ID column", options=all_cols, index=0)
num_candidates = [c for c in all_cols if c != id_col]
df_num = coerce_numeric(df_raw.copy(), num_candidates)
numeric_cols = [c for c in num_candidates if pd.api.types.is_numeric_dtype(df_num[c])]
criteria = st.multiselect("Select numeric criteria", options=numeric_cols, default=numeric_cols)
if len(criteria) == 0:
    st.stop()

# Step 3 — Benefit/Cost
st.subheader("Step 2 — Mark Benefit or Cost")
benefit_flags = {}
for c in criteria:
    choice = st.radio(f"{c}", ["Benefit (higher is better)", "Cost (lower is better)"], index=0, key=f"bc_{c}")
    benefit_flags[c] = choice.startswith("Benefit")

# Step 4 — Weights
st.subheader("Step 3 — Set Weights")
weights = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        weights[c] = st.number_input(f"Weight: {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")

st.caption(f"Total weight (auto-normalized): **{sum(weights.values()):.2f}**")

# Step 5 — VIKOR Parameter
st.subheader("Step 4 — VIKOR Parameter")
v_param = st.slider("VIKOR v (strategy of majority)", 0.0, 1.0, 0.5, 0.05)

# Run VIKOR
st.markdown("---")
if st.button("🚀 Run VIKOR"):
    df_work = df_num[[id_col] + criteria].dropna()
    vikor_df = vikor(df_work, id_col, criteria, weights, benefit_flags, v=v_param)
    st.subheader("VIKOR Results")
    st.dataframe(vikor_df, use_container_width=True)
    chart_vikor(vikor_df, id_col)
    download_csv(vikor_df, "vikor_results.csv")
    st.success(f"🎯 Best Alternative: {vikor_df.iloc[0][id_col]}")
