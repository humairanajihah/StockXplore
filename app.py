# ===============================
# StockXplore VIKOR App (Fixed)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -----------------------------
# App Config
# -----------------------------
APP_TITLE = "StockXplore: Big Data-Powered VIKOR System"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

st.title(APP_TITLE)
st.markdown("This app applies **VIKOR (MCDM)** to rank alternatives based on multiple criteria.")

# -----------------------------
# Sample Data Function
# -----------------------------
def load_sample_data(n=10, seed=42):
    np.random.seed(seed)
    data = {
        "EPS":  np.round(np.random.uniform(0.5, 6.0, n), 2),
        "DPS":  np.round(np.random.uniform(0.0, 3.0, n), 2),
        "NTA":  np.round(np.random.uniform(0.5, 8.0, n), 2),
        "PE":   np.round(np.random.uniform(6.0, 40.0, n), 2),
        "DY":   np.round(np.random.uniform(0.0, 12.0, n), 2),
        "ROE":  np.round(np.random.uniform(0.0, 30.0, n), 2),
        "PTBV": np.round(np.random.uniform(0.5, 5.0, n), 2),
    }
    return pd.DataFrame(data)

# -----------------------------
# VIKOR Functions
# -----------------------------
def normalize_for_vikor(X, benefit_flags):
    m, n = X.shape
    D = np.zeros_like(X, dtype=float)
    f_star = np.zeros(n)
    f_minus = np.zeros(n)

    for j in range(n):
        col = X[:, j].astype(float)
        if benefit_flags[j]:
            f_star[j] = np.nanmax(col)
            f_minus[j] = np.nanmin(col)
            denom = f_star[j] - f_minus[j]
            D[:, j] = (f_star[j] - col) / denom if denom != 0 else 0
        else:
            f_star[j] = np.nanmin(col)
            f_minus[j] = np.nanmax(col)
            denom = f_minus[j] - f_star[j]
            D[:, j] = (col - f_star[j]) / denom if denom != 0 else 0
    return D, f_star, f_minus

def vikor(df, criteria, weights_dict, benefit_dict, v=0.5):
    X = df[criteria].astype(float).values
    benefit_flags = [bool(benefit_dict[c]) for c in criteria]
    D, f_star, f_minus = normalize_for_vikor(X, benefit_flags)

    w = np.array([float(weights_dict[c]) for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)

    S = (D * w).sum(axis=1)
    R = (D * w).max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

    out = df.copy()
    out["VIKOR_S"] = S
    out["VIKOR_R"] = R
    out["VIKOR_Q"] = Q
    out["VIKOR_Rank"] = Q.argsort() + 1
    return out.sort_values("VIKOR_Q")

# -----------------------------
# Chart Function (No ID)
# -----------------------------
def chart_vikor_with_rank(df):
    df_plot = df.copy().reset_index(drop=True)
    df_plot["Rank"] = df_plot["VIKOR_Rank"]

    # VIKOR Q Bar Chart
    chart_q = alt.Chart(df_plot).mark_bar(color='skyblue').encode(
        x=alt.X("VIKOR_Q:Q", title="VIKOR Q (lower is better)"),
        y=alt.Y("Rank:O", sort='ascending', title="Rank"),
        tooltip=["VIKOR_S:Q", "VIKOR_R:Q", "VIKOR_Q:Q", "VIKOR_Rank:Q"]
    ).properties(height=400, title="VIKOR Q Value by Rank")
    st.altair_chart(chart_q, use_container_width=True)

    # Ranking Bar Chart
    chart_rank = alt.Chart(df_plot).mark_bar(color='orange').encode(
        x=alt.X("Rank:O", sort='ascending', title="Rank"),
        y=alt.Y("VIKOR_Q:Q", title="VIKOR Q Value"),
        tooltip=["VIKOR_S:Q", "VIKOR_R:Q", "VIKOR_Q:Q", "VIKOR_Rank:Q"]
    ).properties(height=400, title="Ranking of Alternatives")
    st.altair_chart(chart_rank, use_container_width=True)

# -----------------------------
# Load Data
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV with numeric criteria", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No CSV uploaded. Using sample dataset.")
    df = load_sample_data()

st.subheader("Data Preview")
st.dataframe(df)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) == 0:
    st.error("No numeric columns found.")
    st.stop()

# -----------------------------
# Step 1: Select Criteria
# -----------------------------
criteria = st.multiselect("Select criteria", options=numeric_cols, default=numeric_cols)
if len(criteria) == 0:
    st.warning("Select at least one criterion.")
    st.stop()

# -----------------------------
# Step 2: Set Benefit / Cost
# -----------------------------
default_benefit = {c: True for c in criteria}
st.subheader("Benefit / Cost")
benefit_flags = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        choice = st.radio(f"{c}", ["Benefit (higher better)", "Cost (lower better)"], index=0)
        benefit_flags[c] = choice.startswith("Benefit")

# -----------------------------
# Step 3: Set Weights
# -----------------------------
st.subheader("Set Criteria Weights")
weights = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        weights[c] = st.number_input(f"Weight for {c}", min_value=0.0, value=1.0, step=0.1)

# -----------------------------
# Step 4: VIKOR Parameter
# -----------------------------
v_param = st.slider("VIKOR v (compromise factor)", 0.0, 1.0, 0.5, 0.05)

# -----------------------------
# Step 5: Run VIKOR
# -----------------------------
if st.button("Run VIKOR"):
    try:
        result_df = vikor(df, criteria, weights, benefit_flags, v=v_param)
        st.subheader("VIKOR Results")
        st.dataframe(result_df)

        chart_vikor_with_rank(result_df)

        st.success(f"Best alternative (lowest Q): {result_df.iloc[0]['VIKOR_Q']:.4f}")

        # Download CSV
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="vikor_results.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ Error running VIKOR: {e}")
