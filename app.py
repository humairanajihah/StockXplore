import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("StockXplore VIKOR System")

# -----------------------------
# Load Sample Data
# -----------------------------
def load_sample_data(n=10):
    np.random.seed(42)
    data = {
        "EPS": np.round(np.random.uniform(0.5, 6.0, n), 2),
        "DPS": np.round(np.random.uniform(0.0, 3.0, n), 2),
        "NTA": np.round(np.random.uniform(0.5, 8.0, n), 2),
        "PE": np.round(np.random.uniform(6.0, 40.0, n), 2),
        "DY": np.round(np.random.uniform(0.0, 12.0, n), 2)
    }
    return pd.DataFrame(data)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()

st.subheader("Data Preview")
st.dataframe(df)

# -----------------------------
# Keep only numeric columns
# -----------------------------
df = df.apply(pd.to_numeric, errors="coerce")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Please upload valid data.")
    st.stop()

criteria = st.multiselect("Select criteria", numeric_cols, default=numeric_cols)
if not criteria:
    st.error("Please select at least one criterion.")
    st.stop()

# Drop rows where all criteria are NaN
df = df.dropna(subset=criteria, how="all")
if df.empty:
    st.error("No valid data left after removing NaN rows.")
    st.stop()

# -----------------------------
# VIKOR Function
# -----------------------------
def vikor(df, criteria, weights=None, benefit=None, v=0.5):
    X = df[criteria].astype(float).values
    if X.size == 0:
        raise ValueError("Input matrix has zero size.")

    n, m = X.shape

    if weights is None:
        weights = np.ones(m) / m
    else:
        weights = np.array([weights[c] for c in criteria])
        weights = weights / weights.sum()

    if benefit is None:
        benefit = [True]*m

    f_star = np.max(X, axis=0)
    f_minus = np.min(X, axis=0)

    D = np.zeros_like(X)
    for j in range(m):
        if f_star[j] == f_minus[j]:
            D[:, j] = 0
        elif benefit[j]:
            D[:, j] = (f_star[j] - X[:, j]) / (f_star[j] - f_minus[j])
        else:
            D[:, j] = (X[:, j] - f_star[j]) / (f_minus[j] - f_star[j])

    S = np.dot(D, weights)
    R = np.max(D * weights, axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    Q = v*(S - S_star)/(S_minus - S_star + 1e-9) + (1-v)*(R - R_star)/(R_minus - R_star + 1e-9)

    df_vikor = df.copy()
    df_vikor["VIKOR_S"] = S
    df_vikor["VIKOR_R"] = R
    df_vikor["VIKOR_Q"] = Q
    df_vikor["Rank"] = Q.argsort() + 1
    return df_vikor.sort_values("VIKOR_Q")

# -----------------------------
# Weights & Benefit/Cost Inputs
# -----------------------------
st.subheader("Weights & Benefit/Cost")
weights = {}
benefit = {}
for c in criteria:
    weights[c] = st.number_input(f"Weight for {c}", min_value=0.0, value=1.0, step=0.1)
    benefit[c] = st.radio(f"{c} type", ["Benefit (higher better)", "Cost (lower better)"], index=0).startswith("Benefit")

v_param = st.slider("VIKOR v parameter", 0.0, 1.0, 0.5, 0.05)

# -----------------------------
# Run VIKOR
# -----------------------------
if st.button("Run VIKOR"):
    try:
        df_result = vikor(df, criteria, weights, benefit, v=v_param)
        st.subheader("VIKOR Result")
        st.dataframe(df_result)

        # -----------------------------
        # Horizontal bar chart by rank
        # -----------------------------
        df_chart = df_result.reset_index(drop=True)
        chart = alt.Chart(df_chart).mark_bar().encode(
            y=alt.Y("Rank:O", sort="ascending", title="Rank"),
            x=alt.X("VIKOR_Q:Q", title="VIKOR Q (lower is better)"),
            tooltip=[alt.Tooltip(c) for c in criteria] + ["VIKOR_Q", "Rank"]
        ).properties(height=400, title="VIKOR Ranking")
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
