# stockxplore_app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ===============================
# App Meta
# ===============================
APP_TITLE = "StockXplore: Big Data-Powered VIKOR System"
APP_TAGLINE = "Interactive MCDM ranking for smarter stock selection"

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")
st.title(APP_TITLE)
st.markdown(APP_TAGLINE)

# ===============================
# Helpers
# ===============================
def load_sample_data(n=10, seed=42):
    np.random.seed(seed)
    data = {
        "Name": [f"Company {i+1}" for i in range(n)],
        "EPS":  np.round(np.random.uniform(0.5, 6.0, n), 2),
        "DPS":  np.round(np.random.uniform(0.0, 3.0, n), 2),
        "ROE":  np.round(np.random.uniform(0.0, 30.0, n), 2),
        "DY":   np.round(np.random.uniform(0.0, 12.0, n), 2),
        "PE":   np.round(np.random.uniform(6.0, 40.0, n), 2),
        "NTA":  np.round(np.random.uniform(0.5, 8.0, n), 2),
        "PTBV": np.round(np.random.uniform(0.5, 5.0, n), 2)
    }
    return pd.DataFrame(data)

def coerce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def normalize_for_vikor(X, benefit_flags):
    m, n = X.shape
    D = np.zeros_like(X, dtype=float)
    f_star = np.zeros(n, dtype=float)
    f_minus = np.zeros(n, dtype=float)
    for j in range(n):
        col = X[:, j].astype(float)
        if benefit_flags[j]:
            f_star[j] = np.nanmax(col)
            f_minus[j] = np.nanmin(col)
            denom = f_star[j] - f_minus[j] or 1.0
            D[:, j] = (f_star[j] - col) / denom
        else:
            f_star[j] = np.nanmin(col)
            f_minus[j] = np.nanmax(col)
            denom = f_minus[j] - f_star[j] or 1.0
            D[:, j] = (col - f_star[j]) / denom
    return D, f_star, f_minus

def vikor(df, criteria, weights_dict, benefit_dict, v=0.5):
    X = df[criteria].values.astype(float)
    benefit_flags = [benefit_dict[c] for c in criteria]
    D, f_star, f_minus = normalize_for_vikor(X, benefit_flags)
    w = np.array([weights_dict[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)
    S = (D * w).sum(axis=1)
    R = (D * w).max(axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    Q = v * (S - S_star) / (S_minus - S_star or 1) + (1 - v) * (R - R_star) / (R_minus - R_star or 1)
    df_result = df.copy()
    df_result['S'] = S
    df_result['R'] = R
    df_result['Q'] = Q
    df_result['Rank'] = df_result['Q'].rank(method='min').astype(int)
    return df_result.sort_values('Q')

# ===============================
# Sidebar / Data Input
# ===============================
with st.sidebar:
    st.header("Data Input")
    uploaded = st.file_uploader("Upload CSV (must have 'Name' column)", type="csv")
    use_sample = st.button("Use Sample Data")

if uploaded:
    df_raw = pd.read_csv(uploaded)
elif use_sample:
    df_raw = load_sample_data()
else:
    st.info("Upload CSV or use sample dataset to continue.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df_raw.head(20))

# Identify numeric columns
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in your dataset.")
    st.stop()

# ===============================
# Criteria Selection
# ===============================
st.subheader("Select Criteria (numeric columns)")
criteria = st.multiselect("Pick numeric criteria for VIKOR", numeric_cols, default=numeric_cols)
if not criteria:
    st.warning("Please select at least one numeric criterion.")
    st.stop()

# Benefit / Cost
st.subheader("Mark Criteria as Benefit or Cost")
benefit_flags = {}
for c in criteria:
    benefit_flags[c] = st.radio(f"{c}", ["Benefit (higher is better)", "Cost (lower is better)"], index=0) == "Benefit (higher is better)"

# Set Weights
st.subheader("Set Weights for Each Criterion")
weights = {}
cols = st.columns(len(criteria))
for i, c in enumerate(criteria):
    with cols[i]:
        weights[c] = st.number_input(f"Weight: {c}", min_value=0.0, value=1.0, step=0.1)

# VIKOR v parameter
v = st.slider("VIKOR v (strategy of majority)", 0.0, 1.0, 0.5, 0.05)

# ===============================
# Run VIKOR
# ===============================
if st.button("🚀 Run VIKOR"):
    df_num = coerce_numeric(df_raw.copy(), criteria)
    df_num = df_num.dropna(subset=criteria)
    if df_num.empty:
        st.error("No valid data after removing missing numeric values.")
        st.stop()
    df_result = vikor(df_num, criteria, weights, benefit_flags, v)
    st.subheader("VIKOR Result Table")
    st.dataframe(df_result[["Name"] + criteria + ["S","R","Q","Rank"]])

    # ===============================
    # Chart (Top 50 for readability)
    # ===============================
    st.subheader("VIKOR Ranking Chart (Top 50)")
    top_n = 50
    df_top = df_result.head(top_n)
    chart = alt.Chart(df_top).mark_bar().encode(
        x=alt.X('Name:N', sort=None),
        y='Q:Q',
        tooltip=['Name','Q','Rank']
    ).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)

    # Download CSV
    csv_data = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download VIKOR Results", data=csv_data, file_name="vikor_results.csv")
