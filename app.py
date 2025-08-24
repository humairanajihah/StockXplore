import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===============================
# App Config
# ===============================
st.set_page_config(page_title="StockXplore • VIKOR", layout="wide")
st.title("📊 StockXplore: Big Data-Powered VIKOR System")
st.markdown("Rank stock alternatives based on multiple financial criteria using **VIKOR (MCDM)**.")

# ===============================
# Sample Data Loader
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
        "PTBV": np.round(np.random.uniform(0.5, 5.0, n), 2),
        "NTA":  np.round(np.random.uniform(0.5, 8.0, n), 2),
    }
    return pd.DataFrame(data)

# ===============================
# VIKOR Functions
# ===============================
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
            denom = f_star[j] - f_minus[j]
            D[:, j] = (f_star[j] - col) / denom if denom != 0 else 0.0
        else:
            f_star[j] = np.nanmin(col)
            f_minus[j] = np.nanmax(col)
            denom = f_minus[j] - f_star[j]
            D[:, j] = (col - f_star[j]) / denom if denom != 0 else 0.0
    return D, f_star, f_minus

def vikor(df, criteria, weights_dict, benefit_dict, v=0.5):
    X = df[criteria].values
    benefit_flags = [benefit_dict[c] for c in criteria]
    D, f_star, f_minus = normalize_for_vikor(X, benefit_flags)

    w = np.array([weights_dict[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w) / len(w)

    S = (D * w).sum(axis=1)
    R = (D * w).max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

    result = df.copy()
    result["S"] = S
    result["R"] = R
    result["Q"] = Q
    result["Rank"] = result["Q"].rank(method="min").astype(int)
    result = result.sort_values("Q").reset_index(drop=True)
    return result

# ===============================
# Data Upload / Load
# ===============================
uploaded = st.file_uploader("Upload CSV with 'Name' column and numeric criteria", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset loaded.")
else:
    st.info("No file uploaded. Using sample dataset (10 companies).")
    df = load_sample_data()

st.subheader("📁 Data Preview")
st.dataframe(df.head(20))

all_cols = list(df.columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ===============================
# User Inputs: Criteria & Weights
# ===============================
st.subheader("Step 1 — Select Criteria")
criteria = st.multiselect("Select numeric criteria for VIKOR", options=numeric_cols, default=numeric_cols)

st.subheader("Step 2 — Set Weights")
weights = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        weights[c] = st.number_input(f"Weight: {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")
weights_sum = sum(weights.values())
st.caption(f"Total weight (auto-normalized): {weights_sum:.2f}")

st.subheader("Step 3 — Mark Benefit or Cost")
benefit_flags = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        choice = st.radio(f"{c}", ["Benefit (higher is better)", "Cost (lower is better)"], index=0)
        benefit_flags[c] = choice.startswith("Benefit")

v_param = st.slider("VIKOR v (strategy of majority)", 0.0, 1.0, 0.5, 0.05)

# ===============================
# Run VIKOR
# ===============================
if st.button("🚀 Run VIKOR"):
    if len(criteria) == 0:
        st.warning("Select at least one criterion!")
    else:
        try:
            df_result = vikor(df, criteria, weights, benefit_flags, v=v_param)
            st.subheader("🏁 VIKOR Results")
            st.dataframe(df_result[["Name"] + criteria + ["S", "R", "Q", "Rank"]])

            # ===============================
            # Plotly Ranking Chart
            # ===============================
            st.subheader("📊 VIKOR Q Ranking Chart")
            num_rows = len(df_result)
            top_n = 50
            if num_rows <= top_n:
                fig = px.bar(df_result, x="Name", y="Q", title="VIKOR Q Values", labels={"Q":"Q Value"})
            else:
                df_top = df_result.nsmallest(top_n, "Q")
                fig = px.bar(df_top, x="Name", y="Q", title=f"Top {top_n} VIKOR Ranked Alternatives", labels={"Q":"Q Value"})
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error running VIKOR: {e}")
