# ===============================
# app.py
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===============================
# App Meta
# ===============================
APP_TITLE = "StockXplore: Big Data-Powered VIKOR System"
APP_TAGLINE = "VIKOR ranking for smarter stock screening."

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")
st.title(APP_TITLE)
st.write(APP_TAGLINE)

# ===============================
# Load / Sample Data
# ===============================
def load_sample_data(n=10, seed=42):
    np.random.seed(seed)
    data = {
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

uploaded = st.file_uploader("Upload CSV (must have 'Name' column)", type="csv")
if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    st.info("Using sample dataset")
    df_raw = load_sample_data()

# Clean column names
df_raw.columns = df_raw.columns.str.strip()

# Check Name column
if "Name" not in df_raw.columns:
    st.error("Your dataset must have a 'Name' column for alternatives.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df_raw.head(20))

# Numeric criteria
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
criteria = st.multiselect("Select numeric criteria", options=numeric_cols, default=numeric_cols)

if len(criteria) == 0:
    st.warning("Select at least one criterion.")
    st.stop()

# -------------------------------
# Step 1: Benefit / Cost
# -------------------------------
st.subheader("Mark Benefit / Cost")
benefit_dict = {}
for c in criteria:
    benefit_dict[c] = st.radio(f"{c}:", ["Benefit (higher better)", "Cost (lower better)"], index=0) == "Benefit (higher better)"

# -------------------------------
# Step 2: Input Weights
# -------------------------------
st.subheader("Set Criteria Weights")
weights_dict = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        weights_dict[c] = st.number_input(f"Weight: {c}", min_value=0.0, value=1.0, step=0.1)

# Normalize weights
w_array = np.array(list(weights_dict.values()), dtype=float)
if w_array.sum() != 0:
    w_array = w_array / w_array.sum()
else:
    w_array = np.ones(len(criteria)) / len(criteria)
weights_dict = dict(zip(criteria, w_array))

# -------------------------------
# Step 3: VIKOR Calculation
# -------------------------------
def vikor(df, criteria, benefit_dict, weights_dict, v=0.5):
    X = df[criteria].values.astype(float)
    f_star = np.array([X[:,i].max() if benefit_dict[criteria[i]] else X[:,i].min() for i in range(len(criteria))])
    f_minus = np.array([X[:,i].min() if benefit_dict[criteria[i]] else X[:,i].max() for i in range(len(criteria))])
    
    D = np.zeros_like(X)
    for j in range(len(criteria)):
        denom = f_star[j] - f_minus[j] if f_star[j] != f_minus[j] else 1e-9
        if benefit_dict[criteria[j]]:
            D[:,j] = (f_star[j] - X[:,j]) / denom
        else:
            D[:,j] = (X[:,j] - f_star[j]) / denom
    
    w = np.array([weights_dict[c] for c in criteria])
    S = np.dot(D, w)
    R = np.max(D * w, axis=1)
    
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    
    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)
    
    df_result = df[["Name"]].copy()
    df_result["S"] = S
    df_result["R"] = R
    df_result["Q"] = Q
    df_result["Rank"] = df_result["Q"].rank(method="min").astype(int)
    df_result = df_result.sort_values("Q")
    
    return df_result

v = st.slider("VIKOR v (compromise factor)", 0.0, 1.0, 0.5, 0.05)
if st.button("Run VIKOR"):
    df_result = vikor(df_raw, criteria, benefit_dict, weights_dict, v=v)
    
    st.subheader("VIKOR Ranking")
    st.dataframe(df_result)
    
    st.success(f"Best alternative: {df_result.iloc[0]['Name']}")
    
    # -------------------------------
    # Step 4: Plotly Chart
    # -------------------------------
    st.subheader("Ranking Graph (Q Values)")
    fig = px.bar(df_result, x="Name", y="Q", text="Q", title="VIKOR Ranking (Lower Q is Better)")
    fig.update_layout(xaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig, use_container_width=True)
