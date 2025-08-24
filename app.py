# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# App Config
# ===============================
st.set_page_config(page_title="StockXplore • VIKOR", layout="wide")
st.title("📊 StockXplore: Big Data-Powered VIKOR System")
st.markdown("Rank stocks using the **VIKOR (MCDM)** method. Upload your dataset, define criteria, set weights, and get interactive rankings.")

# ===============================
# Sample Data Function
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

# ===============================
# Helper Functions
# ===============================
def to_float_safe(series):
    def _conv(x):
        if pd.isna(x):
            return np.nan
        try:
            return float(str(x).replace(",", "").replace("%",""))
        except:
            return np.nan
    return series.map(_conv)

def coerce_numeric(df, cols):
    df_copy = df.copy()
    for c in cols:
        df_copy[c] = to_float_safe(df_copy[c])
    return df_copy

# ===============================
# VIKOR Calculation
# ===============================
def vikor(df, criteria, weights=None, v=0.5):
    X = df[criteria].astype(float).values
    if X.shape[0] == 0 or X.shape[1] == 0:
        return pd.DataFrame(), {}
    
    # Default: all criteria are benefit except typical cost ones
    benefit_flags = [True if c.upper() not in ["PE","PTBV"] else False for c in criteria]

    # Normalize
    D = np.zeros_like(X)
    for j in range(len(criteria)):
        col = X[:, j]
        if benefit_flags[j]:
            f_star, f_minus = np.nanmax(col), np.nanmin(col)
            denom = f_star - f_minus
            D[:, j] = (f_star - col)/denom if denom !=0 else 0
        else:
            f_star, f_minus = np.nanmin(col), np.nanmax(col)
            denom = f_minus - f_star
            D[:, j] = (col - f_star)/denom if denom !=0 else 0

    # Weights
    if weights is None:
        w = np.ones(len(criteria))/len(criteria)
    else:
        w_arr = np.array([float(weights.get(c,1.0)) for c in criteria])
        w = w_arr / w_arr.sum() if w_arr.sum() !=0 else np.ones(len(criteria))/len(criteria)

    S = (D * w).sum(axis=1)
    R = (D * w).max(axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    Q = v * (S - S_star)/(S_minus - S_star + 1e-9) + (1-v)*(R - R_star)/(R_minus - R_star + 1e-9)

    df_result = df.copy()
    df_result["S"] = S
    df_result["R"] = R
    df_result["Q"] = Q
    df_result["Rank"] = df_result["Q"].rank(method="min").astype(int)
    df_result = df_result.sort_values("Q").reset_index(drop=True)
    return df_result

# ===============================
# Upload / Load Data
# ===============================
uploaded_file = st.file_uploader("📂 Upload CSV (Name column + numeric criteria)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Raw Data")
    st.dataframe(df)
else:
    st.info("No file uploaded. Using sample dataset.")
    df = load_sample_data()
    st.dataframe(df)

# Select numeric criteria
all_cols = list(df.columns)
if "Name" not in all_cols:
    st.error("Your dataset must have a 'Name' column for alternatives.")
    st.stop()

numeric_candidates = [c for c in all_cols if c != "Name"]
df_numeric = coerce_numeric(df, numeric_candidates)
numeric_cols = [c for c in numeric_candidates if pd.api.types.is_numeric_dtype(df_numeric[c])]

criteria = st.multiselect("Select numeric criteria for VIKOR", numeric_cols, default=numeric_cols)
if len(criteria)==0:
    st.warning("Select at least one numeric criterion.")
    st.stop()

# Set weights
st.subheader("Set weights for each criterion")
weights = {}
cols = st.columns(len(criteria))
for i, c in enumerate(criteria):
    with cols[i]:
        weights[c] = st.number_input(f"{c}", min_value=0.0, value=1.0, step=0.1)

# Set VIKOR v
v_param = st.slider("VIKOR v (compromise factor)", 0.0, 1.0, 0.5, 0.05)

# Run VIKOR
if st.button("🚀 Run VIKOR"):
    try:
        df_result = vikor(df_numeric[["Name"] + criteria], criteria, weights=weights, v=v_param)
        if df_result.empty:
            st.error("No data to process. Check your criteria selection.")
        else:
            # Display table
            display_cols = ["Name"] + [c for c in criteria if c in df_result.columns] + [col for col in ["S","R","Q","Rank"] if col in df_result.columns]
            st.subheader("🏁 VIKOR Ranking")
            st.dataframe(df_result[display_cols])

            # Display chart
            st.subheader("📊 Ranking Chart by Q Value")
            plt.figure(figsize=(10,6))
            plt.bar(df_result["Name"], df_result["Q"], color='skyblue')
            plt.xlabel("Stock Name")
            plt.ylabel("VIKOR Q (Lower is better)")
            plt.title("Stock Ranking")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error: {e}")
