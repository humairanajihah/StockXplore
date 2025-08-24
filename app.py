import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="StockXplore • VIKOR", layout="wide")
st.title("📊 StockXplore: Big Data-Powered VIKOR System")
st.markdown("Rank stocks using the **VIKOR (MCDM)** method. Enter weights, mark benefit/cost, and see interactive results.")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded = st.file_uploader("Upload CSV (must have 'Name' column + numeric criteria)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📁 Raw Data")
    st.dataframe(df)
else:
    # Sample data if no upload
    n = 10
    df = pd.DataFrame({
        "Name": [f"Stock {i+1}" for i in range(n)],
        "EPS": np.round(np.random.uniform(0.5, 6.0, n), 2),
        "DPS": np.round(np.random.uniform(0.0, 3.0, n), 2),
        "ROE": np.round(np.random.uniform(0, 30, n), 2),
        "PE": np.round(np.random.uniform(6, 40, n), 2),
        "PTBV": np.round(np.random.uniform(0.5, 5.0, n), 2)
    })
    st.info("Using sample data.")
    st.dataframe(df)

# -----------------------------
# Select criteria (numeric)
# -----------------------------
criteria = st.multiselect("Select numeric criteria to use", options=[c for c in df.columns if c != "Name"], default=[c for c in df.columns if c != "Name"])

if not criteria:
    st.stop()

# -----------------------------
# Input weights
# -----------------------------
st.subheader("Set Criteria Weights")
weights = {}
cols = st.columns(len(criteria))
for i, c in enumerate(criteria):
    with cols[i]:
        w = st.number_input(f"Weight {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")
        weights[c] = w
weights_array = np.array(list(weights.values()))
weights_array = weights_array / weights_array.sum()  # normalize

# -----------------------------
# Benefit/Cost selection
# -----------------------------
st.subheader("Mark Benefit or Cost")
benefit_dict = {}
for i, c in enumerate(criteria):
    choice = st.radio(f"{c}", ["Benefit (higher is better)", "Cost (lower is better)"], index=0, key=f"bc_{c}")
    benefit_dict[c] = choice.startswith("Benefit")

# -----------------------------
# VIKOR calculation
# -----------------------------
def vikor(df, criteria, weights_array, benefit_dict, v=0.5):
    # Numeric conversion & fill NaN
    X = df[criteria].apply(pd.to_numeric, errors='coerce').fillna(0).values
    m = len(criteria)

    f_star = np.max(X, axis=0)
    f_minus = np.min(X, axis=0)

    D = np.zeros_like(X, dtype=float)
    for j in range(m):
        denom = f_star[j] - f_minus[j]
        if denom == 0:
            D[:, j] = 0
        elif benefit_dict[criteria[j]]:
            D[:, j] = (f_star[j] - X[:, j]) / denom
        else:
            D[:, j] = (X[:, j] - f_star[j]) / denom

    S = np.dot(D, weights_array)
    R = np.max(D * weights_array, axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

    df_vikor = df.copy()
    df_vikor["S"] = S
    df_vikor["R"] = R
    df_vikor["Q"] = Q
    df_vikor["Rank"] = Q.argsort() + 1
    return df_vikor.sort_values("Q").reset_index(drop=True)

v = st.slider("VIKOR v (majority vs individual strategy)", 0.0, 1.0, 0.5, 0.05)
df_result = vikor(df, criteria, weights_array, benefit_dict, v=v)

# -----------------------------
# Display results
# -----------------------------
st.subheader("🏁 VIKOR Ranking")
st.dataframe(df_result[["Name"] + criteria + ["S","R","Q","Rank"]])

# -----------------------------
# Plot chart
# -----------------------------
st.subheader("📊 Ranking Chart")
plt.figure(figsize=(10,6))
plt.bar(df_result["Name"], df_result["Q"], color='skyblue')
plt.xlabel("Stock Name")
plt.ylabel("VIKOR Q Value (Lower is better)")
plt.title("Stock Ranking by VIKOR Q")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(plt)

