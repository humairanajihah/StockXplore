import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="VIKOR-Stocks", layout="wide")
st.title("📊 VIKOR-Stocks: Intelligent Stock Ranking System")
st.markdown("This app applies the **VIKOR (MCDM)** method to rank stock alternatives based on multiple financial criteria.")

# -----------------------------
# Upload CSV File
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file (First column = Alternatives, rest = Criteria)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Raw Data")
    st.dataframe(df)

    # Split alternatives & criteria
    alternatives = df.iloc[:, 0].values
    criteria = df.iloc[:, 1:]

    # -----------------------------
    # Step 1: Define benefit & cost criteria
    # -----------------------------
    benefit_criteria = ['EPS', 'DPS', 'NTA', 'DY', 'ROE', 'GPM', 'OPM', 'ROA']
    cost_criteria = ['PE', 'PTBV']

    # -----------------------------
    # Step 2: Normalize Decision Matrix
    # -----------------------------
    norm = pd.DataFrame()
    for col in criteria.columns:
        if col in benefit_criteria:
            norm[col] = (criteria[col] - criteria[col].min()) / (criteria[col].max() - criteria[col].min())
        elif col in cost_criteria:
            norm[col] = (criteria[col].max() - criteria[col]) / (criteria[col].max() - criteria[col].min())

    st.markdown("### ✅ Step 1: Normalized Matrix")
    st.dataframe(norm)

    # -----------------------------
    # Step 3: Determine Best & Worst values
    # -----------------------------
    f_star = norm.max()
    f_minus = norm.min()

    st.markdown("### ⭐ Step 2: Best (f*) and Worst (f-) Values")
    st.write("Best (f*):", f_star.to_dict())
    st.write("Worst (f-):", f_minus.to_dict())

    # -----------------------------
    # Step 4: Compute S and R
    # -----------------------------
    weights = np.ones(len(norm.columns)) / len(norm.columns)  # Equal weights
    weights_series = pd.Series(weights, index=norm.columns)

    S = ((weights_series * (f_star - norm) / (f_star - f_minus + 1e-9)).sum(axis=1))
    R = ((weights_series * (f_star - norm) / (f_star - f_minus + 1e-9)).max(axis=1))

    st.markdown("### 📉 Step 3: Utility (Sᵢ) and Regret (Rᵢ)")
    st.dataframe(pd.DataFrame({'Alternative': alternatives, 'S': S, 'R': R}))

    # -----------------------------
    # Step 5: Compute Q index
    # -----------------------------
    v = 0.5  # compromise factor
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

    # -----------------------------
    # Step 6: Rank Alternatives
    # -----------------------------
    result_df = pd.DataFrame({
        'Alternative': alternatives,
        'S': S,
        'R': R,
        'Q': Q
    }).sort_values(by='Q').reset_index(drop=True)

    st.subheader("🏁 Final VIKOR Ranking")
    st.dataframe(result_df)

    st.success(f"🎯 Best Alternative: {result_df.iloc[0]['Alternative']}")

    # -----------------------------
    # Step 7: Visualization
    # -----------------------------
    st.markdown("### 📊 VIKOR Q Value Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(result_df['Alternative'], result_df['Q'], color='skyblue')
    ax.set_xlabel("Alternative")
    ax.set_ylabel("Q Value")
    ax.set_title("Ranking Based on Q Values (Lower is Better)")
    ax.set_xticklabels(result_df['Alternative'], rotation=90)
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file with 1 alternative column and multiple criteria columns.")
