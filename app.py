import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="StockXplore VIKOR", layout="wide")
st.title("📊 StockXplore: VIKOR Stock Ranking System")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV (First column = Alternatives, rest = numeric criteria)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Raw Data")
    st.dataframe(df)

    # -----------------------------
    # Auto-detect name column
    # -----------------------------
    # Pick first non-numeric column as Name
    name_col = None
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            name_col = col
            break

    if name_col is None:
        st.error("No non-numeric column found for alternative names. Please include one.")
        st.stop()

    st.markdown(f"**Detected alternatives column:** `{name_col}`")

    # Numeric criteria
    criteria = [c for c in df.columns if c != name_col and pd.api.types.is_numeric_dtype(df[c])]
    if not criteria:
        st.error("No numeric criteria found.")
        st.stop()

    # -----------------------------
    # Step 1: Input weights
    # -----------------------------
    st.subheader("Step 1 — Set weights for each criterion")
    weights = []
    for c in criteria:
        w = st.number_input(f"Weight: {c}", min_value=0.0, value=1.0, step=0.1)
        weights.append(w)
    weights = np.array(weights)
    weights = weights / weights.sum() if weights.sum() != 0 else np.ones_like(weights) / len(weights)

    # Step 2: Mark benefit/cost
    st.subheader("Step 2 — Mark Benefit or Cost")
    benefit_flags = []
    for c in criteria:
        choice = st.radio(f"{c}", ["Benefit (higher better)", "Cost (lower better)"], index=0, key=f"bc_{c}")
        benefit_flags.append(choice.startswith("Benefit"))

    # Step 3: VIKOR parameter v
    v_param = st.slider("Step 3 — VIKOR v (compromise factor)", 0.0, 1.0, 0.5, 0.05)

    # -----------------------------
    # Step 4: Compute VIKOR
    # -----------------------------
    df_numeric = df[[name_col] + criteria].copy()

    X = df_numeric[criteria].values.astype(float)
    m, n = X.shape
    D = np.zeros_like(X)
    f_star = np.zeros(n)
    f_minus = np.zeros(n)

    for j in range(n):
        col = X[:, j]
        if benefit_flags[j]:
            f_star[j] = np.max(col)
            f_minus[j] = np.min(col)
            denom = f_star[j] - f_minus[j]
            D[:, j] = (f_star[j] - col) / denom if denom != 0 else 0
        else:
            f_star[j] = np.min(col)
            f_minus[j] = np.max(col)
            denom = f_minus[j] - f_star[j]
            D[:, j] = (col - f_star[j]) / denom if denom != 0 else 0

    S = (D * weights).sum(axis=1)
    R = (D * weights).max(axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    Q = v_param * (S - S_star) / (S_minus - S_star + 1e-9) + (1 - v_param) * (R - R_star) / (R_minus - R_star + 1e-9)

    df_result = pd.DataFrame({
        name_col: df_numeric[name_col],
        **{c: df_numeric[c] for c in criteria},
        "S": S,
        "R": R,
        "Q": Q
    })
    df_result["Rank"] = df_result["Q"].rank(method="min").astype(int)
    df_result = df_result.sort_values("Rank")

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("🏁 VIKOR Ranking Results")
    st.dataframe(df_result)

    # -----------------------------
    # Bar chart of Q values
    # -----------------------------
    st.subheader("📊 Q Value Ranking Chart")
    plt.figure(figsize=(10, 6))
    plt.bar(df_result[name_col], df_result["Q"], color='skyblue')
    plt.xlabel("Alternative")
    plt.ylabel("Q Value")
    plt.title("VIKOR Ranking Based on Q Values (Lower is Better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

else:
    st.info("Please upload a CSV file to start.")

