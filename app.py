import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import spearmanr, pearsonr

# ============================================================
# 🛠 Utility Functions
# ============================================================
def clean_numeric(x):
    """Convert messy values (%, commas, (), spaces) to float."""
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace(',', '').replace('%', '')
            x = x.strip()
            if x.startswith('(') and x.endswith(')'):
                x = '-' + x[1:-1]
        return float(x)
    except:
        return np.nan

def normalize_matrix(df, criteria_types):
    """Normalize decision matrix (for ARAS & VIKOR)."""
    norm = df.copy()
    for i, col in enumerate(df.columns):
        if criteria_types[i] == "benefit":
            norm[col] = df[col] / df[col].max()
        else:  # cost
            norm[col] = df[col].min() / df[col]
    return norm

def aras(df, weights, criteria_types):
    norm = normalize_matrix(df, criteria_types)
    weighted = norm * weights
    scores = weighted.sum(axis=1)
    return scores

def vikor(df, weights, criteria_types, v=0.5):
    norm = normalize_matrix(df, criteria_types)
    f_star = norm.max()
    f_minus = norm.min()
    S = ((f_star - norm) * weights).sum(axis=1) / (f_star - f_minus).replace(0, 1).sum()
    R = (((f_star - norm) * weights).max(axis=1)) / (f_star - f_minus).replace(0, 1).max()
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + \
        (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)
    return Q

# ============================================================
# 🚀 Streamlit App
# ============================================================
st.set_page_config(page_title="StockXplore", layout="wide")
st.title("📊 StockXplore: Big Data-Powered VIKOR–ARAS System")
st.caption("Undergraduates Pioneering Tomorrow’s Breakthroughs")

# ============================================================
# STEP 1: Upload or Load Dataset
# ============================================================
st.header("Step 1️⃣ : Load Dataset")

# Example dataset
sample_data = {
    "Stock": ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
    "EPS": [5.6, 6.2, 4.8, 3.2, 2.1],
    "DPS": [0.8, 1.0, 0.0, 0.0, 0.0],
    "NTA": [20, 25, 18, 15, 10],
    "PE": [30, 28, 35, 40, 60],
    "DY": [1.2, 1.5, 0.0, 0.0, 0.0],
    "ROE": [18, 22, 15, 12, 10],
    "PTBV": [5, 6, 7, 8, 10],
}
df = pd.DataFrame(sample_data)

uploaded = st.file_uploader("📂 Upload your stock dataset (CSV)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)

# Clean data
for col in df.columns[1:]:
    df[col] = df[col].apply(clean_numeric)
before = len(df)
df = df.dropna()
after = len(df)
if after < before:
    st.warning(f"⚠️ Dropped {before - after} rows with invalid/missing values.")

st.dataframe(df, use_container_width=True)

# ============================================================
# STEP 2: Define Criteria Types and Weights
# ============================================================
st.header("Step 2️⃣ : Define Criteria & Weights")

criteria = df.columns[1:]
criteria_types = []
weights = []

cols = st.columns(2)
with cols[0]:
    st.subheader("Criteria Type")
with cols[1]:
    st.subheader("Criteria Weight")

for i, c in enumerate(criteria):
    col1, col2 = st.columns(2)
    with col1:
        t = st.selectbox(f"{c} type", ["benefit", "cost"], index=0, key=f"type_{i}")
    with col2:
        w = st.number_input(f"{c} weight", min_value=0.0, max_value=1.0, value=1.0, key=f"w_{i}")
    criteria_types.append(t)
    weights.append(w)

weights = np.array(weights)
if weights.sum() == 0:
    st.error("⚠️ All weights are zero. Please adjust.")
    st.stop()
weights = weights / weights.sum()

# ============================================================
# STEP 3: Configure VIKOR
# ============================================================
st.header("Step 3️⃣ : Configure VIKOR Parameter")
v = st.slider("VIKOR v parameter", 0.0, 1.0, 0.5, 0.1)

# ============================================================
# STEP 4: Run MCDM Methods
# ============================================================
st.header("Step 4️⃣ : Run Ranking Methods")

if st.button("🚀 Run MCDM"):
    try:
        # ---- ARAS ----
        aras_scores = aras(df[criteria], weights, criteria_types)
        aras_rank = aras_scores.rank(ascending=False, method="dense").astype(int)

        # ---- VIKOR ----
        vikor_scores = vikor(df[criteria], weights, criteria_types, v)
        vikor_rank = vikor_scores.rank(ascending=True, method="dense").astype(int)

        results = pd.DataFrame({
            "Stock": df["Stock"],
            "ARAS Score": aras_scores,
            "ARAS Rank": aras_rank,
            "VIKOR Score": vikor_scores,
            "VIKOR Rank": vikor_rank
        }).sort_values("ARAS Rank")

        st.subheader("🏆 Ranking Results")
        st.dataframe(results, use_container_width=True)

        # ============================================================
        # STEP 5: Visualization
        # ============================================================
        st.header("Step 5️⃣ : Visualization")
        chart = alt.Chart(results).transform_fold(
            ["ARAS Rank", "VIKOR Rank"], as_=["Method", "Rank"]
        ).mark_bar().encode(
            x="Stock:N", y="Rank:Q", color="Method:N", column="Method:N"
        )
        st.altair_chart(chart, use_container_width=True)

        # ============================================================
        # STEP 6: Correlation Analysis
        # ============================================================
        st.header("Step 6️⃣ : Correlation Analysis")
        spear, _ = spearmanr(results["ARAS Rank"], results["VIKOR Rank"])
        pear, _ = pearsonr(results["ARAS Rank"], results["VIKOR Rank"])
        rmse = np.sqrt(((results["ARAS Rank"] - results["VIKOR Rank"])**2).mean())
        st.write(f"**Spearman:** {spear:.3f}")
        st.write(f"**Pearson:** {pear:.3f}")
        st.write(f"**RMSE:** {rmse:.3f}")

        # ============================================================
        # STEP 7: Download Results
        # ============================================================
        st.header("Step 7️⃣ : Download Results")
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results CSV", data=csv,
                           file_name="results.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
