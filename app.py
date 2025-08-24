# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ===============================
# App Meta
# ===============================
APP_TITLE = "StockXplore: Big Data-Powered VIKOR System for Smarter Stock Selection"
APP_TAGLINE = "Undergraduates Pioneering Tomorrow’s Breakthroughs — VIKOR ranking for stock screening."

st.set_page_config(page_title="StockXplore • VIKOR", layout="wide", page_icon="📈")

# ===============================
# Helpers: Data & Cleaning
# ===============================
def load_sample_data(n=10, seed=42):
    np.random.seed(seed)
    data = {
        "Ticker": [f"S{str(i+1).zfill(3)}" for i in range(n)],
        "Name": [f"Company {i+1}" for i in range(n)],
        "EPS":  np.round(np.random.uniform(0.5, 6.0, n), 2),   # benefit
        "DPS":  np.round(np.random.uniform(0.0, 3.0, n), 2),   # benefit
        "NTA":  np.round(np.random.uniform(0.5, 8.0, n), 2),   # benefit
        "PE":   np.round(np.random.uniform(6.0, 40.0, n), 2),  # cost
        "DY":   np.round(np.random.uniform(0.0, 12.0, n), 2),  # benefit
        "ROE":  np.round(np.random.uniform(0.0, 30.0, n), 2),  # benefit
        "PTBV": np.round(np.random.uniform(0.5, 5.0, n), 2),   # cost
    }
    return pd.DataFrame(data)

def to_float_safe(series):
    def _conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace(",", "").replace("%", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        try:
            return float(s)
        except:
            return np.nan
    return series.map(_conv)

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = to_float_safe(out[c])
    return out

# ===============================
# VIKOR Core
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
            D[:, j] = 0.0 if denom==0 else (f_star[j]-col)/denom
        else:
            f_star[j] = np.nanmin(col)
            f_minus[j] = np.nanmax(col)
            denom = f_minus[j]-f_star[j]
            D[:, j] = 0.0 if denom==0 else (col-f_star)/denom
    return D, f_star, f_minus

def vikor(df, id_col, criteria, weights_dict, benefit_dict, v=0.5):
    if df.empty:
        raise ValueError("No valid rows available for VIKOR calculation.")
    X = df[criteria].astype(float).values
    if X.size == 0:
        raise ValueError("Criteria array is empty. Check numeric columns and uploaded data.")
    benefit_flags = [bool(benefit_dict[c]) for c in criteria]
    D, f_star, f_minus = normalize_for_vikor(X, benefit_flags)
    w = np.array([float(weights_dict[c]) for c in criteria], dtype=float)
    w = w / w.sum() if w.sum()!=0 else np.ones_like(w)/len(w)
    S = (D*w).sum(axis=1)
    R = (D*w).max(axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    S_denom = S_minus - S_star if S_minus != S_star else 1.0
    R_denom = R_minus - R_star if R_minus != R_star else 1.0
    Q = v*(S-S_star)/S_denom + (1-v)*(R-R_star)/R_denom
    out = df[[id_col]].copy()
    out["VIKOR_S"] = S
    out["VIKOR_R"] = R
    out["VIKOR_Q"] = Q
    out["VIKOR_Rank"] = out["VIKOR_Q"].rank(method="min").astype(int)
    return out.sort_values("VIKOR_Q")

# ===============================
# Chart
# ===============================
def chart_vikor(df, id_col):
    # Ensure id_col is treated as nominal
    df_plot = df.copy()
    df_plot[id_col] = df_plot[id_col].astype(str)
    chart = alt.Chart(df_plot.sort_values("VIKOR_Q", ascending=True)).mark_bar().encode(
        x=alt.X("VIKOR_Q:Q", title="VIKOR Q (lower is better)"),
        y=alt.Y(f"{id_col}:N", sort='-x', title=id_col),
        tooltip=[id_col, "VIKOR_S:Q", "VIKOR_R:Q", "VIKOR_Q:Q", "VIKOR_Rank:Q"]
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("📈 StockXplore • VIKOR")
    st.caption(APP_TAGLINE)
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1) Upload CSV or use sample.\n2) Pick ID column & numeric criteria.\n3) Set weights & mark Benefit/Cost.\n4) Run VIKOR & view chart.")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Built by undergraduates to power smarter stock decisions.")

# ===============================
# Main App
# ===============================
st.title(APP_TITLE)
st.write(APP_TAGLINE)

# Step 1 — Load Data
uploaded = st.file_uploader("Upload CSV (alternatives as rows, criteria as columns)", type=["csv"])
if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, encoding="latin1")
    st.success("Dataset loaded.")
else:
    st.info("No file uploaded. Using sample dataset.")
    df_raw = load_sample_data()

st.dataframe(df_raw.head(20), use_container_width=True)

# Step 2 — Select ID and numeric criteria
all_cols = list(df_raw.columns)
id_col = st.selectbox("Select ID column", options=all_cols, index=0)
num_candidates = [c for c in all_cols if c != id_col]
df_num = coerce_numeric(df_raw.copy(), num_candidates)
numeric_cols = [c for c in num_candidates if pd.api.types.is_numeric_dtype(df_num[c])]

if len(numeric_cols) == 0:
    st.error("❌ No numeric columns detected. Please check your CSV.")
    st.stop()

criteria = st.multiselect("Select numeric criteria", options=numeric_cols, default=numeric_cols)
if len(criteria) == 0:
    st.warning("Select at least one criterion.")
    st.stop()

# Drop rows with missing values
df_num = df_num.dropna(subset=criteria)
if df_num.shape[0]==0:
    st.error("❌ No valid rows left after removing missing values.")
    st.stop()

# Step 3 — Benefit/Cost
st.subheader("Mark Benefit or Cost")
default_benefit = {c: (c.upper() not in ["PE", "PTBV"]) for c in criteria}
benefit_flags = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        idx = 0 if default_benefit[c] else 1
        choice = st.radio(f"{c}", ["Benefit (higher is better)", "Cost (lower is better)"], index=idx, key=f"bc_{c}")
        benefit_flags[c] = choice.startswith("Benefit")

# Step 4 — Weights
st.subheader("Set Weights")
weights = {}
cols = st.columns(min(4, len(criteria)))
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        weights[c] = st.number_input(f"Weight: {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")

# Step 5 — VIKOR parameter
v_param = st.slider("VIKOR v (compromise factor)", 0.0, 1.0, 0.5, 0.05)
st.markdown("---")
run = st.button("🚀 Run VIKOR")

if run:
    try:
        vikor_df = vikor(df_num[[id_col]+criteria], id_col, criteria, weights, benefit_flags, v=v_param)
        st.subheader("VIKOR Results")
        st.dataframe(vikor_df, use_container_width=True)
        chart_vikor(vikor_df, id_col)
    except Exception as e:
        st.error(f"❌ Error running VIKOR: {e}")

# Footer
st.markdown("---")
st.caption("© 2025 StockXplore • VIKOR • Streamlit")
