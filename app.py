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
# Helpers
# ===============================
def load_sample_data(n=10, seed=42):
    np.random.seed(seed)
    data = {
        "Ticker": [f"S{str(i+1).zfill(3)}" for i in range(n)],
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

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
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
        col = X[:, j]
        if benefit_flags[j]:
            f_star[j] = np.nanmax(col)
            f_minus[j] = np.nanmin(col)
            denom = f_star[j] - f_minus[j]
            D[:, j] = 0.0 if denom == 0 else (f_star[j] - col)/denom
        else:
            f_star[j] = np.nanmin(col)
            f_minus[j] = np.nanmax(col)
            denom = f_minus[j] - f_star[j]
            D[:, j] = 0.0 if denom == 0 else (col - f_star[j])/denom
    return D, f_star, f_minus

def vikor(df, id_col, criteria, weights_dict, benefit_dict, v=0.5):
    X = df[criteria].astype(float).values
    benefit_flags = [bool(benefit_dict[c]) for c in criteria]
    D, f_star, f_minus = normalize_for_vikor(X, benefit_flags)

    w = np.array([float(weights_dict[c]) for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)

    S = (D * w).sum(axis=1)
    R = (D * w).max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    S_denom = S_minus - S_star if S_minus != S_star else 1.0
    R_denom = R_minus - R_star if R_minus != R_star else 1.0

    Q = v * (S - S_star)/S_denom + (1-v)*(R - R_star)/R_denom

    out = df[[id_col]].copy()
    out["VIKOR_S"] = S
    out["VIKOR_R"] = R
    out["VIKOR_Q"] = Q
    out["VIKOR_Rank"] = out["VIKOR_Q"].rank(method="min").astype(int)

    return out.sort_values("VIKOR_Q").reset_index(drop=True)

# ===============================
# Chart
# ===============================
def chart_vikor(df, id_col):
    df_plot = df.copy()
    df_plot[id_col] = df_plot[id_col].astype(str)
    df_plot['Y_label'] = df_plot[id_col] + "_" + df_plot.index.astype(str)
    df_plot = df_plot.sort_values("VIKOR_Q", ascending=True)

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("VIKOR_Q:Q", title="VIKOR Q (lower is better)"),
            y=alt.Y("Y_label:N", sort='-x', title=id_col),
            tooltip=[id_col, "VIKOR_S:Q", "VIKOR_R:Q", "VIKOR_Q:Q", "VIKOR_Rank:Q"]
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("📈 StockXplore • VIKOR")
    st.caption(APP_TAGLINE)
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown(
        "1) Upload dataset or use sample.\n"
        "2) Pick ID and numeric criteria.\n"
        "3) Set weights and mark Benefit/Cost.\n"
        "4) Run VIKOR, review chart, download results."
    )

# ===============================
# Main
# ===============================
st.title(APP_TITLE)
st.write(APP_TAGLINE)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
    except:
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, encoding="latin1")
else:
    st.info("No file uploaded. Using sample data.")
    df_raw = load_sample_data()

# Clean column names
df_raw.columns = df_raw.columns.str.strip()
st.dataframe(df_raw.head(20), use_container_width=True)

all_cols = df_raw.columns.tolist()
id_col = st.selectbox("Select ID column", all_cols, index=0)
num_candidates = [c for c in all_cols if c != id_col]
df_num = coerce_numeric(df_raw.copy(), num_candidates)
numeric_cols = [c for c in num_candidates if pd.api.types.is_numeric_dtype(df_num[c])]
criteria = st.multiselect("Select numeric criteria", numeric_cols, default=numeric_cols)

if len(criteria) == 0:
    st.warning("Select at least one criterion.")
    st.stop()

df_num = df_num.dropna(subset=criteria)

# Step 2 — Benefit/Cost
st.subheader("Step 2 — Mark Benefit or Cost")
cols = st.columns(min(4, len(criteria)))
benefit_flags = {}
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        choice = st.radio(c, ["Benefit (higher better)", "Cost (lower better)"], index=0, key=f"bc_{c}")
        benefit_flags[c] = choice.startswith("Benefit")

# Step 3 — Weights
st.subheader("Step 3 — Set Weights")
cols = st.columns(min(4, len(criteria)))
weights = {}
for i, c in enumerate(criteria):
    with cols[i % len(cols)]:
        weights[c] = st.number_input(f"Weight {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")
st.caption(f"Total weight: {sum(weights.values()):.2f} (auto-normalized)")

# Step 4 — VIKOR Parameter
st.subheader("Step 4 — VIKOR v Parameter")
v_param = st.slider("VIKOR v (0=individual, 1=group)", 0.0, 1.0, 0.5, 0.05)

st.markdown("---")
if st.button("🚀 Run VIKOR"):
    try:
        vikor_df = vikor(df_num[[id_col]+criteria], id_col, criteria, weights, benefit_flags, v=v_param)
        st.subheader("VIKOR Results")
        st.dataframe(vikor_df, use_container_width=True)
        st.markdown("### 📊 VIKOR Q Bar Chart")
        chart_vikor(vikor_df, id_col)
        csv = vikor_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="vikor_results.csv")
    except Exception as e:
        st.error(f"❌ Error running VIKOR: {e}")
