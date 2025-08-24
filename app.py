# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
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
        # Hypothetical fundamentals
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
    """Convert messy strings (commas, %, spaces, (neg)) to float safely."""
    def _conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace(",", "")
        s = s.replace("%", "")
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
    """
    X: numpy array (m x n)
    benefit_flags: list[bool] length n (True=benefit, False=cost)
    Returns:
      D (distances matrix), f_star, f_minus
      Where D_ij = normalized distance from best on criterion j
    """
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
            if denom == 0:
                D[:, j] = 0.0
            else:
                # distance to best (max)
                D[:, j] = (f_star[j] - col) / denom
        else:
            f_star[j] = np.nanmin(col)
            f_minus[j] = np.nanmax(col)
            denom = f_minus[j] - f_star[j]
            if denom == 0:
                D[:, j] = 0.0
            else:
                # distance to best (min)
                D[:, j] = (col - f_star[j]) / denom
    return D, f_star, f_minus

def vikor(df, id_col, criteria, weights_dict, benefit_dict, v=0.5):
    """
    df: DataFrame with id_col and criteria columns
    weights_dict: {criterion: weight}
    benefit_dict: {criterion: True/False}
    v: strategy weight (0..1)
    Returns: results_df (id, S, R, Q, Rank), meta
    """
    X = df[criteria].astype(float).values
    benefit_flags = [bool(benefit_dict[c]) for c in criteria]
    D, f_star, f_minus = normalize_for_vikor(X, benefit_flags)

    w = np.array([float(weights_dict[c]) for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w) / len(w)

    S = (D * w).sum(axis=1)    # group utility
    R = (D * w).max(axis=1)    # individual regret

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    S_denom = (S_minus - S_star) if S_minus != S_star else 1.0
    R_denom = (R_minus - R_star) if R_minus != R_star else 1.0

    Q = v * (S - S_star) / S_denom + (1 - v) * (R - R_star) / R_denom

    out = df[[id_col]].copy()
    out["VIKOR_S"] = S
    out["VIKOR_R"] = R
    out["VIKOR_Q"] = Q
    out["VIKOR_Rank"] = out["VIKOR_Q"].rank(method="min").astype(int)

    # Compromise solution checks
    order = out.sort_values("VIKOR_Q", ascending=True).reset_index(drop=True)
    best_Q = order.loc[0, "VIKOR_Q"]
    second_Q = order.loc[1, "VIKOR_Q"] if len(order) > 1 else np.nan
    DQ = second_Q - best_Q if not np.isnan(second_Q) else np.nan
    acceptable_advantage = (DQ >= (0.5 / (len(df) - 1))) if (len(df) > 1 and not np.isnan(DQ)) else True

    best_id = order.loc[0, id_col]
    best_S_id = out.sort_values("VIKOR_S").iloc[0][id_col]
    best_R_id = out.sort_values("VIKOR_R").iloc[0][id_col]
    acceptable_stability = (best_id == best_S_id) or (best_id == best_R_id)

    meta = {
        "weights": w,
        "f_star": f_star,
        "f_minus": f_minus,
        "v": float(v),
        "acceptable_advantage": bool(acceptable_advantage),
        "acceptable_stability": bool(acceptable_stability),
        "best_id": best_id,
    }
    return out.sort_values("VIKOR_Q"), meta

# ===============================
# UI Bits
# ===============================
def weight_inputs(criteria):
    st.subheader("Step 3 — Set Weights")
    cols = st.columns(min(4, len(criteria)))
    weights = {}
    for i, c in enumerate(criteria):
        with cols[i % len(cols)]:
            weights[c] = st.number_input(f"Weight: {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")
    st.caption(f"Total weight (auto-normalized): **{sum(weights.values()):.2f}**")
    return weights

def benefit_cost_inputs(criteria, defaults):
    st.subheader("Step 2 — Mark Benefit or Cost")
    cols = st.columns(min(4, len(criteria)))
    bc = {}
    for i, c in enumerate(criteria):
        with cols[i % len(cols)]:
            idx = 0 if defaults.get(c, True) else 1
            choice = st.radio(f"{c}", ["Benefit (higher is better)", "Cost (lower is better)"], index=idx, key=f"bc_{c}")
            bc[c] = choice.startswith("Benefit")
    return bc

def chart_vikor(df, id_col):
    chart = (
        alt.Chart(df.sort_values("VIKOR_Q", ascending=True))
        .mark_bar()
        .encode(
            x=alt.X("VIKOR_Q:Q", title="VIKOR Q (lower is better)"),
            y=alt.Y(f"{id_col}:N", sort='-x', title=id_col),
            tooltip=[alt.Tooltip(id_col, title=id_col), "VIKOR_S:Q", "VIKOR_R:Q", "VIKOR_Q:Q", "VIKOR_Rank:Q"]
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

def download_csv(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name=filename, mime="text/csv")

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("📈 StockXplore • VIKOR")
    st.caption(APP_TAGLINE)
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1) Upload your dataset or use the sample.\n"
                "2) Pick an ID column and numeric criteria.\n"
                "3) Set weights and mark Benefit/Cost.\n"
                "4) Run VIKOR, review charts, and download results.")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Built by undergraduates to power smarter decisions with VIKOR.")

# ===============================
# Main
# ===============================
st.title(APP_TITLE)
st.write(APP_TAGLINE)

# Step 1 — Data
st.header("Step 1 — Load Data")
uploaded = st.file_uploader("Upload CSV (rows=alternatives, columns=criteria)", type=["csv"])

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, encoding="latin1")
    st.success("Dataset loaded.")
else:
    st.info("No file uploaded. Using sample dataset (10 hypothetical stocks, 7 criteria).")
    df_raw = load_sample_data()

st.dataframe(df_raw.head(20), use_container_width=True)

# Choose ID and criteria
all_cols = list(df_raw.columns)
id_col = st.selectbox("Select ID column (e.g., Ticker or Name)", options=all_cols, index=0)

# Only numeric columns for criteria (after coercion)
num_candidates = [c for c in all_cols if c != id_col]
df_num = coerce_numeric(df_raw.copy(), num_candidates)
numeric_cols = [c for c in num_candidates if pd.api.types.is_numeric_dtype(df_num[c])]

criteria = st.multiselect("Select numeric criteria", options=numeric_cols, default=numeric_cols)

if len(criteria) == 0:
    st.warning("Please select at least one criterion to proceed.")
    st.stop()

# Drop rows with NaN in selected criteria
before = len(df_num)
df_num = df_num.dropna(subset=criteria)
after = len(df_num)
if after < before:
    st.caption(f"Dropped **{before - after}** rows with missing/invalid numeric values in selected criteria.")

# Step 2 — Benefit/Cost
default_benefit = {c: (c.upper() not in ["PE", "PTBV"]) for c in criteria}
benefit_flags = benefit_cost_inputs(criteria, defaults=default_benefit)

# Step 3 — Weights
weights = weight_inputs(criteria)

# Step 4 — VIKOR controls
st.subheader("Step 4 — VIKOR Parameter")
v_param = st.slider("VIKOR v (strategy of majority)", 0.0, 1.0, 0.5, 0.05)

st.markdown("---")
run = st.button("🚀 Run VIKOR")

if run:
    try:
        # Prepare working frame
        df_work = df_num[[id_col] + criteria].copy()
        # Compute
        vikor_df, meta = vikor(df_work, id_col, criteria, weights, benefit_flags, v=v_param)

        st.subheader("VIKOR Results")
        st.dataframe(vikor_df[[id_col, "VIKOR_S", "VIKOR_R", "VIKOR_Q", "VIKOR_Rank"]], use_container_width=True)

        # Chart
        chart_vikor(vikor_df[[id_col, "VIKOR_S", "VIKOR_R", "VIKOR_Q", "VIKOR_Rank"]].copy(), id_col)

        # Compromise info
        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("**Compromise Solution Check**")
            st.json({
                "best_id": meta["best_id"],
                "acceptable_advantage": meta["acceptable_advantage"],
                "acceptable_stability": meta["acceptable_stability"],
                "note": "Accepted" if (meta["acceptable_advantage"] and meta["acceptable_stability"]) else "Check conditions",
                "v": meta["v"]
            })
        with c2:
            st.markdown("**Normalized Weights**")
            st.write({c: float(w) for c, w in zip(criteria, meta["weights"])})

        # Simple correlations (if user wants to compare S/R/Q)
        with st.expander("🔎 Diagnostics"):
            try:
                sr = pd.Series(vikor_df["VIKOR_S"]).rank()
                rr = pd.Series(vikor_df["VIKOR_R"]).rank()
                qr = pd.Series(vikor_df["VIKOR_Q"]).rank()
                spearman_SR = sr.corr(rr, method="spearman")
                spearman_SQ = sr.corr(qr, method="spearman")
                spearman_RQ = rr.corr(qr, method="spearman")
                st.write({
                    "Spearman(S,R)": round(float(spearman_SR), 3),
                    "Spearman(S,Q)": round(float(spearman_SQ), 3),
                    "Spearman(R,Q)": round(float(spearman_RQ), 3),
                })
            except Exception as e:
                st.info(f"Diagnostics unavailable: {e}")

        # Download
        download_csv(vikor_df, "vikor_results.csv")

    except Exception as e:
        st.error(f"❌ Error running VIKOR: {e}")

# Footer
st.markdown("---")
st.caption("© 2025 StockXplore • VIKOR • Streamlit")
