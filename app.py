\
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------
# App Meta
# ------------------------------
APP_TITLE = "StockXplore: Big Data-Powered VIKOR–ARAS System for Smarter Stock Selection"
APP_TAGLINE = "Undergraduates Pioneering Tomorrow’s Breakthroughs — MCDM ranking with VIKOR & ARAS for stock screening."

st.set_page_config(page_title="StockXplore", layout="wide", page_icon="📈")

# ------------------------------
# Utility: Sample Data
# ------------------------------
def load_sample_data():
    np.random.seed(42)
    data = {
        "Ticker": ["AAA","BBB","CCC","DDD","EEE","FFF","GGG","HHH","III","JJJ"],
        "Name": [
            "Alpha Berhad","Beta Holdings","Ceria Corp","Daya Energy","Ekovest",
            "Fajar Tech","Gemilang","Harmoni","Inovasi","Jitu Capital"
        ],
        # Common equity factors (hypothetical values)
        "EPS": np.round(np.random.uniform(0.1, 5.0, 10), 2),       # Benefit
        "DPS": np.round(np.random.uniform(0.0, 2.0, 10), 2),       # Benefit
        "NTA": np.round(np.random.uniform(0.5, 10.0, 10), 2),      # Benefit
        "PE":  np.round(np.random.uniform(5.0, 35.0, 10), 2),      # Cost (lower better)
        "DY":  np.round(np.random.uniform(0.0, 12.0, 10), 2),      # Benefit (%)
        "ROE": np.round(np.random.uniform(0.0, 25.0, 10), 2),      # Benefit (%)
        "PTBV": np.round(np.random.uniform(0.2, 5.0, 10), 2),      # Cost (lower better)
    }
    return pd.DataFrame(data)

# ------------------------------
# MCDM Core
# ------------------------------
def normalize_for_vikor(df, criteria, benefit_flags):
    X = df[criteria].astype(float).values
    X_norm = np.zeros_like(X, dtype=float)
    f_star = np.zeros(len(criteria), dtype=float)  # best
    f_minus = np.zeros(len(criteria), dtype=float) # worst

    for j, c in enumerate(criteria):
        col = X[:, j].astype(float)
        if benefit_flags[c]:
            f_star[j] = np.nanmax(col)
            f_minus[j] = np.nanmin(col)
            # Larger is better -> distance to max
            denom = (f_star[j] - f_minus[j]) if f_star[j] != f_minus[j] else 1.0
            X_norm[:, j] = (f_star[j] - col) / denom
        else:
            f_star[j] = np.nanmin(col)
            f_minus[j] = np.nanmax(col)
            # Smaller is better -> distance to min
            denom = (f_minus[j] - f_star[j]) if f_star[j] != f_minus[j] else 1.0
            X_norm[:, j] = (col - f_star[j]) / denom
    return X_norm, f_star, f_minus

def vikor(df, id_col, criteria, weights, benefit_flags, v=0.5):
    # Normalize distances
    X_norm, f_star, f_minus = normalize_for_vikor(df, criteria, benefit_flags)

    # Apply weights
    w = np.array([weights[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)

    S = (X_norm * w).sum(axis=1)           # group utility (sum of weighted distances)
    R = (X_norm * w).max(axis=1)           # individual regret (max weighted distance)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    # Avoid division by zero
    S_denom = (S_minus - S_star) if S_minus != S_star else 1.0
    R_denom = (R_minus - R_star) if R_minus != R_star else 1.0

    Q = v * (S - S_star) / S_denom + (1 - v) * (R - R_star) / R_denom

    out = df[[id_col]].copy()
    out["VIKOR_S"] = S
    out["VIKOR_R"] = R
    out["VIKOR_Q"] = Q
    out["VIKOR_Rank"] = out["VIKOR_Q"].rank(method="min").astype(int)

    # Determine compromise solution info (VIKOR rules)
    # Sort by Q
    order = out.sort_values("VIKOR_Q", ascending=True).reset_index(drop=True)
    best_Q = order.loc[0, "VIKOR_Q"]
    second_Q = order.loc[1, "VIKOR_Q"] if len(order) > 1 else np.nan
    # Acceptable advantage
    DQ = second_Q - best_Q if not np.isnan(second_Q) else np.nan
    acceptable_advantage = DQ >= (0.5 / (len(df) - 1)) if len(df) > 1 and not np.isnan(DQ) else True
    # Acceptable stability: best also ranks first by S or R
    first_id = order.loc[0, id_col]
    best_S_id = out.sort_values("VIKOR_S").iloc[0][id_col]
    best_R_id = out.sort_values("VIKOR_R").iloc[0][id_col]
    acceptable_stability = (first_id == best_S_id) or (first_id == best_R_id)

    compromise_note = "Accepted" if (acceptable_advantage and acceptable_stability) else "Check conditions"
    return out.sort_values("VIKOR_Q"), {
        "f_star": f_star, "f_minus": f_minus, "weights": w, "v": v,
        "acceptable_advantage": bool(acceptable_advantage),
        "acceptable_stability": bool(acceptable_stability),
        "compromise_note": compromise_note,
        "best_id": first_id
    }

def aras(df, id_col, criteria, weights, benefit_flags):
    X = df[criteria].astype(float).copy()

    # Handle cost criteria by reciprocal transform (1/x) then normalize by column sum
    X_transformed = X.copy().astype(float)
    eps = 1e-9
    for c in criteria:
        if benefit_flags[c]:
            # benefit: keep as is
            pass
        else:
            X_transformed[c] = 1.0 / (X_transformed[c].values + eps)

    # Create ideal alternative a0 (best performance per criterion after transform)
    a0 = {}
    for c in criteria:
        col = X_transformed[c].values
        a0[c] = np.max(col)  # since higher after transform is better

    # Build matrix with a0 on top
    M = pd.concat([pd.DataFrame([a0], columns=criteria), X_transformed.reset_index(drop=True)], ignore_index=True)

    # Normalize by column sums
    col_sums = M.sum(axis=0).replace(0, 1.0)
    N = M / col_sums

    # Apply weights (normalized)
    w = np.array([weights[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)
    W = pd.DataFrame([w], columns=criteria).iloc[0]
    V = N * W.values

    # Compute overall performance score S_i
    S = V.sum(axis=1)

    # Utility degree K_i = S_i / S_0 (with i>=1 alternatives; 0 is ideal)
    S0 = S.iloc[0] if S.iloc[0] != 0 else 1.0
    K = S / S0

    # Prepare output excluding ideal row
    out = df[[id_col]].copy()
    out["ARAS_S"] = S.iloc[1:].values
    out["ARAS_K"] = K.iloc[1:].values
    out["ARAS_Rank"] = out["ARAS_K"].rank(ascending=False, method="min").astype(int)
    meta = {"weights": w, "col_sums": col_sums.to_dict()}
    return out.sort_values("ARAS_Rank"), meta

# ------------------------------
# UI Helpers
# ------------------------------
def weight_inputs(criteria, default_weights=None):
    st.subheader("Step 3 — Set Weights")
    cols = st.columns(min(4, len(criteria)))
    weights = {}
    for idx, c in enumerate(criteria):
        with cols[idx % len(cols)]:
            default = 1.0 if default_weights is None else float(default_weights.get(c, 1.0))
            weights[c] = st.number_input(f"Weight: {c}", min_value=0.0, value=default, step=0.1, key=f"w_{c}")
    total = sum(weights.values())
    st.caption(f"Total weight = **{total:.2f}** (will be normalized automatically).")
    return weights

def benefit_cost_inputs(criteria, defaults=None):
    st.subheader("Step 2 — Choose Benefit/Cost for Each Criterion")
    cols = st.columns(min(4, len(criteria)))
    bc = {}
    for idx, c in enumerate(criteria):
        with cols[idx % len(cols)]:
            default = True if defaults is None else bool(defaults.get(c, True))
            bc[c] = st.radio(f"{c} is …", ["Benefit (higher is better)", "Cost (lower is better)"],
                             index=0 if default else 1, key=f"bc_{c}")
    # Convert to bool map
    return {c: (bc[c].startswith("Benefit")) for c in criteria}

def show_table(df, title="Table"):
    st.markdown(f"### {title}")
    st.dataframe(df, use_container_width=True, hide_index=True)

def download_frame(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=filename, mime="text/csv")

def rank_bar_chart(df, id_col, score_col, method_name):
    # Bar chart sorted by score
    chart = (
        alt.Chart(df.sort_values(score_col, ascending=True))
        .mark_bar()
        .encode(
            x=alt.X(score_col, title=score_col),
            y=alt.Y(id_col, sort='-x', title=id_col),
            tooltip=[id_col, score_col]
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

def dual_rank_chart(vikor_df, aras_df, id_col):
    v = vikor_df[[id_col, "VIKOR_Rank"]].copy().rename(columns={"VIKOR_Rank":"Rank"})
    v["Method"] = "VIKOR"
    a = aras_df[[id_col, "ARAS_Rank"]].copy().rename(columns={"ARAS_Rank":"Rank"})
    a["Method"] = "ARAS"
    comb = pd.concat([v, a], ignore_index=True)

    chart = (
        alt.Chart(comb)
        .mark_circle(size=90)
        .encode(
            x=alt.X("Method:N", title="Method"),
            y=alt.Y("Rank:Q", scale=alt.Scale(domain=(1, comb["Rank"].max()+1), reverse=True), title="Rank (1 = Best)"),
            color="Method:N",
            tooltip=[id_col, "Method", "Rank"]
        )
        .facet(row=alt.Row(f"{id_col}:N", header=alt.Header(labelAngle=0)))
        .properties(spacing=10)
    )
    st.altair_chart(chart, use_container_width=True)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("📈 StockXplore")
    st.caption(APP_TAGLINE)
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1) Upload your stock dataset (CSV) or use the sample.\n"
                "2) Pick an ID column and select criteria.\n"
                "3) Set weights and mark Benefit/Cost.\n"
                "4) Run VIKOR and/or ARAS and compare results.")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Built with ❤️ by undergraduates for future-ready investing using MCDM.")

# ------------------------------
# Main
# ------------------------------
st.title(APP_TITLE)
st.write(APP_TAGLINE)

# Data input
st.markdown("## Step 1 — Data")
uploaded = st.file_uploader("Upload CSV (stocks × criteria)", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="latin1")
    st.success("Dataset loaded.")
else:
    st.info("No file uploaded. Using sample dataset (10 hypothetical stocks with 7 criteria).")
    df = load_sample_data()

# Display preview
st.dataframe(df.head(20), use_container_width=True)

# Select ID column
all_cols = list(df.columns)
id_col = st.selectbox("Select ID column (e.g., Ticker or Name)", options=all_cols, index=0)

# Select criteria columns (numerical only suggestion)
num_cols = [c for c in all_cols if c != id_col and np.issubdtype(df[c].dtype, np.number)]
criteria = st.multiselect("Select criteria (numeric)", options=num_cols, default=num_cols)

if len(criteria) == 0:
    st.warning("Please select at least one criterion to proceed.")
    st.stop()

# Benefit/Cost flags
default_benefit = {c: (c.upper() not in ["PE","PTBV"]) for c in criteria}
benefit_flags = benefit_cost_inputs(criteria, defaults=default_benefit)

# Weights
weights = weight_inputs(criteria)

# Method controls
st.markdown("## Step 4 — Run Methods")
colL, colR = st.columns(2)
with colL:
    run_vikor = st.checkbox("Run VIKOR", value=True)
    v_param = st.slider("VIKOR v (strategy of majority)", 0.0, 1.0, 0.5, 0.05)
with colR:
    run_aras = st.checkbox("Run ARAS", value=True)

tab_results, tab_compare = st.tabs(["Results", "Compare"])

with tab_results:
    if run_vikor:
        st.subheader("VIKOR Results")
        vikor_df, vikor_meta = vikor(df, id_col, criteria, weights, benefit_flags, v=v_param)
        show_table(vikor_df[[id_col,"VIKOR_S","VIKOR_R","VIKOR_Q","VIKOR_Rank"]], "VIKOR Table")
        c1, c2 = st.columns(2)
        with c1:
            rank_bar_chart(vikor_df[[id_col,"VIKOR_Q"]], id_col, "VIKOR_Q", "VIKOR")
        with c2:
            st.markdown("**Compromise Solution Check**")
            st.json({
                "best_id": vikor_meta["best_id"],
                "acceptable_advantage": vikor_meta["acceptable_advantage"],
                "acceptable_stability": vikor_meta["acceptable_stability"],
                "note": vikor_meta["compromise_note"],
                "v": vikor_meta["v"]
            })
        download_frame(vikor_df, "vikor_results.csv")

    if run_aras:
        st.subheader("ARAS Results")
        aras_df, aras_meta = aras(df, id_col, criteria, weights, benefit_flags)
        show_table(aras_df[[id_col,"ARAS_S","ARAS_K","ARAS_Rank"]], "ARAS Table")
        rank_bar_chart(aras_df[[id_col,"ARAS_K"]], id_col, "ARAS_K", "ARAS")
        download_frame(aras_df, "aras_results.csv")

with tab_compare:
    if run_vikor and run_aras:
        vikor_df, _ = vikor(df, id_col, criteria, weights, benefit_flags, v=v_param)
        aras_df, _ = aras(df, id_col, criteria, weights, benefit_flags)
        st.subheader("Ranking Comparison (VIKOR vs ARAS)")
        # Join ranks
        merged = pd.merge(
            vikor_df[[id_col, "VIKOR_Rank"]],
            aras_df[[id_col, "ARAS_Rank"]],
            on=id_col, how="outer"
        )
        st.dataframe(merged.sort_values("VIKOR_Rank"), use_container_width=True, hide_index=True)
        dual_rank_chart(vikor_df, aras_df, id_col)

        # Correlations of ranks (Spearman/Pearson)
        try:
            sp = merged["VIKOR_Rank"].corr(merged["ARAS_Rank"], method="spearman")
            pe = merged["VIKOR_Rank"].corr(merged["ARAS_Rank"], method="pearson")
            st.markdown(f"**Rank Correlations** — Spearman: `{sp:.3f}`, Pearson: `{pe:.3f}`")
        except Exception:
            st.warning("Could not compute correlation (missing or invalid ranks).")

        # RMSE of (scaled) scores if available
        try:
            # scale both to 0-1 for a rough comparison
            vikor_df["Q_scaled"] = (vikor_df["VIKOR_Q"] - vikor_df["VIKOR_Q"].min()) / max(1e-9, (vikor_df["VIKOR_Q"].max() - vikor_df["VIKOR_Q"].min()))
            aras_df["K_scaled"] = (aras_df["ARAS_K"] - aras_df["ARAS_K"].min()) / max(1e-9, (aras_df["ARAS_K"].max() - aras_df["ARAS_K"].min()))
            merged_scores = pd.merge(
                vikor_df[[id_col, "Q_scaled"]],
                aras_df[[id_col, "K_scaled"]],
                on=id_col, how="inner"
            )
            rmse = np.sqrt(np.mean((merged_scores["Q_scaled"] - merged_scores["K_scaled"])**2))
            st.markdown(f"**Score RMSE (scaled)**: `{rmse:.4f}` (lower = closer)")
        except Exception:
            st.warning("Could not compute RMSE of scores.")

        # Download combined
        csv = merged.to_csv(index=False).encode("utf-8")
        st.download_button("Download Comparison (CSV)", data=csv, file_name="comparison_vikor_aras.csv", mime="text/csv")
    else:
        st.info("Run both VIKOR and ARAS to compare rankings.")

# Footer
st.markdown("---")
st.caption("© 2025 StockXplore • VIKOR & ARAS • Streamlit")
