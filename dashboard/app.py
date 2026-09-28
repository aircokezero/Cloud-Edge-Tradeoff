"""
Experiment 8 — Dashboard, Responsible AI Reporting & Final Portfolio
Cloud-Edge Tradeoff: Adaptive Workload Placement

Run locally:
    streamlit run app.py

Deploy: push this folder to GitHub, then connect it at share.streamlit.io
(see README.md for the full walkthrough).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Cloud-Edge Tradeoff Dashboard", layout="wide")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
SAMPLE_DIR = BASE_DIR / "sample_data"

TRACKS = {
    "Edge (IIoT)": {
        "key": "edge",
        "model_path": MODELS_DIR / "edge_efficiency_best_model.pkl",
        "metrics_path": METRICS_DIR / "edge_metrics.csv",
        "reference_path": SAMPLE_DIR / "iiot_reference_sample.csv",
        "target": "Edge_Efficiency_Score",
        "input_insight": (
            "The edge model uses machine and network conditions to estimate the efficiency score. "
            "In the current model, `Network_Latency` and `Edge_Processing_Time` have the strongest "
            "influence: higher values generally lower the prediction. Higher `Vibration` and "
            "`Temperature`, and `Predicted_Failure` = 1, also tend to lower it. `Pressure`, "
            "`Fuzzy_PID_Output`, and `Maintenance_Status` have comparatively little influence "
            "in the current model. These are learned associations, not causal guarantees."
        ),
        # Static insight written from the current model's SHAP summary plot.
        # Re-check and update this text if the model is retrained.
        "shap_insight": (
            "**Key insights — Edge efficiency model**\n\n"
            "- **Latency and processing time drive the score.** `Edge_Processing_Time` and "
            "`Network_Latency` have by far the widest spread (roughly \u221219 to +15 points). "
            "Low values raise predicted efficiency; high values lower it the most.\n"
            "- **Machine condition is the second tier.** Higher `Vibration` and `Temperature` "
            "pull the score down (up to about \u22127 and \u22125 points); lower values push it up.\n"
            "- **A predicted failure costs efficiency.** `Predicted_Failure` = 1 lowers the score "
            "by about 3 points, while 0 gives a small lift of about +1.\n"
            "- **Little influence:** `Fuzzy_PID_Output`, `Maintenance_Status` and `Pressure` sit "
            "close to zero.\n\n"
            "**Actionable takeaway:** cutting edge processing time and network latency is the "
            "biggest lever for efficiency; vibration and temperature are useful early-warning signals."
        ),
        # Static insight written from the current Experiment 4 metrics table.
        # Re-check and update this text if the metrics file is re-exported.
        "metrics_insight": (
            "**Key insights — Edge efficiency model comparison**\n\n"
            "- **Best overall: `GradientBoosting_tuned`.** It has the lowest RMSE (2.587) and the "
            "highest R\u00b2 (0.9897), so it makes the fewest large errors in predicting "
            "`Edge_Efficiency_Score`.\n"
            "- **Tuning helped Gradient Boosting, not Random Forest.** Tuning cut Gradient "
            "Boosting's RMSE by about 2.3% and its MAE by about 8.8%. For Random Forest, RMSE "
            "rose slightly (2.663 \u2192 2.673) and MAE fell slightly (0.776 \u2192 0.769), which "
            "is effectively no change.\n"
            "- **The two families make different kinds of error.** The Random Forest variants have "
            "the lowest MAE (about 0.77 vs 0.90), so they are closer on typical rows, while tuned "
            "Gradient Boosting has smaller worst-case misses. RMSE is roughly 3x MAE for every "
            "model, which means a few large errors dominate the RMSE.\n"
            "- **All four models are close.** R\u00b2 stays between 0.989 and 0.990 and RMSE is "
            "within about 3% across models, so the choice matters less than the quality of the "
            "input data.\n\n"
            "**Takeaway:** use `GradientBoosting_tuned` when large misses are costly; the "
            "Random Forest models are a reasonable alternative if typical-case accuracy matters more."
        ),
        "numeric_features": [
            "Temperature", "Pressure", "Vibration",
            "Network_Latency", "Edge_Processing_Time", "Fuzzy_PID_Output",
        ],
        "categorical_features": {
            "Maintenance_Status": ["Normal", "Warning", "Failure"],
            "Predicted_Failure": [0, 1],
        },
    },
    "Infrastructure (Cloud/Pod)": {
        "key": "infra",
        "model_path": MODELS_DIR / "infra_efficiency_best_model.pkl",
        "metrics_path": METRICS_DIR / "infra_metrics.csv",
        "reference_path": SAMPLE_DIR / "telemetry_reference_sample.csv",
        "target": "Infrastructure_Efficiency_Score",
        "input_insight": (
            "The infrastructure model uses host/pod telemetry to estimate efficiency. In the "
            "current model, `CPU (%)` and `Energy (watts)` have the strongest influence: higher "
            "values generally lower the prediction. Higher `MEM (%)` and `MEM (B)` also tend to "
            "lower it. `Telemetry_Type` has a modest influence, while `Scenario`, network "
            "throughput (`rx`/`tx`), filesystem use (`fs (%)`), and `Pod_Status` have comparatively "
            "little influence. These are learned associations, not causal guarantees."
        ),
        # Static insight written from the current model's SHAP summary plot.
        # Re-check and update this text if the model is retrained.
        "shap_insight": (
            "**Key insights — Infrastructure efficiency model**\n\n"
            "- **Energy and CPU dominate.** High `Energy (watts)` lowers the score by roughly "
            "12\u201316 points and low energy raises it by about 8\u201315. `CPU (%)` behaves the "
            "same way, and the heaviest CPU loads carry the largest penalty (down to about \u221227).\n"
            "- **Memory is the next factor.** High `MEM (B)` and `MEM (%)` reduce the score "
            "(up to about \u22128 and \u22123 points); low memory use gives a small lift.\n"
            "- **Telemetry type matters modestly.** `Telemetry_Type` = Pod adds about +2 points and "
            "Node subtracts about 2.\n"
            "- **Little influence:** `Scenario`, `rx (B/sec)`, `fs (%)`, `tx (B/sec)` and "
            "`Pod_Status` are close to zero.\n\n"
            "**Actionable takeaway:** workloads with high energy draw, CPU and memory are the least "
            "efficient, so they are the first candidates for rebalancing or offloading."
        ),
        # Static insight written from the current Experiment 4 metrics table.
        # Re-check and update this text if the metrics file is re-exported.
        "metrics_insight": (
            "**Key insights — Infrastructure efficiency model comparison**\n\n"
            "- **Best overall: `GradientBoosting_tuned`.** It has the lowest RMSE (2.587) and the "
            "highest R\u00b2 (0.9897), so it makes the fewest large errors in predicting "
            "`Infrastructure_Efficiency_Score`.\n"
            "- **Tuning helped Gradient Boosting, not Random Forest.** Tuning cut Gradient "
            "Boosting's RMSE by about 2.3% and its MAE by about 8.8%. For Random Forest, RMSE "
            "rose slightly (2.663 \u2192 2.673) and MAE fell slightly (0.776 \u2192 0.769), which "
            "is effectively no change.\n"
            "- **The two families make different kinds of error.** The Random Forest variants have "
            "the lowest MAE (about 0.77 vs 0.90), so they are closer on typical rows, while tuned "
            "Gradient Boosting has smaller worst-case misses. RMSE is roughly 3x MAE for every "
            "model, which means a few large errors dominate the RMSE.\n"
            "- **All four models are close.** R\u00b2 stays between 0.989 and 0.990 and RMSE is "
            "within about 3% across models, so the choice matters less than the quality of the "
            "input data.\n\n"
            "**Takeaway:** use `GradientBoosting_tuned` when large misses are costly; the "
            "Random Forest models are a reasonable alternative if typical-case accuracy matters more."
        ),
        "numeric_features": [
            "CPU (%)", "Energy (watts)", "MEM (%)", "fs (%)",
            "rx (B/sec)", "tx (B/sec)", "MEM (B)",
        ],
        "categorical_features": {
            "Telemetry_Type": ["Node", "Pod"],
            "Pod_Status": ["Pods On", "Pods Off"],
            "Scenario": ["Node_Pods_On", "Node_Pods_Off", "Pod_Pods_On", "Pod_Pods_Off"],
        },
    },
}


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def make_shap_inputs(model, data: pd.DataFrame, categorical_cols):
    """Make a pipeline with text columns usable by SHAP.

    SHAP's tabular masker does numeric arithmetic on the data, so string columns
    (e.g. Maintenance_Status) crash it. Fix: encode each categorical column as
    integer codes for SHAP, and wrap model.predict so it decodes the codes back
    to the original category values (in a named DataFrame) before predicting.

    Returns (encoded_data, predict_fn).
    """
    encoded = data.copy()
    categories = {}
    for col in categorical_cols:
        cats = sorted(encoded[col].unique())
        categories[col] = np.array(cats)
        code_map = {v: i for i, v in enumerate(cats)}
        encoded[col] = encoded[col].map(code_map).astype(float)

    columns = list(encoded.columns)

    def predict_fn(X):
        frame = pd.DataFrame(np.asarray(X, dtype=float), columns=columns)
        for col, cats in categories.items():
            idx = np.clip(np.rint(frame[col].to_numpy()).astype(int), 0, len(cats) - 1)
            frame[col] = cats[idx]
        return model.predict(frame)

    return encoded, predict_fn


@st.cache_data(show_spinner=False)
def compute_shap(track_key: str, _model, _reference_df, feature_cols, categorical_cols):
    available = _reference_df[list(feature_cols)].dropna()
    encoded, predict_fn = make_shap_inputs(_model, available, categorical_cols)
    background = encoded.sample(min(50, len(encoded)), random_state=42)
    display_sample = encoded.sample(min(100, len(encoded)), random_state=1)
    # Model-agnostic explainer works regardless of which algorithm Experiment 4
    # selected (tree ensemble, linear, or KNN) now that inputs are all numeric.
    explainer = shap.Explainer(predict_fn, background)
    return explainer(display_sample), display_sample


def load_reference(t: dict):
    """Load a track's reference sample with numeric NaNs filled as 0.

    Telemetry has structural NaNs (node rows lack pod fields and vice versa).
    Experiment 4 trained with these filled as 0, so do the same here; otherwise
    dropna() in the SHAP tab would discard every row on the infra track.
    """
    df = load_csv(t["reference_path"])
    if df is None:
        return None
    df = df.copy()
    num = [c for c in t["numeric_features"] if c in df.columns]
    df[num] = df[num].fillna(0)
    return df


def track_inputs(t: dict, ref, prefix: str) -> dict:
    """Render number/select widgets for one track and return the input row."""
    values = {}
    for feat in t["numeric_features"]:
        default = float(ref[feat].median()) if ref is not None and feat in ref.columns else 0.0
        values[feat] = st.number_input(feat, value=default, key=f"{prefix}_{t['key']}_{feat}")
    for feat, options in t["categorical_features"].items():
        values[feat] = st.selectbox(feat, options, key=f"{prefix}_{t['key']}_{feat}")
    return values


def render_placement_page():
    """Predict both efficiency scores, compare them, and recommend a platform."""
    edge_t = TRACKS["Edge (IIoT)"]
    infra_t = TRACKS["Infrastructure (Cloud/Pod)"]

    st.title("Preferred Computing Platform")
    st.caption(
        "This project explores adaptive workload placement by comparing predicted efficiency "
        "for edge IIoT and cloud/pod infrastructure. Enter the conditions for each platform; "
        "the dashboard predicts both scores and recommends the higher one."
    )

    edge_model = load_model(edge_t["model_path"])
    infra_model = load_model(infra_t["model_path"])
    missing = [
        str(t["model_path"].relative_to(BASE_DIR))
        for t, m in ((edge_t, edge_model), (infra_t, infra_model)) if m is None
    ]
    if missing:
        st.warning("Model file(s) not found: " + ", ".join(f"`{p}`" for p in missing))
        return

    tie_pct = st.slider(
        "Tie margin (%)", 0.0, 20.0, 2.0, 0.5,
        help="If the two scores differ by no more than this percentage of the larger "
             "score, the platforms are reported as comparable instead of picking a winner.",
    )

    with st.form("placement_form"):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Edge (IIoT) conditions")
            edge_inputs = track_inputs(edge_t, load_reference(edge_t), "place")
        with right:
            st.markdown("#### Infrastructure (Cloud/Pod) conditions")
            infra_inputs = track_inputs(infra_t, load_reference(infra_t), "place")
        submitted = st.form_submit_button("Compare platforms")

    if not submitted:
        st.info("Fill in both sets of conditions and press **Compare platforms**.")
        return

    try:
        edge_score = float(edge_model.predict(pd.DataFrame([edge_inputs]))[0])
        infra_score = float(infra_model.predict(pd.DataFrame([infra_inputs]))[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    gap = edge_score - infra_score
    gap_pct = abs(gap) / max(abs(edge_score), abs(infra_score), 1e-9) * 100

    if gap_pct <= tie_pct:
        st.info(f"\u2696\uFE0F **Comparable** — scores are within {gap_pct:.1f}% of each other.")
    elif gap > 0:
        st.success(f"\u2705 **Preferred platform: Edge** — scores {gap_pct:.1f}% higher than Cloud/Pod.")
    else:
        st.success(f"\u2705 **Preferred platform: Cloud/Pod infrastructure** — scores {gap_pct:.1f}% higher than Edge.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Edge efficiency", f"{edge_score:.2f}")
    c2.metric("Infrastructure efficiency", f"{infra_score:.2f}")
    c3.metric("Edge minus Infra", f"{gap:+.2f}")

    st.bar_chart(
        pd.DataFrame({"Predicted efficiency": [edge_score, infra_score]},
                     index=["Edge", "Infrastructure"])
    )
    st.caption(
        "Assumes a higher score is better and that the two scores are on a comparable "
        "scale. If the models were trained on differently scaled targets, treat the "
        "comparison as indicative only."
    )


st.sidebar.title("Cloud-Edge Tradeoff")
st.sidebar.caption("Adaptive Workload Placement — Dashboard")
page = st.sidebar.radio("View", ["Track Analysis", "Placement Recommendation"])
if page == "Placement Recommendation":
    render_placement_page()
    st.stop()

track_label = st.sidebar.radio("Track", list(TRACKS.keys()))
track = TRACKS[track_label]

model = load_model(track["model_path"])
reference_df = load_reference(track)
metrics_df = load_csv(track["metrics_path"])

st.title("Adaptive Workload Placement — Efficiency Dashboard")
st.caption(f"Track: {track_label}")

feature_cols = track["numeric_features"] + list(track["categorical_features"].keys())

tab_predict, tab_shap, tab_metrics, tab_drift = st.tabs(
    ["\U0001F52E Predict", "\U0001F9E0 Explainability (SHAP)", "\U0001F4CA Model Metrics", "\U0001F4C8 Data Drift"]
)

# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader(f"Predict {track['target']}")
    st.markdown(
        "Provide the workload measurements and categories below. Numeric inputs represent "
        "sensor or resource usage; category inputs describe machine status, failure state, "
        "or telemetry context. The trained model combines them to estimate the efficiency score."
    )
    with st.expander("How these inputs influence the prediction"):
        st.markdown(track["input_insight"])
    if model is None:
        st.warning(
            f"Model file not found at `{track['model_path'].relative_to(BASE_DIR)}`. "
            "Place the corresponding .pkl to enable predictions."
        )
    else:
        with st.form(f"predict_form_{track['key']}"):
            cols = st.columns(2)
            input_values = {}
            for i, feat in enumerate(track["numeric_features"]):
                default = (
                    float(reference_df[feat].median())
                    if reference_df is not None and feat in reference_df.columns
                    else 0.0
                )
                input_values[feat] = cols[i % 2].number_input(
                    feat, value=default, key=f"{track['key']}_{feat}"
                )
            for i, (feat, options) in enumerate(track["categorical_features"].items()):
                input_values[feat] = cols[i % 2].selectbox(
                    feat, options, key=f"{track['key']}_{feat}"
                )
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = pd.DataFrame([input_values])
            try:
                prediction = model.predict(row)[0]
                st.success(f"Predicted {track['target']}: **{prediction:.2f}**")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------------------------
# SHAP tab
# ---------------------------------------------------------------------------
with tab_shap:
    st.subheader("Global Feature Importance (SHAP)")
    st.markdown(
        "SHAP estimates how each input moves model predictions relative to a baseline. "
        "A positive contribution raises the predicted score; a negative contribution lowers it. "
        "The size and spread show model influence, not proof of cause and effect."
    )
    if model is None:
        st.warning("Model not loaded — see the Predict tab for setup instructions.")
    elif reference_df is None:
        st.warning(
            f"Reference sample not found at `{track['reference_path'].relative_to(BASE_DIR)}`. "
            "Needed as SHAP background/display data — see README.md for how to export it."
        )
    else:
        st.info(track["shap_insight"])
        with st.spinner("Computing SHAP values..."):
            shap_values, display_sample = compute_shap(
                track["key"], model, reference_df,
                tuple(feature_cols), tuple(track["categorical_features"].keys()),
            )

        fig = plt.figure()
        shap.summary_plot(shap_values, display_sample, show=False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Each dot is one sample; horizontal position shows how much that feature moved "
            "the predicted efficiency score. Wider spread indicates a stronger influence. "
            "Categorical features are shown as integer codes (alphabetical order of their "
            "categories) for plotting purposes only — the model itself sees the real labels."
        )
        st.caption(
            "SHAP describes what the model relies on, not proven cause and effect, and "
            "these notes reflect the current trained models."
        )

# ---------------------------------------------------------------------------
# Metrics tab
# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Baseline vs. Tuned Model Comparison")
    st.markdown(
        "These metrics summarize prediction error on the model evaluation data. Lower **RMSE** "
        "and **MAE** indicate more accurate predictions; RMSE penalizes large misses more heavily. "
        "Higher **R²** indicates that the model explains more variation in the target."
    )
    if metrics_df is None:
        st.warning(
            f"Metrics file not found at `{track['metrics_path'].relative_to(BASE_DIR)}`. "
            "Export edge_all_results / infra_all_results as CSV — see README.md."
        )
    else:
        display_df = metrics_df.copy()
        has_all = all(c in display_df.columns for c in ["RMSE", "MAE", "R2"])
        if has_all:
            styled = display_df.style.highlight_min(
                subset=["RMSE", "MAE"], color="#c6f6d5"
            ).highlight_max(subset=["R2"], color="#c6f6d5")
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)

        # Auto-generated summary of the best model, computed from the metrics file
        # so it stays correct if Experiment 4 results are re-exported.
        if has_all:
            scored = display_df.copy()
            for _c in ("RMSE", "MAE", "R2"):
                scored[_c] = pd.to_numeric(scored[_c], errors="coerce")
            scored = scored.dropna(subset=["RMSE", "MAE", "R2"])
            if not scored.empty:
                best_idx = scored["RMSE"].idxmin()
                best = scored.loc[best_idx]
                label_cols = [c for c in scored.columns if c not in ("RMSE", "MAE", "R2")][:2]
                label = " / ".join(str(best[c]) for c in label_cols) or f"row {best_idx}"

                st.success(f"\U0001F3C6 **Best model (lowest RMSE): {label}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{best['RMSE']:.3f}")
                m2.metric("MAE", f"{best['MAE']:.3f}")
                m3.metric("R\u00b2", f"{best['R2']:.3f}")

                st.info(track["metrics_insight"])

# ---------------------------------------------------------------------------
# Drift tab
# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Data Drift Check")
    st.markdown(
        "Upload a new batch of data in the same column format to check whether its "
        "feature distributions have drifted from the training-time reference sample."
    )
    st.caption(
        "Numeric features are compared with a Kolmogorov-Smirnov test and categorical features "
        "with a chi-square test. A p-value below 0.05 flags a distribution difference; this is "
        "a statistical signal, not by itself evidence that model accuracy has fallen."
    )
    uploaded = st.file_uploader("Upload CSV", type="csv", key=f"drift_{track['key']}")

    if reference_df is None:
        st.warning(
            f"Reference sample not found at `{track['reference_path'].relative_to(BASE_DIR)}`. "
            "Needed to compare against — see README.md."
        )
    elif uploaded is not None:
        current_df = pd.read_csv(uploaded)
        _cur_num = [c for c in track["numeric_features"] if c in current_df.columns]
        current_df[_cur_num] = current_df[_cur_num].fillna(0)
        rows = []

        for feat in track["numeric_features"]:
            if feat in current_df.columns and feat in reference_df.columns:
                stat, p_value = stats.ks_2samp(
                    reference_df[feat].dropna(), current_df[feat].dropna()
                )
                rows.append({
                    "feature": feat, "type": "numeric", "test": "Kolmogorov-Smirnov",
                    "p_value": p_value, "drift_detected": p_value < 0.05,
                })

        for feat in track["categorical_features"]:
            if feat in current_df.columns and feat in reference_df.columns:
                ref_counts = reference_df[feat].value_counts()
                cur_counts = current_df[feat].value_counts()
                categories = sorted(set(ref_counts.index) | set(cur_counts.index))
                ref_freq = [ref_counts.get(c, 0) for c in categories]
                cur_freq = [cur_counts.get(c, 0) for c in categories]
                try:
                    _, p_value, _, _ = stats.chi2_contingency([ref_freq, cur_freq])
                except ValueError:
                    p_value = float("nan")
                rows.append({
                    "feature": feat, "type": "categorical", "test": "Chi-square",
                    "p_value": p_value,
                    "drift_detected": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                })

        drift_df = pd.DataFrame(rows)
        st.dataframe(drift_df, use_container_width=True)

        n_drifted = int(drift_df["drift_detected"].sum()) if not drift_df.empty else 0
        if n_drifted > 0:
            st.error(f"\u26A0\uFE0F {n_drifted} feature(s) show statistically significant drift (p < 0.05).")
        else:
            st.success("\u2705 No statistically significant drift detected across checked features.")
    else:
        st.info("Waiting for a CSV upload.")