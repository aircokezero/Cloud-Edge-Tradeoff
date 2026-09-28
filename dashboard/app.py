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


st.sidebar.title("Cloud-Edge Tradeoff")
st.sidebar.caption("Adaptive Workload Placement — Experiment 8 Dashboard")
track_label = st.sidebar.radio("Track", list(TRACKS.keys()))
track = TRACKS[track_label]

model = load_model(track["model_path"])
reference_df = load_csv(track["reference_path"])
if reference_df is not None:
    # Telemetry has structural NaNs (node rows lack pod fields and vice versa).
    # Experiment 4 trained with these filled as 0, so do the same here; otherwise
    # dropna() in the SHAP tab would discard every row on the infra track.
    reference_df = reference_df.copy()
    _num = [c for c in track["numeric_features"] if c in reference_df.columns]
    reference_df[_num] = reference_df[_num].fillna(0)
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
    if model is None:
        st.warning(
            f"Model file not found at `{track['model_path'].relative_to(BASE_DIR)}`. "
            "Place the corresponding .pkl from Experiment 4 there to enable predictions."
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
    if model is None:
        st.warning("Model not loaded — see the Predict tab for setup instructions.")
    elif reference_df is None:
        st.warning(
            f"Reference sample not found at `{track['reference_path'].relative_to(BASE_DIR)}`. "
            "Needed as SHAP background/display data — see README.md for how to export it."
        )
    else:
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

# ---------------------------------------------------------------------------
# Metrics tab
# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Baseline vs. Tuned Model Comparison (from Experiment 4)")
    if metrics_df is None:
        st.warning(
            f"Metrics file not found at `{track['metrics_path'].relative_to(BASE_DIR)}`. "
            "Export edge_all_results / infra_all_results from Experiment 4 as CSV — see README.md."
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

# ---------------------------------------------------------------------------
# Drift tab
# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Data Drift Check")
    st.markdown(
        "Upload a new batch of data in the same column format to check whether its "
        "feature distributions have drifted from the training-time reference sample."
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