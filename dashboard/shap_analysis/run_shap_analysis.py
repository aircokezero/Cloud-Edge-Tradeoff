"""
Experiment 8 — Standalone SHAP analysis for BOTH Experiment 4 regression models.

This is deliberately separate from the dashboard's live SHAP tab
(dashboard/app.py) — the dashboard computes SHAP on-demand for whichever one
track a user happens to select, which is good for interactive exploration
but doesn't produce a fixed, referenceable artifact. This script explicitly
runs SHAP against BOTH regression models in one pass and saves the results
as PNG files + a CSV of feature-importance rankings, so there's concrete,
reviewable evidence for each model side by side — for the report, not just
the live app.

Run this in Colab (needs both .pkl models + both reference sample CSVs —
see dashboard/README.md for how to export them from Experiments 3/4), or
locally once your venv has the dashboard's requirements installed:

    python shap_analysis/run_shap_analysis.py

Outputs (written to shap_analysis/output/):
    edge_shap_summary.png
    edge_shap_importance.csv
    infra_shap_summary.png
    infra_shap_importance.csv
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).parent.parent  # dashboard/
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRACKS = {
    "edge": {
        "label": "Edge_Efficiency_Score",
        "model_path": MODELS_DIR / "edge_efficiency_best_model.pkl",
        "reference_path": SAMPLE_DIR / "iiot_reference_sample.csv",
        "numeric_cols": [
            "Temperature", "Pressure", "Vibration",
            "Network_Latency", "Edge_Processing_Time", "Fuzzy_PID_Output",
        ],
        "categorical_cols": ["Maintenance_Status", "Predicted_Failure"],
    },
    "infra": {
        "label": "Infrastructure_Efficiency_Score",
        "model_path": MODELS_DIR / "infra_efficiency_best_model.pkl",
        "reference_path": SAMPLE_DIR / "telemetry_reference_sample.csv",
        "numeric_cols": [
            "CPU (%)", "Energy (watts)", "MEM (%)", "fs (%)",
            "rx (B/sec)", "tx (B/sec)", "MEM (B)",
        ],
        "categorical_cols": ["Telemetry_Type", "Pod_Status", "Scenario"],
    },
}


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


def run_shap_for_track(track_key: str, config: dict) -> None:
    print(f"\n=== {track_key.upper()} track — {config['label']} ===")

    if not config["model_path"].exists():
        print(f"  SKIPPED: model not found at {config['model_path']}")
        return
    if not config["reference_path"].exists():
        print(f"  SKIPPED: reference sample not found at {config['reference_path']}")
        return

    model = joblib.load(config["model_path"])
    reference_df = pd.read_csv(config["reference_path"])

    feature_cols = config["numeric_cols"] + config["categorical_cols"]

    # Telemetry has structural NaNs (node rows lack pod fields and vice versa).
    # Experiment 4 trained with numeric NaNs filled as 0, so do the same here;
    # only rows still missing a categorical value after that are dropped.
    subset = reference_df[feature_cols].copy()
    subset[config["numeric_cols"]] = subset[config["numeric_cols"]].fillna(0)
    available = subset.dropna()
    if len(available) < 10:
        print(f"  SKIPPED: only {len(available)} usable rows after dropping NaNs — need more.")
        return

    # Encode categoricals as integer codes so SHAP's masker can do arithmetic on them.
    encoded, predict_fn = make_shap_inputs(model, available, config["categorical_cols"])
    background = encoded.sample(min(50, len(encoded)), random_state=42)
    display_sample = encoded.sample(min(150, len(encoded)), random_state=1)

    print(f"  Computing SHAP values on {len(display_sample)} rows "
          f"(background: {len(background)} rows)...")
    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(display_sample)

    # --- Save the global summary plot ---
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, display_sample, show=False)
    plt.title(f"SHAP Summary — {config['label']} ({track_key} track)")
    plt.tight_layout()
    out_png = OUTPUT_DIR / f"{track_key}_shap_summary.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # --- Save a feature-importance ranking as CSV (mean |SHAP value|) ---
    importance = (
        pd.DataFrame(shap_values.values, columns=feature_cols)
        .abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap_value")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    out_csv = OUTPUT_DIR / f"{track_key}_shap_importance.csv"
    importance.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    print(f"  Top 3 features: {', '.join(importance['feature'].head(3).tolist())}")


def main():
    print("Running SHAP analysis for both Experiment 4 regression models...")
    for track_key, config in TRACKS.items():
        run_shap_for_track(track_key, config)
    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()