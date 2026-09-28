# Experiment 8 — Dashboard, Responsible AI Reporting & Final Portfolio

A Streamlit dashboard serving the Experiment 4 models, with live SHAP
explainability, a model-comparison table, and an on-demand data-drift check.

**Windows / Git CMD instructions** — same convention as Experiments 6 and 7.

## What's in this folder

```
dashboard\
├── app.py                          <- the Streamlit app
├── requirements.txt
├── models\
│   ├── edge_efficiency_best_model.pkl     <- from Experiment 4
│   └── infra_efficiency_best_model.pkl    <- from Experiment 4
├── metrics\
│   ├── edge_metrics.csv             <- export from Experiment 4 (see below)
│   └── infra_metrics.csv            <- export from Experiment 4 (see below)
└── sample_data\
    ├── iiot_reference_sample.csv        <- small export from Experiment 3/4 data
    └── telemetry_reference_sample.csv   <- small export from Experiment 3/4 data
```

Four artifacts need to come from your earlier notebooks before this runs
with real data — the app runs and shows clear warnings without them, but
every tab needs its file to actually do anything.

## 1. Export the two model files (from Experiment 4)

Already have these from Experiment 6 — just copy them here too:
```
edge_efficiency_best_model.pkl
infra_efficiency_best_model.pkl
```

## 2. Export the metrics CSVs (from Experiment 4)

Add this cell at the end of the Experiment 4 notebook, right after
`edge_all_results`/`infra_all_results` are built in Section 7:

```python
edge_all_results.reset_index().rename(columns={"index": "Model"}).to_csv(
    "edge_metrics.csv", index=False)
infra_all_results.reset_index().rename(columns={"index": "Model"}).to_csv(
    "infra_metrics.csv", index=False)

from google.colab import files
files.download("edge_metrics.csv")
files.download("infra_metrics.csv")
```

Place both downloaded files in `dashboard\metrics\`.

## 3. Export the reference samples (from Experiment 3, where `iiot`/`telemetry` are loaded)

Add this cell in the Experiment 3 (or 4) notebook, after `iiot` and
`telemetry` are loaded:

```python
iiot.sample(min(500, len(iiot)), random_state=42).to_csv(
    "iiot_reference_sample.csv", index=False)
telemetry.sample(min(500, len(telemetry)), random_state=42).to_csv(
    "telemetry_reference_sample.csv", index=False)

from google.colab import files
files.download("iiot_reference_sample.csv")
files.download("telemetry_reference_sample.csv")
```

Place both downloaded files in `dashboard\sample_data\`. These 500-row
samples serve two purposes in the app: SHAP background/display data, and
the reference distribution the Data Drift tab compares uploads against.
They're small on purpose — no need for the full dataset here, and it keeps
the deployed app fast and the repo light.

**Note:** these samples contain your actual telemetry data (not PII, but
still your project's data) and, unlike the `.pkl` models, are small enough
that committing them directly to git is reasonable — no DVC/gitignore
concerns here given the size.

## 4. Run locally first (Git CMD)

```
cd dashboard
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser tab automatically (usually `http://localhost:8501`).
Check all four tabs — Predict, Explainability, Model Metrics, Data Drift —
with both tracks selected in the sidebar, before deploying.

## 5. Deploy to Streamlit Community Cloud (for the public link deliverable)

1. Push this whole `dashboard\` folder (with the four artifacts in place)
   to your GitHub repo's `main` branch.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → select your repo → set:
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. Click **Deploy**. First deploy takes a few minutes while it installs
   `requirements.txt`.
5. Once live, copy the generated URL
   (`https://<something>.streamlit.app`) — that's your **Streamlit app
   link** deliverable.

### If the deploy fails on package versions
Streamlit Cloud runs on Linux with a specific Python version it picks
automatically. If `requirements.txt`'s pins don't have a matching wheel
(same class of issue as the pandas/numpy problem from Experiment 6), relax
the pins the same way:
```
streamlit>=1.39.0
pandas>=2.2.3
numpy>=2.0
scikit-learn>=1.5.2
joblib>=1.4.2
shap>=0.46.0
matplotlib>=3.9.2
scipy>=1.14.1
```

### One thing worth deciding before deploying: are the .pkl files going to GitHub?
Streamlit Cloud only sees what's in your GitHub repo — it can't reach your
local machine or Google Drive. So whatever you decided earlier about
`models/*.pkl` (commit directly vs. DVC-track) determines what you need to
do here:
- **If committed directly to git:** nothing extra needed, they deploy
  alongside the code.
- **If DVC-tracked/gitignored:** Streamlit Cloud won't have them unless you
  either (a) commit them to this repo specifically despite the general
  policy — reasonable if they're small, since a deployed dashboard can't
  run `dvc pull` on its own without extra setup — or (b) add a
  DVC-pull step to `app.py`'s startup using `st.secrets` for the DagsHub
  token (Streamlit Cloud's equivalent of a GitHub Actions secret, set via
  the app's **Settings → Secrets** in the Streamlit Cloud dashboard). Option
  (a) is simpler and is what this README assumes by default.

## 6. Run the standalone SHAP analysis (both regression models, explicitly)

The dashboard's Explainability tab computes SHAP live for whichever one
track you happen to have selected — good for interactive exploration, but
not a fixed artifact you can point to in a report. For that, run:

```
cd dashboard
call .venv\Scripts\activate.bat
python shap_analysis\run_shap_analysis.py
```

This explicitly runs SHAP against **both** `edge_efficiency_best_model.pkl`
and `infra_efficiency_best_model.pkl` in one pass (using the same
model-agnostic `shap.Explainer` approach as the dashboard) and writes to
`dashboard\shap_analysis\output\`:

```
edge_shap_summary.png        <- global SHAP summary plot, Edge track
edge_shap_importance.csv     <- mean |SHAP value| per feature, ranked
infra_shap_summary.png       <- global SHAP summary plot, Infrastructure track
infra_shap_importance.csv    <- mean |SHAP value| per feature, ranked
```

Use these two PNGs directly in the Experiment 8 report's Results section —
they're the concrete "SHAP was conducted for both regression models"
evidence, independent of whether anyone actually clicks through the live
dashboard.

## 7. Testing the Data Drift tab
Use one of your own held-out rows, or quickly make a tiny test CSV with the
same columns as the reference sample — even a handful of rows with
noticeably shifted values (e.g., much higher `Temperature`) is enough to
see the tab flag drift correctly.
