# Responsible AI Report
### Cloud-Edge Tradeoff: Adaptive Workload Placement

## 1. Purpose & Scope

This project trains models to predict two engineered efficiency scores —
`Edge_Efficiency_Score` (IIoT sensor/edge conditions) and
`Infrastructure_Efficiency_Score` (cloud/pod telemetry) — and a
predictive-maintenance classifier (`Predicted_Failure`) from raw machine
sensor data. The intended use is as **decision support** for an
infrastructure/operations team deciding whether to route a workload to the
edge or the cloud, and for flagging machines that may need maintenance
attention — not as a fully autonomous system acting without human review.

## 2. Data Governance & Privacy

- The two datasets (`iiot_with_efficiency.csv`,
  `telemetry_merged_with_efficiency.csv`) consist of **machine and
  infrastructure telemetry** — temperature, vibration, network latency,
  CPU/memory/energy usage, and similar signals. They describe **equipment
  and systems, not individuals** — no personally identifiable information
  (PII) is collected, stored, or processed anywhere in this pipeline.
- Because the data is not about people, the traditional human-subject
  framing of "consent" does not directly apply. The relevant governance
  question instead is **data ownership and access authorization** for the
  source sensor/infrastructure logs.
- Access to the raw datasets is controlled through DVC + a private DagsHub
  remote with token-based authentication (`.dvc/config.local`, gitignored —
  see Experiment 6/7 setup notes). Raw data is never committed to public
  git history; only the small `sample_data/*.csv` reference exports used by
  the Experiment 8 dashboard are committed directly, since they're small
  enough not to warrant DVC tracking and contain no sensitive fields.

## 3. Fairness Audit (Experiment 5)

`Maintenance_Status` was used as a stand-in "sensitive attribute" for the
predictive-maintenance classifier, since the dataset has no demographic
fields — the concern being that the model should not be more or less
reliable depending on a machine's maintenance condition. Key findings:

- The audit found **mild selection-rate disparity** across
  `Maintenance_Status` groups, alongside a notable **accuracy/recall
  disparity** driven substantially by the `"Failure"` group having only
  76 samples, **all** with `Predicted_Failure = 1` (zero negative
  examples).
- This meant Fairlearn's `ThresholdOptimizer` (post-processing mitigation)
  could not be applied to that group at all — it requires both classes
  present per group to build a threshold trade-off curve, which the data
  structurally does not provide.
- `ExponentiatedGradient` with a `DemographicParity` constraint (in-processing
  mitigation) was applied instead, since it only requires per-group
  selection rate, not both classes.
- **Conclusion carried forward:** the disparity for the `"Failure"` group
  is primarily a **data scarcity problem**, not something any mitigation
  algorithm can fully resolve. No fairness technique can validate model
  behavior against ground truth that doesn't exist in the data.

## 4. Explainability

- **Experiment 5** applied SHAP (`TreeExplainer`, global feature
  attribution) and LIME (per-instance local explanation) to the
  predictive-maintenance classifier, including a targeted inspection of
  false-positive/false-negative cases.
- **Experiment 8's dashboard** extends this into an operational tool: the
  Explainability tab computes live SHAP summary plots for whichever
  regression model (edge or infrastructure) is currently deployed, using a
  model-agnostic explainer so it works regardless of which algorithm
  Experiment 4 ultimately selected as "best." A separate standalone script
  (`dashboard/shap_analysis/run_shap_analysis.py`) runs SHAP explicitly
  against **both** regression models in one pass, producing a fixed summary
  plot and ranked feature-importance CSV per model — concrete, reviewable
  evidence for each model rather than something that only exists if someone
  happens to click through the live app.

## 5. Model Limitations

- The regression targets (`Edge_Efficiency_Score`,
  `Infrastructure_Efficiency_Score`) are themselves **engineered formulas**
  over the same raw features the models are trained on. Strong regression
  performance partly reflects the models recovering a known scoring
  formula, not discovering an independent causal relationship — predictions
  should be read as reproducing the documented scoring methodology, not as
  ground truth external validation of it.
- The predictive-maintenance classifier has a **known blind spot** for
  machines already in `"Failure"` maintenance status (Section 3) —
  predictions for that group should get extra human scrutiny rather than
  automated action, since the model's reliability there cannot currently be
  properly audited.
- No models in this project were validated against data from a time period,
  machine population, or infrastructure configuration outside what's in the
  original datasets — performance on genuinely novel conditions is
  unverified.

## 6. Drift Monitoring

- The Experiment 8 dashboard's Data Drift tab lets an operator upload a new
  batch of telemetry and compares its per-feature distribution against the
  original training-time reference sample: Kolmogorov–Smirnov tests for
  numeric features, chi-square tests for categorical features, flagging any
  feature with p < 0.05.
- This is currently a **manual, on-demand** check, not an automated
  production monitoring pipeline. A natural next step is wiring this logic
  into a scheduled job via the Experiment 7 GitHub Actions pipeline, with
  drift results posted somewhere a human reviews regularly (e.g., a
  workflow summary or an issue auto-filed on drift detection).

## 7. Human Oversight & Recommended Use

- This system is intended as **decision support**, not autonomous action —
  a human should review predictions before they drive real workload
  placement or maintenance decisions, especially for the `"Failure"`
  maintenance category flagged in Section 3.
- Recommend periodically re-running the Experiment 5 fairness audit and the
  Experiment 8 drift check as new data accumulates — particularly to check
  whether the `"Failure"` category's sample-size problem has improved
  enough to attempt a full equalized-odds audit (not just demographic
  parity).

## 8. Responsible AI Checklist

- [x] Fairness audit conducted — Section 3
- [x] Explainability tooling provided (SHAP + LIME in Experiment 5; live
      SHAP in the Experiment 8 dashboard) — Section 4
- [x] Model limitations documented — Section 5
- [x] Drift-monitoring tooling provided — Section 6
- [x] No PII / personal data involved — Section 2
- [ ] Automated (non-manual) retraining/monitoring pipeline — not yet
      implemented
- [ ] Full equalized-odds fairness audit for the `"Failure"` maintenance
      category — blocked on data scarcity (Section 3), revisit once more
      samples are available
- [ ] Formal data-ownership/access-authorization documentation — as the datasets 
were extracted from ieee or kaggle, there is no formal documentation for such.
