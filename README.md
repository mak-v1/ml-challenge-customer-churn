# Telco Customer Churn Modeling

## 1. Problem Summary & Approach
Goal: Predict whether a telecom customer will churn (binary classification) using the public Telco Customer Churn dataset. We emphasized:
- Clean, auditable preprocessing (no silent imputations).
- Preserving semantically meaningful categorical levels (e.g. "No internet service" distinct from "No").
- Imbalance-aware evaluation using PR-AUC and threshold tuning (maximize F1 on validation, then apply to test).
- Comparative baselines: Logistic Regression (linear), Random Forest (tree ensemble), SVM (RBF) to balance interpretability vs. non-linearity.
- Reproducible artifacts: metrics (JSON), reports (TXT), curves (PNG), feature importances, comparison PDF, and an inference API.

## 2. Key EDA & Data Quality Findings
| Topic | Finding | Decision / Rationale |
|-------|---------|----------------------|
| Hidden numeric coercion | `TotalCharges` had whitespace / blank entries (11 rows) all with `tenure == 0`. | Treat as "not yet billable"; drop those rows. |
| Duplicates | No duplicate `customerID`; after removing `customerID`, 42 full-row duplicates remain. | Drop `customerID` then de-duplicate to avoid overweighting identical profiles. |
| Missingness | No true internal NaNs after cleaning; only intentional blanks tied to new customers. | Ensured final processed dataset has zero NaNs. |
| Churn balance | ~26.5% churn (moderate imbalance). | Use stratified splits, PR-AUC focus, threshold tuning. |
| Tri-level service vars | Add-ons had states: Yes / No / No internet service. | Keep all three (do not collapse). |
| Phone/gender fields | Low signal; little churn separation. | Dropped to reduce noise & dimensionality. |
| `TotalCharges` vs engineered | Raw `TotalCharges` noisy + semi-collinear with `tenure * MonthlyCharges`. | Replace with `avg_monthly_spend = TotalCharges / tenure`. |
| Feature engineering | Added `streaming_count` and `security_support_count`. | Compact, non-redundant group signals. |

## 3. Preprocessing Pipeline (see `preprocess.py`)
1. Load raw CSV unchanged (no premature NA coercion).
2. Coerce `TotalCharges` to numeric, identify rows with `tenure == 0` ⇒ drop (11 rows).
3. Engineer: `avg_monthly_spend`, `streaming_count`, `security_support_count`.
4. Binary map: `Churn`, `Partner`, `Dependents`, `PaperlessBilling` to {0,1}.
5. Drop: `TotalCharges` (after deriving spend), `customerID`, `gender`, `PhoneService`, `MultipleLines`.
6. Enforce dtypes; de-duplicate remaining rows.
7. Save to `./data/processed/telco_clean.csv`.

## 4. Modeling Strategy
- Split: 60% train / 20% validation / 20% test (stratified).
- Encoding & scaling:
  - Logistic Regression & SVM: StandardScaler (numerics) + OneHot(drop='first').
  - Random Forest: passthrough numerics + full OneHot (no drop).
- Class imbalance: `class_weight='balanced'` across all models.
- Threshold: tuned on validation probabilities to maximize F1; applied to test.
- Selection metric: PR-AUC (average precision) on validation.
- Explainability: LR coefficients, RF impurity importances, permutation importance (AP scorer) for all.

## 5. Model Performance (Test Set)
(See JSON in `reports/models/<model>/final_test_metrics.json` and comparison PDF.)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Threshold* |
|-------|---------|--------|----------|-----------|--------|----|-----------|
| Logistic Regression | 0.843 | 0.613 | 0.767 | 0.544 | 0.757 | 0.633 | 0.540 |
| CatBoost | 0.844 | 0.641 | 0.773 | 0.554 | 0.754 | 0.638 | 0.528 |
| Random Forest | 0.816 | 0.574 | 0.737 | 0.504 | 0.741 | 0.600 | 0.264 |
| SVM (RBF) | 0.830 | 0.603 | 0.766 | 0.541 | 0.765 | 0.634 | 0.303 |

`*` Threshold is the F1-max point on validation; persisted in JSON and used (default) by the API.

CatBoost now delivers the strongest PR-AUC (0.641) and the highest ROC-AUC (0.844), edging out Logistic Regression while preserving similar recall. Logistic Regression remains a strong, highly interpretable baseline; SVM roughly matches LR performance; Random Forest lags mainly on PR-AUC. Engineering choices (counts, average spend, tri-level preservation) supported both linear separation and boosted / kernel methods.

## 6. Comparison Artifacts
- Individual model artifacts: `reports/models/<lr|rf|svm>/`
  - `baseline_val_metrics.json`, `final_test_metrics.json`
  - `baseline_val_report.txt`, `final_test_report.txt`
  - `feature_importance.csv`
  - `roc.png`, `pr.png`, `model.joblib`
- Cross-model PDF: `reports/model_comparison.pdf` (heatmaps, confusion matrices, bar charts).

## 7. Production Inference (`serve_api.py`)
FastAPI service performing minimal on-the-fly feature engineering replicating training logic:
- Validates presence of raw-required fields (subset of original schema).
- Rejects `tenure <= 0` (non-billable / excluded in training).
- Computes engineered features: `avg_monthly_spend`, `streaming_count`, `security_support_count`.
- Drops `TotalCharges` before inference (aligned with training set). 
- Applies trained logistic regression pipeline (`model.joblib`).
- Uses threshold from metrics JSON unless overridden by `THRESHOLD` environment variable.

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Directory containing `model.joblib` + metrics | `./reports/models/lr` |
| `MODEL_PATH` | Direct path to model (overrides `MODEL_DIR`) | `<MODEL_DIR>/model.joblib` |
| `THRESHOLD` | Decision threshold override | Loaded from JSON or 0.5 |

## 8. Recommended Next Steps
| Area | Recommendation |
|------|---------------|
| Calibration | Evaluate probability calibration (e.g. isotonic) if business thresholds are cost-sensitive. |
| Cost-sensitive metrics | Incorporate churn retention economics (expected value of intervention). |
| Drift monitoring | Track distributions of key categorical proportions & average spend monthly. |
| Feature expansion | Add tenure buckets, recency-of-change features, or interaction terms if justified. |
| Model tuning | Light randomized search for RF/SVM hyperparameters per-perf constraints. |
| Explainability | Add SHAP values for top model. |
| CI/CD | Add tests for API schema validation & regression tests on metrics. |

## 9. How to Run
### Prerequisites
- Python 3.11+ recommended.
- Install dependencies (example minimal list):
```bash
pip install -r requirements.txt
```

If no `requirements.txt` yet, minimal packages:
```
fastapi
uvicorn
pandas
numpy
scikit-learn
matplotlib
joblib
```

### 1. Preprocess Data
```bash
python preprocess.py
```
Outputs: `data/processed/telco_clean.csv`

### 2. Train Models Individually
```bash
python train_lr.py
python train_rf.py
python train_svm.py
```
Artifacts appear under `reports/models/<model_key>/`.

### 3. Generate Comparison PDF
```bash
python compare_models.py
```
Outputs: `reports/model_comparison.pdf`.

### 4. Serve Inference API
```bash
uvicorn serve_api:app --reload --port 8000
```
Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 24,
    "PaperlessBilling": 1,
    "PaymentMethod": "Electronic check",
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "MonthlyCharges": 85.3,
    "TotalCharges": 2040.6
  }'
```
Response:
```json
{
  "churn_probability": 0.62,
  "churn_label": 1,
  "threshold": 0.54,
  "model_version": "model.joblib"
}
```

## 10. Repository Structure (Key Files)
```
preprocess.py                # Data cleaning & feature engineering
train_lr.py / train_rf.py / train_svm.py/ train_catboost.py  # Individual model training scripts
model_common.py              # Shared training utilities
compare_models.py            # PDF comparison report
serve_api.py                 # FastAPI inference service
data/processed/telco_clean.csv
reports/                     # All artifacts (EDA, models, comparison)
```

## 11. Reproducibility Notes
- Deterministic seeds set for model splits & algorithms where applicable.
- Class weights + threshold tuning stabilize performance across reruns.
- All transformations are inside the sklearn pipeline for safe serialization.

## 12. License & Attribution
Dataset: Telco Customer Churn (IBM Sample Dataset). Use consistent with original licensing.

---
Generated as part of a structured ML challenge: auditable preprocessing, transparent modeling, and deployment-ready inference.
