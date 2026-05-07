# CreditIQ — End-to-End Credit Risk Scoring System

> Production-grade ML pipeline for credit default prediction with explainability, experiment tracking, drift monitoring, and REST API deployment.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![Evidently](https://img.shields.io/badge/Monitoring-Evidently_AI-purple)

---

## What this project does

CreditIQ takes applicant financial data and:

1. Predicts probability of loan default (binary classification)
2. Explains **why** using SHAP values — which features drove the decision
3. Tracks every experiment with MLflow
4. Monitors model drift in production with Evidently AI
5. Serves predictions via a FastAPI REST endpoint
6. Visualises everything in a Streamlit dashboard

---

## Results

| Model               | ROC-AUC | KS Statistic | Gini   | PR-AUC |
| ------------------- | ------- | ------------ | ------ | ------ |
| Random Forest ✅    | 0.7740  | 0.4571       | 0.5479 | 0.5818 |
| LightGBM            | 0.7354  | 0.4159       | 0.4709 | 0.5141 |
| XGBoost             | 0.7352  | 0.4635       | 0.4705 | 0.5313 |
| Logistic Regression | 0.7321  | 0.3714       | 0.4641 | 0.5170 |

**Top predictive features (SHAP):**

1. `checking_status` — 0.1263
2. `duration` — 0.0512
3. `purpose` — 0.0352
4. `credit_amount` — 0.0284
5. `savings_status` — 0.0259

---

## Architecture

Raw Data (German Credit / Home Credit)
│
▼
Phase 1 — Data Pipeline
loader.py · validator.py · preprocessor.py · splitter.py
│
▼
Phase 2 — Modelling (MLflow tracked)
Logistic Regression · Random Forest · XGBoost · LightGBM
Metrics: ROC-AUC · KS Statistic · Gini · PR-AUC · Brier Score
│
▼
Phase 3 — Explainability
SHAP TreeExplainer · Global importance · Per-applicant waterfall
│
▼
Phase 4 — MLOps
MLflow experiment registry · DVC data versioning
Evidently AI — data drift + model performance monitoring
│
▼
Phase 5 — Serving
FastAPI REST API · Streamlit dashboard

---

## Tech Stack

| Layer               | Technology                      |
| ------------------- | ------------------------------- |
| Models              | XGBoost, LightGBM, scikit-learn |
| Experiment tracking | MLflow                          |
| Data versioning     | DVC                             |
| Explainability      | SHAP                            |
| Monitoring          | Evidently AI                    |
| API                 | FastAPI + Uvicorn               |
| Dashboard           | Streamlit                       |
| Tuning              | Optuna                          |

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/creditiq.git
cd creditiq
conda create -n creditiq python=3.10 -y
conda activate creditiq
pip install -r requirements.txt

# 2. Train all models
python -m src.models.train

# 3. Generate SHAP explanations
python -m src.explainability.shap_explainer

# 4. Generate drift reports
python -m src.monitoring.drift_detector

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000

# 6. Launch the dashboard
python -m streamlit run src/app/streamlit_app.py
```

- **API docs:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501
- **MLflow UI:** run `mlflow ui` → http://localhost:5000

---

## Project Structure

creditiq/
├── src/
│ ├── data/ # loader, splitter, validator
│ ├── features/ # feature engineering, preprocessing pipeline
│ ├── models/ # train, evaluate, tune
│ ├── explainability/ # SHAP + LIME explainers
│ ├── monitoring/ # Evidently AI drift detection
│ ├── api/ # FastAPI app + Pydantic schemas
│ └── app/ # Streamlit dashboard
├── data/
│ ├── raw/ # source CSVs (gitignored)
│ ├── processed/ # engineered features (DVC tracked)
│ └── reference/ # reference dataset for drift detection
├── models/ # serialised model artifacts
├── reports/ # SHAP plots + Evidently HTML reports
├── notebooks/ # EDA and exploration
├── tests/ # unit tests
├── params.yaml # all hyperparameters in one place
└── dvc.yaml # DVC pipeline stages

---

## Key design decisions

**Why KS Statistic and Gini alongside ROC-AUC?**
These are the industry-standard metrics in banking and credit scoring. Accuracy and F1 are insufficient for regulated lending decisions.

**Why SHAP over feature importances?**
SHAP values are consistent, locally accurate, and show both direction and magnitude of each feature's contribution. Feature importances from tree models don't tell you whether high credit amount increases or decreases default risk.

**Why Evidently AI?**
In production, models silently degrade as customer behaviour shifts. Evidently catches distribution shifts in input features before they become business problems.

**Why a Pipeline object?**
The serialised artifact contains both the preprocessor and the model. At inference time, raw applicant data goes in and a probability comes out — no separate transformation step that could be missed or applied inconsistently.

---

## Dataset

- **Development:** German Credit Risk (UCI / OpenML, 1 000 rows, 20 features)
- **Scale target:** Home Credit Default Risk (Kaggle, 300 K rows, 122 features)
