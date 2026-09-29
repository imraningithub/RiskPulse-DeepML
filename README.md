# RiskPulse DeepML 🛡️⚡
> **Calibrated Machine Learning & Explainable AI (SHAP) Credit Risk Platform**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-riskpulse--ui.onrender.com-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://riskpulse-ui.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://riskpulse-api-4alt.onrender.com/docs)

### 🔗 Try it live
| | Link |
|---|---|
| **Web dashboard (Streamlit)** | https://riskpulse-ui.onrender.com |
| **REST API (FastAPI)** | https://riskpulse-api-4alt.onrender.com |
| **Interactive API docs** | https://riskpulse-api-4alt.onrender.com/docs |

> ⏱️ Hosted on Render's free tier. If nobody has visited for a while the services sleep, so the **first load can take 30-60 seconds** to wake up. After that it is fast.

`RiskPulse DeepML` is an end-to-end credit risk assessment system that predicts the probability a loan applicant will default, explains *why*, and serves it through a REST API and an interactive dashboard. It is built with **Scikit-Learn**, **XGBoost**, **probability calibration (Platt scaling)**, **SHAP**, **FastAPI**, **Streamlit** and **Docker**, and deployed on **Render**.

---

## ✨ Features
- **Single applicant scoring**: enter a loan application and get a default probability, a risk category (e.g. *Low Risk*) and a decision.
- **Explainable predictions**: SHAP-based per-feature drivers show which inputs push the risk up or down (`/explain`).
- **Batch scoring**: upload many applications at once and get portfolio-level risk analytics (`/predict/batch`).
- **Threshold simulator**: explore the trade-off between catching defaulters and rejecting good applicants as the decision threshold moves.
- **Calibrated probabilities**: the output is a usable probability, not just a score (`CalibratedClassifierCV`, sigmoid). The tuned decision threshold is **0.45**.
- **Validated inputs**: Pydantic schemas reject out-of-range values with clear errors.
- **Deploy-safe health check**: `/health` returns HTTP 503 if the model fails to load, so a broken deploy fails loudly instead of serving wrong answers.

---

## 📊 Model Performance
Evaluated on a stratified 20% hold-out set (6,305 applicants, ~22% defaults), from `notebooks/04_model_training_and_evaluation.ipynb`:

| Model | Accuracy | Precision (default) | Recall (default) | F1 (default) |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.82 | 0.56 | 0.79 | 0.65 |
| **XGBoost** | **0.92** | **0.82** | **0.79** | **0.81** |

The XGBoost model was tuned with 5-fold stratified cross-validation (`RandomizedSearchCV`, targeting PR-AUC), then calibrated and given an optimised decision threshold. The notebook also benchmarks a Keras MLP neural network. The dataset is the public *Credit Risk Dataset* (`data/credit_risk_dataset.csv`), and the modelling workflow is documented across four notebooks (EDA → validation & cleaning → feature engineering → training & evaluation).

---

## 🔌 API Reference
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service and model status (`503` if the model is not loaded) |
| `POST` | `/predict` | Default probability and risk category for one applicant |
| `POST` | `/explain` | Same as `/predict` plus SHAP feature drivers |
| `POST` | `/predict/batch` | Score a list of applicants in one request |

Example:
```bash
curl -X POST https://riskpulse-api-4alt.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 28,
    "person_income": 65000,
    "person_home_ownership": "RENT",
    "person_emp_length": 4.0,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.14,
    "loan_percent_income": 0.15,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 4
  }'
```
```json
{"default_probability": 0.0234, "is_default": 0, "risk_category": "Low Risk", "threshold_used": 0.45}
```

---

## 🧱 Architecture
```text
   Browser ──► Streamlit UI (ui/app.py) ──HTTP──► FastAPI (app/main.py) ──► PredictPipeline
                                                                             ├─ credit_risk_model.pkl  (calibrated XGBoost pipeline)
                                                                             └─ best_threshold.pkl     (decision threshold)
```
The UI is a thin client: all modelling logic lives in the API, which loads the model once at startup.

**Tech stack:** Python · pandas · scikit-learn · XGBoost · imbalanced-learn · SHAP · FastAPI · Pydantic · Streamlit · pytest · Docker · Render · GitHub Actions

---

## 🏗️ Project Structure

```text
Credit Risk/
├── app/                              # FastAPI Web Server & Pydantic Schemas
│   ├── main.py                       # FastAPI application entry point & routes
│   ├── schemas.py                    # Pydantic request & response validation schemas
│   └── dependencies.py               # Dependency injection & model singleton loader
├── ui/
│   └── app.py                        # Streamlit dashboard for single & batch loan scoring
├── src/                              # Modular Machine Learning Engine
│   ├── config/configuration.py       # Data & artifact path configuration
│   ├── pipeline/
│   │   ├── train_pipeline.py         # Model training pipeline script
│   │   └── predict_pipeline.py       # Real-time inference engine
│   └── utils/                        # Logging & custom exception helpers
├── artifacts/                        # Serialized production model & threshold
│   ├── credit_risk_model.pkl         # Trained & calibrated XGBoost pipeline
│   └── best_threshold.pkl            # Optimised decision threshold
├── data/                             # Raw & cleaned credit risk datasets
├── notebooks/                        # EDA → cleaning → feature engineering → modelling
├── tests/                            # pytest suite (API + pipeline)
├── .github/workflows/keep-warm.yml   # Scheduled ping that keeps the free Render services awake
├── render.yaml                       # Render Blueprint (API + UI services)
├── Dockerfile                        # Production container image
├── docker-compose.yml                # Local orchestration (API + UI)
└── requirements.txt                  # Pinned Python dependencies
```

---

## 🌟 Run It Locally

### 1. With a Python virtual environment
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**FastAPI backend:**
```bash
uvicorn app.main:app --reload --port 8000
```
- Swagger docs: `http://localhost:8000/docs` · Health: `http://localhost:8000/health`

**Streamlit dashboard** (in a second terminal; it talks to `http://localhost:8000` by default, override with the `API_URL` env var):
```bash
streamlit run ui/app.py --server.port 8501
```
- Open `http://localhost:8501`

### 2. With Docker
```bash
docker-compose up --build
```
- FastAPI backend: `http://localhost:8000`
- Streamlit dashboard: `http://localhost:8501`

---

## ☁️ Deployment (Render)
The app runs as two Render web services defined in [`render.yaml`](render.yaml):

| Service | Start command |
|---|---|
| `riskpulse-api` | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (health check: `/health`) |
| `riskpulse-ui` | `streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0` (`API_URL` points at the API) |

Pushing to `main` auto-deploys both. Python and the ML libraries are pinned to the versions the model was trained with, because pickled scikit-learn/XGBoost models can break across versions. A GitHub Actions workflow pings both services every 10 minutes during the daytime (UTC 07:00-18:59) to reduce cold starts on the free tier.

---

## 🧪 Running Tests
```bash
pytest
```
