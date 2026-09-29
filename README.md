# RiskPulse DeepML 🛡️⚡
> **Calibrated Machine Learning, Deep Learning & Explainable AI (SHAP) Credit Risk Platform**

`RiskPulse DeepML` is an end-to-end, enterprise-grade Machine Learning system designed for credit risk assessment and default probability forecasting. Built with **Scikit-Learn**, **XGBoost**, **Probability Calibration (Platt Scaling)**, **FastAPI Microservice**, **Streamlit Web Dashboard**, and **Docker Containerization**.

---

## 🏗️ Production Directory Architecture

```text
Credit Risk/
├── app/                              # FastAPI Web Server & Pydantic Schemas
│   ├── __init__.py
│   ├── main.py                       # FastAPI application entry point & routes
│   ├── schemas.py                    # Pydantic request & response validation schemas
│   └── dependencies.py               # Dependency injection & model singleton loader
├── ui/                               # Streamlit Web UI Dashboard
│   ├── __init__.py
│   └── app.py                        # Streamlit dashboard for single & batch loan scoring
├── src/                              # Modular Machine Learning Engine
│   ├── __init__.py
│   ├── config/                       # Data & artifact configuration specs
│   │   ├── __init__.py
│   │   └── configuration.py
│   ├── components/                   # Core ML components (Ingestion, Trainer, Calibrator, Evaluator)
│   │   └── __init__.py
│   ├── pipeline/                     # Production training & prediction orchestrators
│   │   ├── __init__.py
│   │   ├── train_pipeline.py         # End-to-end model retraining pipeline script
│   │   └── predict_pipeline.py       # Real-time inference prediction engine
│   └── utils/                        # Logging & Exception handling helpers
│       ├── __init__.py
│       ├── logger.py                 # Centralized logging configuration
│       └── exception.py              # Custom exception wrapper
├── artifacts/                        # Serialized production model & threshold artifacts
│   ├── credit_risk_model.pkl         # Trained & calibrated XGBoost pipeline
│   └── best_threshold.pkl            # Decision threshold optimization artifact
├── data/                             # Raw & processed credit risk datasets
│   ├── credit_risk_dataset.csv       # Raw credit risk dataset
│   └── cleaned_credit_risk_dataset.csv # Processed & validated dataset
├── notebooks/                        # Exploratory Data Analysis & Notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_validation_and_cleaning.ipynb
│   ├── 03_feature_engineering_and_preprocessing.ipynb
│   └── 04_model_training_and_evaluation.ipynb
├── tests/                            # PyTest unit & integration test suite
│   ├── __init__.py
│   ├── test_api.py                   # FastAPI endpoint tests
│   └── test_pipeline.py              # Inference & data frame creation tests
├── Dockerfile                        # Multi-stage production container image
├── docker-compose.yml                # Microservice orchestration (API + UI)
├── .dockerignore                     # Docker build exclusion rules
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🌟 Quick Start Guide

### 1. Running Locally with Python Virtual Environment

Activate your virtual environment and install requirements:
```bash
# On Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

#### Launch the FastAPI Microservice:
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`

#### Launch the Streamlit Web Dashboard:
```bash
streamlit run ui/app.py --server.port 8501
```
- Open Browser: `http://localhost:8501`

---

### 2. Deployment via Docker & Docker Compose

To build and run both the API backend and Streamlit UI in isolated containers:
```bash
docker-compose up --build
```
- **FastAPI Backend**: `http://localhost:8000`
- **Streamlit Web Dashboard**: `http://localhost:8501`

---

## 🧪 Running Tests
```bash
pytest
```
