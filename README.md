# RiskPulse DeepML 🛡️⚡
> **Calibrated Machine Learning, Deep Learning & Explainable AI (SHAP) Credit Risk Platform**

`RiskPulse DeepML` is an end-to-end, enterprise-grade Machine Learning system designed for credit risk assessment and default probability forecasting. Built with **Scikit-Learn**, **XGBoost**, **Keras/TensorFlow**, **Probability Calibration (Platt / Isotonic Scaling)**, **SHAP (SHapley Additive exPlanations)**, and served via a high-performance **FastAPI microservice**.

---

## 🏗️ Architecture Overview

```text
RiskPulse DeepML/
├── data/
│   ├── credit_risk_dataset.csv       # Raw credit risk dataset
│   └── cleaned_credit_risk_dataset.csv # Processed & validated dataset
├── src/                              # Modular Machine Learning Pipeline
│   ├── config/                       # Centralized configurations & hyperparameter specs
│   ├── utils/                        # Logging, custom exceptions & metrics helpers
│   ├── components/                   # Ingestion, Validation, Preprocessing, Trainer, Evaluator
│   └── pipeline/                     # Training & Inference orchestrators
├── app/                              # FastAPI Web Server & Pydantic API Schemas
├── artifacts/                        # Saved models, encoders, scalers, calibration curves & SHAP plots
├── notebooks/                        # Exploratory Data Analysis & experiment notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_validation_and_cleaning.ipynb
│   ├── 03_feature_engineering_and_preprocessing.ipynb
│   └── 04_model_training_and_evaluation.ipynb
├── tests/                            # Unit & Integration test suite
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## 🌟 Key Features

- **Multi-Model Benchmarking**: Baseline Logistic Regression, XGBoost, and Deep Neural Networks (Keras MLP) evaluated side-by-side.
- **Automated Preprocessing Pipelines**: Scikit-Learn `ColumnTransformer` & Pipelines for leak-free scaling, encoding, and class imbalance handling.
- **Threshold Optimization & Calibration**: Precision-Recall utility tuning and probability calibration (Platt / Isotonic Scaling) for financial risk scoring.
- **SHAP Explainability & Error Auditing**: Global/local feature attributions and False Positive / False Negative risk inspection.
- **FastAPI Microservice**: High-throughput RESTful API for real-time inference with Pydantic payload validation.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📌 Project Roadmap

- [x] **Step 1: Project Setup & Modular Architecture**
  - Set up `RiskPulse DeepML` workspace, `venv` environment, and modular `src/` directory layout.

- [/] **Step 2: EDA, Data Validation & Outlier Cleaning** (`notebooks/`)
  - Perform exploratory data analysis, remove duplicates, filter invalid demographic/financial values, and export cleaned dataset.

- [ ] **Step 3: Feature Engineering, Splitting & Preprocessing Pipelines**
  - Define target ($y = \text{loan\_status}$), establish stratified train-test split, address class imbalance, and build Scikit-Learn `ColumnTransformer` pipelines.

- [ ] **Step 4: Model Training, Cross-Validation & Benchmarking**
  - Train and evaluate Baseline Logistic Regression, XGBoost, and Keras Deep Neural Networks using Stratified Cross-Validation.

- [ ] **Step 5: Threshold Optimization & Probability Calibration**
  - Optimize classification decision thresholds for loan default risk and calibrate raw output probabilities using Platt / Isotonic Scaling.

- [ ] **Step 6: Explainable AI (SHAP) & Error Auditing**
  - Generate SHAP global summary and local waterfall plots for regulatory transparency; audit False Positives and False Negatives.

- [ ] **Step 7: Production Pipeline Refactoring & Model Artifact Serialization**
  - Refactor notebook logic into modular `src/components/` and serialize final trained models, encoders, and scalers to `artifacts/`.

- [ ] **Step 8: FastAPI REST Service & Docker Deployment**
  - Build real-time inference API endpoints (`/predict`, `/health`) with Pydantic validation and package into Docker containers.
