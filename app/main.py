import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    LoanApplicationSchema,
    BatchLoanApplicationSchema,
    RiskPredictionResult,
    BatchRiskPredictionResponse,
    ExplainResponse,
    HealthResponse
)
from app.dependencies import get_prediction_pipeline
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils.logger import logger

app = FastAPI(
    title="RiskPulse DeepML API 🛡️⚡",
    description="Enterprise Credit Risk Assessment & Calibrated Default Probability Forecasting REST API",
    version="1.0.0"
)

# Enable CORS for Streamlit / Frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Welcome to RiskPulse DeepML Microservice",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def check_health(response: Response, predictor: PredictPipeline = Depends(get_prediction_pipeline)):
    model_loaded = predictor.model is not None
    if not model_loaded:
        response.status_code = 503
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        threshold=predictor.threshold,
        version="1.0.0"
    )

@app.post("/predict", response_model=RiskPredictionResult, tags=["Inference"])
def predict_single_loan(
    payload: LoanApplicationSchema,
    predictor: PredictPipeline = Depends(get_prediction_pipeline)
):
    try:
        custom_data = CustomData(
            person_age=payload.person_age,
            person_income=payload.person_income,
            person_home_ownership=payload.person_home_ownership,
            person_emp_length=payload.person_emp_length,
            loan_intent=payload.loan_intent,
            loan_grade=payload.loan_grade,
            loan_amnt=payload.loan_amnt,
            loan_int_rate=payload.loan_int_rate,
            loan_percent_income=payload.loan_percent_income,
            cb_person_default_on_file=payload.cb_person_default_on_file,
            cb_person_cred_hist_length=payload.cb_person_cred_hist_length
        )
        features_df = custom_data.get_data_as_data_frame()
        results = predictor.predict(features_df)
        return results[0]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplainResponse, tags=["Inference"])
def explain_loan_prediction(
    payload: LoanApplicationSchema,
    predictor: PredictPipeline = Depends(get_prediction_pipeline)
):
    try:
        custom_data = CustomData(
            person_age=payload.person_age,
            person_income=payload.person_income,
            person_home_ownership=payload.person_home_ownership,
            person_emp_length=payload.person_emp_length,
            loan_intent=payload.loan_intent,
            loan_grade=payload.loan_grade,
            loan_amnt=payload.loan_amnt,
            loan_int_rate=payload.loan_int_rate,
            loan_percent_income=payload.loan_percent_income,
            cb_person_default_on_file=payload.cb_person_default_on_file,
            cb_person_cred_hist_length=payload.cb_person_cred_hist_length
        )
        features_df = custom_data.get_data_as_data_frame()
        pred_results = predictor.predict(features_df)[0]
        drivers = predictor.explain(features_df)

        return ExplainResponse(
            default_probability=pred_results["default_probability"],
            is_default=pred_results["is_default"],
            risk_category=pred_results["risk_category"],
            drivers=drivers
        )
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchRiskPredictionResponse, tags=["Inference"])
def predict_batch_loans(
    payload: BatchLoanApplicationSchema,
    predictor: PredictPipeline = Depends(get_prediction_pipeline)
):
    try:
        data_dicts = [app_item.model_dump() for app_item in payload.applications]
        features_df = pd.DataFrame(data_dicts)
        results = predictor.predict(features_df)
        return BatchRiskPredictionResponse(
            predictions=results,
            total_processed=len(results)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
