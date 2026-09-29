import pytest
import os
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def test_custom_data_frame_creation():
    custom_data = CustomData(
        person_age=30,
        person_income=75000,
        person_home_ownership="OWN",
        person_emp_length=5.0,
        loan_intent="PERSONAL",
        loan_grade="A",
        loan_amnt=12000,
        loan_int_rate=8.5,
        loan_percent_income=0.16,
        cb_person_default_on_file="N",
        cb_person_cred_hist_length=6
    )
    df = custom_data.get_data_as_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["person_income"].iloc[0] == 75000

def test_prediction_pipeline_execution():
    if not os.path.exists("artifacts/credit_risk_model.pkl"):
        pytest.skip("Model artifact not found. Skipping prediction test.")
    
    predictor = PredictPipeline()
    custom_data = CustomData(
        person_age=25,
        person_income=45000,
        person_home_ownership="RENT",
        person_emp_length=2.0,
        loan_intent="MEDICAL",
        loan_grade="C",
        loan_amnt=15000,
        loan_int_rate=14.5,
        loan_percent_income=0.33,
        cb_person_default_on_file="Y",
        cb_person_cred_hist_length=3
    )
    df = custom_data.get_data_as_data_frame()
    results = predictor.predict(df)
    assert len(results) == 1
    assert 0.0 <= results[0]["default_probability"] <= 1.0
