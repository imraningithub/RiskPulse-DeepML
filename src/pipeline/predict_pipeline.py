import os
import sys
import pandas as pd
import numpy as np
import joblib

from src.utils.logger import logger
from src.utils.exception import CustomException

class PredictPipeline:
    def __init__(self, model_path: str = None, threshold_path: str = None):
        self.model_path = model_path or os.path.join("artifacts", "credit_risk_model.pkl")
        self.threshold_path = threshold_path or os.path.join("artifacts", "best_threshold.pkl")
        self.model = None
        self.threshold = 0.5
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded trained credit risk model from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}")

            if os.path.exists(self.threshold_path):
                threshold_data = joblib.load(self.threshold_path)
                if isinstance(threshold_data, (float, int, np.floating)):
                    self.threshold = float(threshold_data)
                elif isinstance(threshold_data, dict) and "threshold" in threshold_data:
                    self.threshold = float(threshold_data["threshold"])
                logger.info(f"Loaded decision threshold: {self.threshold}")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            if self.model is None:
                raise ValueError("Model pipeline is not loaded. Ensure artifacts exist.")

            probabilities = self.model.predict_proba(features)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)

            results = []
            for prob, pred in zip(probabilities, predictions):
                if prob < 0.25:
                    risk_category = "Low Risk"
                elif prob < self.threshold:
                    risk_category = "Moderate Risk"
                elif prob < 0.75:
                    risk_category = "High Risk"
                else:
                    risk_category = "Critical Risk"

                results.append({
                    "default_probability": round(float(prob), 4),
                    "is_default": int(pred),
                    "risk_category": risk_category,
                    "threshold_used": round(self.threshold, 4)
                })

            return results
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        person_age: int,
        person_income: float,
        person_home_ownership: str,
        person_emp_length: float,
        loan_intent: str,
        loan_grade: str,
        loan_amnt: float,
        loan_int_rate: float,
        loan_percent_income: float,
        cb_person_default_on_file: str,
        cb_person_cred_hist_length: int
    ):
        self.person_age = person_age
        self.person_income = person_income
        self.person_home_ownership = person_home_ownership
        self.person_emp_length = person_emp_length
        self.loan_intent = loan_intent
        self.loan_grade = loan_grade
        self.loan_amnt = loan_amnt
        self.loan_int_rate = loan_int_rate
        self.loan_percent_income = loan_percent_income
        self.cb_person_default_on_file = cb_person_default_on_file
        self.cb_person_cred_hist_length = cb_person_cred_hist_length

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_dict = {
                "person_age": [self.person_age],
                "person_income": [self.person_income],
                "person_home_ownership": [self.person_home_ownership],
                "person_emp_length": [self.person_emp_length],
                "loan_intent": [self.loan_intent],
                "loan_grade": [self.loan_grade],
                "loan_amnt": [self.loan_amnt],
                "loan_int_rate": [self.loan_int_rate],
                "loan_percent_income": [self.loan_percent_income],
                "cb_person_default_on_file": [self.cb_person_default_on_file],
                "cb_person_cred_hist_length": [self.cb_person_cred_hist_length]
            }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
