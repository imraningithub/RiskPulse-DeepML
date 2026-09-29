from pydantic import BaseModel, Field
from typing import List

class LoanApplicationSchema(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Applicant age in years", json_schema_extra={"example": 28})
    person_income: float = Field(..., ge=0, description="Annual income in USD", json_schema_extra={"example": 65000})
    person_home_ownership: str = Field(..., description="Home ownership status: RENT, OWN, MORTGAGE, OTHER", json_schema_extra={"example": "RENT"})
    person_emp_length: float = Field(..., ge=0, le=60, description="Employment length in years", json_schema_extra={"example": 4.0})
    loan_intent: str = Field(..., description="Purpose of loan: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION", json_schema_extra={"example": "EDUCATION"})
    loan_grade: str = Field(..., description="Assigned credit risk grade (A to G)", json_schema_extra={"example": "B"})
    loan_amnt: float = Field(..., ge=500, description="Requested loan amount in USD", json_schema_extra={"example": 10000})
    loan_int_rate: float = Field(..., ge=0, le=40, description="Interest rate percentage", json_schema_extra={"example": 11.14})
    loan_percent_income: float = Field(..., ge=0, le=1.0, description="Ratio of loan amount to annual income", json_schema_extra={"example": 0.15})
    cb_person_default_on_file: str = Field(..., description="Historical default record on file: Y or N", json_schema_extra={"example": "N"})
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length in years", json_schema_extra={"example": 4})

class BatchLoanApplicationSchema(BaseModel):
    applications: List[LoanApplicationSchema]

class RiskPredictionResult(BaseModel):
    default_probability: float
    is_default: int
    risk_category: str
    threshold_used: float

class BatchRiskPredictionResponse(BaseModel):
    predictions: List[RiskPredictionResult]
    total_processed: int

class FeatureDriver(BaseModel):
    feature: str
    impact: float
    direction: str

class ExplainResponse(BaseModel):
    default_probability: float
    is_default: int
    risk_category: str
    drivers: List[FeatureDriver]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float
    version: str = "1.0.0"
