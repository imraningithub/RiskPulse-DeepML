import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "credit_risk_dataset.csv")
    cleaned_data_path: str = os.path.join("data", "cleaned_credit_risk_dataset.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "credit_risk_model.pkl")
    threshold_file_path: str = os.path.join("artifacts", "best_threshold.pkl")
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")
