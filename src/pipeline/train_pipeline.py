import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib

from src.config.configuration import DataIngestionConfig, ModelTrainerConfig
from src.utils.logger import logger
from src.utils.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.trainer_config = ModelTrainerConfig()

    def run_pipeline(self):
        try:
            logger.info("Starting production training pipeline execution...")

            # 1. Load cleaned dataset
            if not os.path.exists(self.ingestion_config.cleaned_data_path):
                raise FileNotFoundError(f"Cleaned dataset not found at {self.ingestion_config.cleaned_data_path}")

            df = pd.read_csv(self.ingestion_config.cleaned_data_path)
            X = df.drop(columns=['loan_status'])
            y = df['loan_status']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            preprocessor_xg = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), cat_cols)
                ]
            )

            counts = np.bincount(y_train)
            scale_weight = counts[0] / counts[1]

            # Same recipe as notebooks/04: XGBoost (class imbalance via scale_pos_weight),
            # wrapped in Platt (sigmoid) probability calibration with 5-fold CV.
            base_pipeline = Pipeline([
                ('preprocessor', preprocessor_xg),
                ('classifier', XGBClassifier(
                    learning_rate=0.1,
                    scale_pos_weight=scale_weight,
                    random_state=42
                ))
            ])
            model_pipeline = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=5)

            logger.info("Fitting calibrated XGBoost pipeline on training data...")
            model_pipeline.fit(X_train, y_train)

            os.makedirs(os.path.dirname(self.trainer_config.model_file_path), exist_ok=True)
            joblib.dump(model_pipeline, self.trainer_config.model_file_path)
            logger.info(f"Model saved to {self.trainer_config.model_file_path}")

            # Operating point chosen for the API (configurable); see notebook 04 section 5 for the F1-vs-threshold analysis.
            best_threshold = 0.45
            joblib.dump(best_threshold, self.trainer_config.threshold_file_path)
            logger.info(f"Threshold saved to {self.trainer_config.threshold_file_path}")

            logger.info("Training pipeline completed successfully.")
            return self.trainer_config.model_file_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
