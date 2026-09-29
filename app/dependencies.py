from src.pipeline.predict_pipeline import PredictPipeline

_predictor = None

def get_prediction_pipeline() -> PredictPipeline:
    global _predictor
    if _predictor is None:
        _predictor = PredictPipeline()
    return _predictor
