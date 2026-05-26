from pydantic import BaseModel
from typing import Dict


class MLModelInsights(BaseModel):
    model_type: str
    feature_used: str
    target_variable: str
    coefficient_slope: float
    intercept: float
    model_score_R2: float


class PipelineResults(BaseModel):
    total_students_processed: int
    average_exam_score: float
    average_study_hours: float
    ml_model_insights: MLModelInsights
    analysis_meta: Dict[str, str]


class PredictionResponse(BaseModel):
    status: str
    results: PipelineResults
