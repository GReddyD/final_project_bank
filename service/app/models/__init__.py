"""
Pydantic модели для API.
"""

from app.models.schemas import (
    ClientFeatures,
    ProductRecommendation,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

__all__ = [
    "ClientFeatures",
    "ProductRecommendation",
    "PredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
]
