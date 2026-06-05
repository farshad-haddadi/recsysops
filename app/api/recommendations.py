from fastapi import APIRouter, HTTPException

from app.core.metrics_registry import metrics_registry
from app.core.model_registry import model_registry
from app.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    try:
        recommendations = model_registry.recommend(
            user_id=request.user_id,
            k=request.k,
            model_name=request.model_name,
        )

        return RecommendationResponse(
            user_id=request.user_id,
            model_name=request.model_name,
            recommendations=recommendations,
        )

    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/model-info")
def model_info():
    try:
        return model_registry.get_model_info()

    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/metrics")
def metrics():
    return metrics_registry.get_latest_metrics()