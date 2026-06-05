from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to generate recommendations for")
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    model_name: str = Field(
        default="matrix_factorization",
        description="Model to use: matrix_factorization or two_tower",
    )


class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    title: str | None = None
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    model_name: str
    recommendations: list[RecommendationItem]