from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to generate recommendations for")
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")


class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    title: str | None = None
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    model_name: str
    recommendations: list[RecommendationItem]