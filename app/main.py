from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.recommendations import router as recommendation_router
from app.core.model_registry import model_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_registry.load()
    yield


app = FastAPI(
    title="RecSysOps API",
    description="Production-style recommendation system API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(recommendation_router)