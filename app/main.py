from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.recommendations import router as recommendation_router
from app.core.model_registry import model_registry

from app.routes.compare import router as compare_router


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

app.include_router(recommendation_router)
app.include_router(compare_router)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


