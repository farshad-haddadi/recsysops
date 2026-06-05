from fastapi import APIRouter
import json
from pathlib import Path

router = APIRouter(tags=["comparison"])

@router.get("/compare-models")
def compare_models():
    experiments_dir = Path("artifacts/experiments")

    results = {}

    for file in experiments_dir.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)

        model_name = data["model_name"]

        results[model_name] = {
            "precision_at_k": data["metrics"]["precision_at_k"],
            "recall_at_k": data["metrics"]["recall_at_k"]
        }

    if not results:
        return {"error": "No experiments found"}

    winner = max(
        results,
        key=lambda x: results[x]["precision_at_k"]
    )

    return {
        "models": results,
        "winner": winner
    }