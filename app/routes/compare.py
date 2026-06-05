from fastapi import APIRouter
import json
from pathlib import Path
from typing import Any

router = APIRouter(tags=["comparison"])


@router.get("/compare-models")
def compare_models() -> dict[str, Any]:
    experiments_dir = Path("artifacts/experiments")

    if not experiments_dir.exists():
        return {"error": "No experiments found"}

    results: dict[str, Any] = {}

    for file in experiments_dir.glob("*.json"):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model_name", file.stem)
        metrics = data.get("metrics", {})

        results[model_name] = {
            "source_file": str(file),
            "precision_at_k": metrics.get("precision_at_k"),
            "recall_at_k": metrics.get("recall_at_k"),
            "final_loss": metrics.get("final_loss"),
            "metrics": metrics,
        }

    if not results:
        return {"error": "No experiments found"}

    winner = max(
        results,
        key=lambda model: (
            results[model].get("precision_at_k")
            if results[model].get("precision_at_k") is not None
            else float("-inf")
        ),
    )

    return {
        "models": results,
        "winner": winner,
        "winner_metric": "precision_at_k",
    }