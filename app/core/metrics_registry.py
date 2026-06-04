import json
from pathlib import Path
from typing import Any


class MetricsRegistry:
    def __init__(self, experiments_dir: str = "artifacts/experiments") -> None:
        self.experiments_dir = Path(experiments_dir)

    def get_latest_metrics(self) -> dict[str, Any]:
        if not self.experiments_dir.exists():
            return {"message": "No experiment artifacts found"}

        experiment_files = sorted(
            self.experiments_dir.glob("*.json"),
            reverse=True,
        )

        if not experiment_files:
            return {"message": "No experiment artifacts found"}

        latest_file = experiment_files[0]

        with latest_file.open("r", encoding="utf-8") as file:
            experiment = json.load(file)

        return {
            "source_file": str(latest_file),
            "model_name": experiment.get("model_name"),
            "timestamp_utc": experiment.get("timestamp_utc"),
            "params": experiment.get("params", {}),
            "metrics": experiment.get("metrics", {}),
        }


metrics_registry = MetricsRegistry()