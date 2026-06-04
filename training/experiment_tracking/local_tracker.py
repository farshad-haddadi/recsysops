import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_experiment_result(
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/experiments",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    experiment = {
        "model_name": model_name,
        "timestamp_utc": timestamp,
        "params": params or {},
        "metrics": metrics,
    }

    output_path = output_dir / f"{timestamp}_{model_name}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(experiment, file, indent=2)

    return output_path