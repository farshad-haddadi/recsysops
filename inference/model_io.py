import pickle
from pathlib import Path
from typing import Any


def save_model(
    model: Any,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(model, file)

    return output_path


def load_model(
    model_path: str | Path,
) -> Any:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    with model_path.open("rb") as file:
        model = pickle.load(file)

    return model