from typing import Any

from inference.model_io import load_model
from inference.two_tower_recommender import TwoTowerRecommender
from training.data.load_movielens import load_movielens_100k


class ModelRegistry:
    def __init__(self) -> None:
        self.default_model_name = "matrix_factorization"

        self.matrix_factorization_path = (
            "artifacts/models/matrix_factorization.pkl"
        )
        self.two_tower_path = "artifacts/models/two_tower.pt"

        self.models = {}

    def load(self) -> None:
        _, items = load_movielens_100k("data/raw/ml-100k")

        self.models["matrix_factorization"] = load_model(
            self.matrix_factorization_path
        )

        self.models["two_tower"] = TwoTowerRecommender(
            model_path=self.two_tower_path,
            items=items,
        )

    def recommend(
        self,
        user_id: int,
        k: int,
        model_name: str | None = None,
    ):
        selected_model = model_name or self.default_model_name

        if selected_model not in self.models:
            raise ValueError(f"Unknown model_name: {selected_model}")

        return self.models[selected_model].recommend(
            user_id=user_id,
            k=k,
        )

    def get_model_info(self) -> dict[str, Any]:
        if not self.models:
            raise RuntimeError("Models have not been loaded")

        return {
            "default_model_name": self.default_model_name,
            "available_models": list(self.models.keys()),
            "matrix_factorization_path": self.matrix_factorization_path,
            "two_tower_path": self.two_tower_path,
        }


model_registry = ModelRegistry()