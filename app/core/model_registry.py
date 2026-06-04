from typing import Any

from inference.model_io import load_model


class ModelRegistry:
    def __init__(self) -> None:
        self.model_name = "matrix_factorization"
        self.model_path = "artifacts/models/matrix_factorization.pkl"
        self.recommender = None

    def load(self) -> None:
        self.recommender = load_model(self.model_path)

    def recommend(self, user_id: int, k: int):
        if self.recommender is None:
            raise RuntimeError("Model has not been loaded")

        return self.recommender.recommend(user_id=user_id, k=k)

    def get_model_info(self) -> dict[str, Any]:
        if self.recommender is None:
            raise RuntimeError("Model has not been loaded")

        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "num_users": len(self.recommender.user_to_index),
            "num_items": len(self.recommender.item_to_index),
            "num_factors": self.recommender.num_factors,
        }


model_registry = ModelRegistry()