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


model_registry = ModelRegistry()