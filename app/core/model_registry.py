from app.services.matrix_factorization_recommender import MatrixFactorizationRecommender
from app.services.popularity_recommender import PopularityRecommender
from training.data.load_movielens import load_movielens_100k
from training.evaluation.split import leave_one_out_split


class ModelRegistry:
    def __init__(self) -> None:
        self.model_name = "matrix_factorization"
        self.recommender = None

    def load(self) -> None:
        ratings, items = load_movielens_100k("data/raw/ml-100k")
        train_ratings, _ = leave_one_out_split(ratings)

        self.recommender = MatrixFactorizationRecommender(
            ratings=train_ratings,
            items=items,
            num_factors=20,
            learning_rate=0.01,
            regularization=0.02,
            num_epochs=10,
            random_state=42,
        )

    def recommend(self, user_id: int, k: int):
        if self.recommender is None:
            raise RuntimeError("Model has not been loaded")

        return self.recommender.recommend(user_id=user_id, k=k)


model_registry = ModelRegistry()