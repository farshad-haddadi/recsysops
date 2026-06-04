from app.services.matrix_factorization_recommender import (
    MatrixFactorizationRecommender,
)
from inference.model_io import save_model
from training.data.load_movielens import load_movielens_100k
from training.evaluation.split import leave_one_out_split


def train_and_save_matrix_factorization() -> None:
    ratings, items = load_movielens_100k("data/raw/ml-100k")
    train_ratings, _ = leave_one_out_split(ratings)

    recommender = MatrixFactorizationRecommender(
        ratings=train_ratings,
        items=items,
        num_factors=20,
        learning_rate=0.01,
        regularization=0.02,
        num_epochs=10,
        random_state=42,
    )

    output_path = save_model(
        model=recommender,
        output_path="artifacts/models/matrix_factorization.pkl",
    )

    print(f"Saved matrix factorization model to: {output_path}")


if __name__ == "__main__":
    train_and_save_matrix_factorization()