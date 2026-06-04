import pandas as pd

from app.services.matrix_factorization_recommender import MatrixFactorizationRecommender


def test_matrix_factorization_recommender_returns_recommendations():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [10, 20, 10, 30, 20, 30],
            "rating": [5, 4, 5, 3, 4, 5],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [10, 20, 30, 40],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
        }
    )

    recommender = MatrixFactorizationRecommender(
        ratings=ratings,
        items=items,
        num_factors=5,
        num_epochs=2,
        random_state=42,
    )

    recommendations = recommender.recommend(user_id=1, k=1)

    assert len(recommendations) == 1
    assert "item_id" in recommendations[0]
    assert "score" in recommendations[0]
    assert "rank" in recommendations[0]


def test_matrix_factorization_predicts_float_score():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 4, 5],
            "timestamp": [1, 2, 3],
        }
    )

    recommender = MatrixFactorizationRecommender(
        ratings=ratings,
        num_factors=3,
        num_epochs=1,
    )

    score = recommender.predict_rating(user_id=1, item_id=10)

    assert isinstance(score, float)