import pandas as pd
import pytest

from app.services.popularity_recommender import PopularityRecommender


def test_popularity_recommender_returns_top_k_items():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "item_id": [10, 10, 10, 20, 20, 30],
            "rating": [5, 5, 4, 5, 4, 5],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [10, 20, 30],
            "title": ["Movie A", "Movie B", "Movie C"],
        }
    )

    recommender = PopularityRecommender(ratings=ratings, items=items)
    recommendations = recommender.recommend(user_id=1, k=2)

    assert len(recommendations) == 2
    assert recommendations[0]["item_id"] == 10
    assert recommendations[0]["title"] == "Movie A"
    assert recommendations[0]["rank"] == 1


def test_popularity_recommender_rejects_invalid_k():
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5],
            "timestamp": [1],
        }
    )

    recommender = PopularityRecommender(ratings=ratings)

    with pytest.raises(ValueError):
        recommender.recommend(user_id=1, k=0)