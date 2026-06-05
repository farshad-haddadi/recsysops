from pathlib import Path

import pytest

from inference.two_tower_recommender import TwoTowerRecommender
from training.data.load_movielens import load_movielens_100k


MODEL_PATH = Path("artifacts/models/two_tower.pt")


def test_two_tower_recommender_returns_recommendations():
    if not MODEL_PATH.exists():
        pytest.skip("Two-tower model artifact not available")

    _, items = load_movielens_100k("data/raw/ml-100k")

    recommender = TwoTowerRecommender(
        model_path=MODEL_PATH,
        items=items,
    )

    recommendations = recommender.recommend(
        user_id=1,
        k=5,
    )

    assert len(recommendations) == 5
    assert "item_id" in recommendations[0]
    assert "title" in recommendations[0]
    assert "score" in recommendations[0]