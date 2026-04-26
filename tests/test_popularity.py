import pandas as pd
from app.services.popularity_recommender import PopularityRecommender


def test_popularity_recommender():
    data = pd.DataFrame({
        "user_id": ["u1", "u2", "u1", "u3"],
        "item_id": ["i1", "i1", "i2", "i1"]
    })

    recommender = PopularityRecommender(data)
    recs = recommender.recommend(user_id="u1", k=2)

    assert len(recs) == 2
    assert recs[0]["item_id"] == "i1"