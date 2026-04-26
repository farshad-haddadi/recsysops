import pandas as pd
from typing import List, Dict
from app.services.base_recommender import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def __init__(self, interactions: pd.DataFrame):
        """
        interactions: DataFrame with columns [user_id, item_id]
        """
        self.interactions = interactions
        self.popular_items = self._compute_popularity()

    def _compute_popularity(self):
        item_counts = (
            self.interactions
            .groupby("item_id")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        return item_counts

    def recommend(self, user_id: str, k: int = 10) -> List[Dict]:
        top_k = self.popular_items.head(k)

        return [
            {"item_id": row["item_id"], "score": float(row["count"])}
            for _, row in top_k.iterrows()
        ]