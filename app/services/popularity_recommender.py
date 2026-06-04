from typing import Any

import pandas as pd

from app.services.base_recommender import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def __init__(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None):
        self.ratings = ratings.copy()
        self.items = items.copy() if items is not None else None
        self.popularity_table = self._build_popularity_table()

    def _build_popularity_table(self) -> pd.DataFrame:
        popularity = (
            self.ratings.groupby("item_id")
            .agg(
                num_ratings=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )

        popularity["score"] = popularity["num_ratings"] * popularity["avg_rating"]

        popularity = popularity.sort_values(
            by=["score", "num_ratings", "avg_rating"],
            ascending=False,
        )

        if self.items is not None:
            popularity = popularity.merge(
                self.items[["item_id", "title"]],
                on="item_id",
                how="left",
            )

        return popularity

    def recommend(self, user_id: int, k: int = 10) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be positive")

        top_k = self.popularity_table.head(k)

        recommendations = []

        for rank, row in enumerate(top_k.itertuples(index=False), start=1):
            recommendation = {
                "rank": rank,
                "item_id": int(row.item_id),
                "score": float(row.score),
                "num_ratings": int(row.num_ratings),
                "avg_rating": float(row.avg_rating),
            }

            if hasattr(row, "title"):
                recommendation["title"] = row.title

            recommendations.append(recommendation)

        return recommendations