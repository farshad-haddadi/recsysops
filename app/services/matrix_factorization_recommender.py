from typing import Any

import numpy as np
import pandas as pd

from app.services.base_recommender import BaseRecommender


class MatrixFactorizationRecommender(BaseRecommender):
    def __init__(
        self,
        ratings: pd.DataFrame,
        items: pd.DataFrame | None = None,
        num_factors: int = 20,
        learning_rate: float = 0.01,
        regularization: float = 0.02,
        num_epochs: int = 10,
        random_state: int = 42,
    ):
        self.ratings = ratings.copy()
        self.items = items.copy() if items is not None else None

        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.random_state = random_state

        self.user_to_index: dict[int, int] = {}
        self.item_to_index: dict[int, int] = {}
        self.index_to_item: dict[int, int] = {}

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.global_mean: float = 0.0

        self._fit()

    def _create_id_mappings(self) -> None:
        unique_users = sorted(self.ratings["user_id"].unique())
        unique_items = sorted(self.ratings["item_id"].unique())

        self.user_to_index = {
            int(user_id): idx for idx, user_id in enumerate(unique_users)
        }

        self.item_to_index = {
            int(item_id): idx for idx, item_id in enumerate(unique_items)
        }

        self.index_to_item = {
            idx: int(item_id) for item_id, idx in self.item_to_index.items()
        }

    def _fit(self) -> None:
        self._create_id_mappings()

        rng = np.random.default_rng(self.random_state)

        num_users = len(self.user_to_index)
        num_items = len(self.item_to_index)

        self.user_factors = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(num_users, self.num_factors),
        )

        self.item_factors = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(num_items, self.num_factors),
        )

        self.global_mean = float(self.ratings["rating"].mean())

        training_rows = (
            self.ratings[["user_id", "item_id", "rating"]]
            .to_numpy()
            .copy()
        )

        for _ in range(self.num_epochs):
            rng.shuffle(training_rows)

            for user_id, item_id, rating in training_rows:
                user_id = int(user_id)
                item_id = int(item_id)
                rating = float(rating)

                user_idx = self.user_to_index[user_id]
                item_idx = self.item_to_index[item_id]

                user_vector = self.user_factors[user_idx]
                item_vector = self.item_factors[item_idx]

                prediction = self.global_mean + float(
                    np.dot(user_vector, item_vector)
                )

                error = rating - prediction

                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_vector
                    - self.regularization * user_vector
                )

                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_vector
                    - self.regularization * item_vector
                )

    def predict_rating(
        self,
        user_id: int,
        item_id: int,
    ) -> float:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been trained")

        if user_id not in self.user_to_index:
            return self.global_mean

        if item_id not in self.item_to_index:
            return self.global_mean

        user_idx = self.user_to_index[user_id]
        item_idx = self.item_to_index[item_id]

        prediction = self.global_mean + float(
            np.dot(
                self.user_factors[user_idx],
                self.item_factors[item_idx],
            )
        )

        return prediction

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be positive")

        seen_items = set(
            self.ratings[
                self.ratings["user_id"] == user_id
            ]["item_id"].tolist()
        )

        candidate_items = [
            item_id
            for item_id in self.item_to_index.keys()
            if item_id not in seen_items
        ]

        recommendations = []

        for item_id in candidate_items:
            score = self.predict_rating(
                user_id=user_id,
                item_id=item_id,
            )

            recommendations.append(
                {
                    "item_id": int(item_id),
                    "score": float(score),
                }
            )

        recommendations.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        recommendations = recommendations[:k]

        if self.items is not None:
            item_titles = (
                self.items
                .set_index("item_id")["title"]
                .to_dict()
            )

            for rec in recommendations:
                rec["title"] = item_titles.get(
                    rec["item_id"],
                    "Unknown",
                )

        for rank, rec in enumerate(
            recommendations,
            start=1,
        ):
            rec["rank"] = rank

        return recommendations