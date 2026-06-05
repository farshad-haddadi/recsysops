from pathlib import Path
from typing import Any

import torch

from training.models.two_tower import TwoTowerModel


class TwoTowerRecommender:
    def __init__(
        self,
        model_path: str | Path,
        items,
        embedding_dim: int = 64,
    ) -> None:
        self.model_path = Path(model_path)
        self.items = items.copy()
        self.embedding_dim = embedding_dim

        checkpoint = torch.load(
            self.model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        self.user_to_index = checkpoint["user_to_index"]
        self.item_to_index = checkpoint["item_to_index"]

        self.index_to_item = {
            index: item_id
            for item_id, index in self.item_to_index.items()
        }

        self.model = TwoTowerModel(
            num_users=len(self.user_to_index),
            num_items=len(self.item_to_index),
            embedding_dim=self.embedding_dim,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be positive")

        if user_id not in self.user_to_index:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_index = self.user_to_index[user_id]

        item_indices = torch.arange(
            len(self.item_to_index),
            dtype=torch.long,
        )

        user_indices = torch.full(
            size=(len(self.item_to_index),),
            fill_value=user_index,
            dtype=torch.long,
        )

        with torch.no_grad():
            scores = self.model(
                user_indices=user_indices,
                item_indices=item_indices,
            )

        top_scores, top_indices = torch.topk(
            scores,
            k=min(k, len(scores)),
        )

        item_titles = (
            self.items
            .set_index("item_id")["title"]
            .to_dict()
        )

        recommendations = []

        for rank, (score, item_index) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist()),
            start=1,
        ):
            item_id = self.index_to_item[item_index]

            recommendations.append(
                {
                    "rank": rank,
                    "item_id": int(item_id),
                    "title": item_titles.get(
                        item_id,
                        "Unknown",
                    ),
                    "score": float(score),
                }
            )

        return recommendations