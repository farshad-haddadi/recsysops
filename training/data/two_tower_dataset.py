import random

import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        ratings: pd.DataFrame,
        user_to_index: dict[int, int],
        item_to_index: dict[int, int],
        negative_samples_per_positive: int = 1,
        random_state: int = 42,
    ) -> None:
        self.samples = []

        rng = random.Random(random_state)

        all_item_ids = set(item_to_index.keys())

        user_history = (
            ratings.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        for _, row in ratings.iterrows():
            user_id = row["user_id"]
            item_id = row["item_id"]

            self.samples.append(
                (
                    user_to_index[user_id],
                    item_to_index[item_id],
                    1.0,
                )
            )

            available_negatives = list(
                all_item_ids - user_history[user_id]
            )

            for _ in range(negative_samples_per_positive):
                negative_item = rng.choice(
                    available_negatives
                )

                self.samples.append(
                    (
                        user_to_index[user_id],
                        item_to_index[negative_item],
                        0.0,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        user_idx, item_idx, label = self.samples[index]

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )