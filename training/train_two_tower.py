from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.data.load_movielens import load_movielens_100k
from training.data.two_tower_dataset import TwoTowerDataset
from training.models.two_tower import TwoTowerModel


def train_two_tower() -> None:
    ratings, items = load_movielens_100k(
        "data/raw/ml-100k"
    )

    unique_users = sorted(
        ratings["user_id"].unique()
    )

    unique_items = sorted(
        ratings["item_id"].unique()
    )

    user_to_index = {
        user_id: idx
        for idx, user_id in enumerate(unique_users)
    }

    item_to_index = {
        item_id: idx
        for idx, item_id in enumerate(unique_items)
    }

    dataset = TwoTowerDataset(
        ratings=ratings,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        negative_samples_per_positive=1,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
    )

    model = TwoTowerModel(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=64,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 5

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for (
            user_indices,
            item_indices,
            labels,
        ) in dataloader:

            optimizer.zero_grad()

            scores = model(
                user_indices=user_indices,
                item_indices=item_indices,
            )

            loss = loss_fn(
                scores,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss={avg_loss:.4f}"
        )

    output_dir = Path(
        "artifacts/models"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
        },
        output_dir / "two_tower.pt",
    )

    print(
        "Saved model to "
        "artifacts/models/two_tower.pt"
    )


if __name__ == "__main__":
    train_two_tower()