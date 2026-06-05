import pandas as pd

from training.data.two_tower_dataset import (
    TwoTowerDataset,
)


def test_dataset_contains_samples():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 11, 12],
        }
    )

    dataset = TwoTowerDataset(
        ratings=ratings,
        user_to_index={1: 0, 2: 1},
        item_to_index={
            10: 0,
            11: 1,
            12: 2,
            13: 3,
            14: 4,
        },
    )

    assert len(dataset) > 0


def test_dataset_returns_three_values():
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
        }
    )

    dataset = TwoTowerDataset(
        ratings=ratings,
        user_to_index={1: 0},
        item_to_index={
            10: 0,
            11: 1,
        },
    )

    sample = dataset[0]

    assert len(sample) == 3