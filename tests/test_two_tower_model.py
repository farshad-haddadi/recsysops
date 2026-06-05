import torch

from training.models.two_tower import TwoTowerModel


def test_two_tower_forward_returns_one_score_per_user_item_pair():
    model = TwoTowerModel(
        num_users=10,
        num_items=20,
        embedding_dim=8,
    )

    user_indices = torch.tensor([0, 1, 2])
    item_indices = torch.tensor([5, 6, 7])

    scores = model(
        user_indices=user_indices,
        item_indices=item_indices,
    )

    assert scores.shape == (3,)


def test_two_tower_embedding_shapes_are_correct():
    model = TwoTowerModel(
        num_users=10,
        num_items=20,
        embedding_dim=8,
    )

    user_indices = torch.tensor([0, 1])
    item_indices = torch.tensor([3, 4])

    user_embeddings = model.get_user_embedding(user_indices)
    item_embeddings = model.get_item_embedding(item_indices)

    assert user_embeddings.shape == (2, 8)
    assert item_embeddings.shape == (2, 8)