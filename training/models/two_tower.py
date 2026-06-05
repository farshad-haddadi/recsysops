import torch
from torch import nn


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_tower = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

        self.item_tower = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.user_tower.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.item_tower.weight, mean=0.0, std=0.05)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.user_tower(user_indices)
        item_embeddings = self.item_tower(item_indices)

        scores = torch.sum(user_embeddings * item_embeddings, dim=1)

        return scores

    def get_user_embedding(self, user_indices: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_indices)

    def get_item_embedding(self, item_indices: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_indices)