from abc import ABC, abstractmethod
from typing import Any


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, user_id: int, k: int = 10) -> list[dict[str, Any]]:
        """
        Return top-k recommendations for a user.
        """
        raise NotImplementedError