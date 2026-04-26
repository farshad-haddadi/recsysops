from typing import List, Dict


class BaseRecommender:
    def recommend(self, user_id: str, k: int = 10) -> List[Dict]:
        """
        Return top-k recommendations for a user.

        Args:
            user_id (str): user identifier
            k (int): number of recommendations

        Returns:
            List[Dict]: list of {item_id, score}
        """
        raise NotImplementedError