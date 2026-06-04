from typing import Iterable


def precision_at_k(
    recommended_items: Iterable[int],
    relevant_items: Iterable[int],
    k: int,
) -> float:
    recommended = list(recommended_items)[:k]
    relevant = set(relevant_items)

    if k <= 0:
        raise ValueError("k must be positive")

    if len(recommended) == 0:
        return 0.0

    hits = sum(1 for item in recommended if item in relevant)

    return hits / k


def recall_at_k(
    recommended_items: Iterable[int],
    relevant_items: Iterable[int],
    k: int,
) -> float:
    recommended = list(recommended_items)[:k]
    relevant = set(relevant_items)

    if len(relevant) == 0:
        return 0.0

    hits = sum(1 for item in recommended if item in relevant)

    return hits / len(relevant)