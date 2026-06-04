from training.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 3, 8]

    score = precision_at_k(
        recommended_items=recommended,
        relevant_items=relevant,
        k=5,
    )

    assert score == 0.4


def test_recall_at_k():
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 3, 8]

    score = recall_at_k(
        recommended_items=recommended,
        relevant_items=relevant,
        k=5,
    )

    assert score == 2 / 3