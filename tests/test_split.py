import pandas as pd

from training.evaluation.split import leave_one_out_split


def test_leave_one_out_split_keeps_latest_interaction_for_test():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [10, 20, 30, 40, 50],
            "rating": [5, 4, 3, 5, 2],
            "timestamp": [100, 200, 300, 100, 500],
        }
    )

    train, test = leave_one_out_split(ratings)

    assert len(test) == 2
    assert len(train) == 3

    user_1_test_item = test[test["user_id"] == 1]["item_id"].iloc[0]
    user_2_test_item = test[test["user_id"] == 2]["item_id"].iloc[0]

    assert user_1_test_item == 30
    assert user_2_test_item == 50