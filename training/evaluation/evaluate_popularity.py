from pathlib import Path

from app.services.popularity_recommender import PopularityRecommender
from training.data.load_movielens import load_movielens_100k
from training.evaluation.metrics import precision_at_k, recall_at_k
from training.evaluation.split import leave_one_out_split
from training.experiment_tracking.local_tracker import save_experiment_result


def evaluate_popularity(
    data_dir: str | Path,
    k: int = 10,
) -> dict[str, float]:
    ratings, items = load_movielens_100k(data_dir)

    train_ratings, test_ratings = leave_one_out_split(ratings)

    recommender = PopularityRecommender(
        ratings=train_ratings,
        items=items,
    )

    precision_scores = []
    recall_scores = []

    for user_id, user_test_data in test_ratings.groupby("user_id"):
        relevant_items = user_test_data["item_id"].tolist()

        recommendations = recommender.recommend(
            user_id=int(user_id),
            k=k,
        )

        recommended_items = [rec["item_id"] for rec in recommendations]

        precision_scores.append(
            precision_at_k(
                recommended_items=recommended_items,
                relevant_items=relevant_items,
                k=k,
            )
        )

        recall_scores.append(
            recall_at_k(
                recommended_items=recommended_items,
                relevant_items=relevant_items,
                k=k,
            )
        )

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "num_users_evaluated": float(len(test_ratings["user_id"].unique())),
        "k": float(k),
    }


if __name__ == "__main__":
    k = 10

    results = evaluate_popularity(
        data_dir="data/raw/ml-100k",
        k=k,
    )

    artifact_path = save_experiment_result(
        model_name="popularity",
        metrics=results,
        params={"k": k},
    )

    print("Popularity Recommender Evaluation")
    print("--------------------------------")

    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print(f"\nSaved experiment result to: {artifact_path}")