import json

from training.experiment_tracking.local_tracker import save_experiment_result


def test_save_experiment_result_creates_json_file(tmp_path):
    output_path = save_experiment_result(
        model_name="test_model",
        metrics={"recall_at_k": 0.5},
        params={"k": 10},
        output_dir=tmp_path,
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["model_name"] == "test_model"
    assert data["metrics"]["recall_at_k"] == 0.5
    assert data["params"]["k"] == 10