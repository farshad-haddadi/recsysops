from inference.model_io import load_model, save_model


def test_save_and_load_model_roundtrip(tmp_path):
    model = {"name": "test_model", "version": 1}
    model_path = tmp_path / "model.pkl"

    saved_path = save_model(
        model=model,
        output_path=model_path,
    )

    loaded_model = load_model(saved_path)

    assert loaded_model == model