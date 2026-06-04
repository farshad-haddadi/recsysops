from pathlib import Path

import pandas as pd


RATINGS_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
ITEM_COLUMNS = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    ratings_path = data_dir / "u.data"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=RATINGS_COLUMNS,
        encoding="latin-1",
    )

    return ratings


def load_items(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    items_path = data_dir / "u.item"

    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")

    items = pd.read_csv(
        items_path,
        sep="|",
        names=ITEM_COLUMNS,
        encoding="latin-1",
    )

    return items


def load_movielens_100k(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = load_ratings(data_dir)
    items = load_items(data_dir)

    return ratings, items