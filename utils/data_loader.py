"""
data_loader.py
---------------
Small helper functions that load and lightly clean the MovieLens
"movies.csv" and "ratings.csv" files.

Kept separate from the recommendation logic so that the data-access
code can be tested / reused independently of the ML model.
"""

import os
import re
import pandas as pd

# Folder that holds movies.csv / ratings.csv relative to the project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")


def _extract_year(title: str):
    """Pull a 4-digit year out of a MovieLens-style title, e.g. 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match.group(1)) if match else None


def _clean_title(title: str) -> str:
    """Remove the trailing '(YYYY)' from a title, leaving just the movie name."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def load_movies(path: str = MOVIES_PATH) -> pd.DataFrame:
    """
    Load movies.csv and add a few convenience columns:
      - clean_title : title without the release year
      - year        : release year as an int (NaN if missing)
      - genres_list  : genres as a Python list instead of a '|' separated string
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Make sure movies.csv is inside the 'data' folder."
        )

    movies = pd.read_csv(path)

    # Some MovieLens rows use the literal string "(no genres listed)" -> normalize to empty
    movies["genres"] = movies["genres"].fillna("(no genres listed)")

    movies["clean_title"] = movies["title"].apply(_clean_title)
    movies["year"] = movies["title"].apply(_extract_year)
    movies["genres_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    return movies


def load_ratings(path: str = RATINGS_PATH) -> pd.DataFrame:
    """Load ratings.csv as-is (userId, movieId, rating, timestamp)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Make sure ratings.csv is inside the 'data' folder."
        )
    return pd.read_csv(path)


def compute_average_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ratings.csv into one row per movie:
      movieId, avg_rating, rating_count
    Used only to enrich the display (NOT used for the similarity model,
    since this project uses content-based filtering only).
    """
    agg = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )
    agg["avg_rating"] = agg["avg_rating"].round(2)
    return agg
