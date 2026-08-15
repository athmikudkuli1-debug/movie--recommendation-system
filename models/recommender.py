"""
recommender.py
----------------
Pure content-based movie recommender.

How it works (in plain English):
1. Every movie's genres ("Adventure|Animation|Comedy") are turned into a
   single text "soup" (e.g. "adventure animation comedy").
2. TF-IDF (Term Frequency - Inverse Document Frequency) converts that text
   into a numeric vector for every movie. TF-IDF is part of scikit-learn,
   so no compiled/binary dependencies are needed.
3. Cosine similarity measures how "close" two movies are based on those
   vectors -- movies that share more/rarer genres score higher.
4. To recommend movies similar to a chosen title, we look up its vector's
   similarity to every other movie and return the top N.

No collaborative filtering, no external ML services, no API keys required.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Recommendation:
    """A single recommended movie, ready to be displayed by the UI."""
    movie_id: int
    title: str
    year: Optional[int]
    genres: str
    similarity: float
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None


class ContentBasedRecommender:
    """
    Builds a TF-IDF + cosine-similarity model over movie genres and
    exposes simple search / recommend methods for the Streamlit app.
    """

    def __init__(self, movies: pd.DataFrame):
        """
        Parameters
        ----------
        movies : DataFrame produced by utils.data_loader.load_movies()
                 Must contain: movieId, clean_title, year, genres, genres_list
        """
        self.movies = movies.reset_index(drop=True)
        self._tfidf_matrix = None
        self._similarity_matrix = None
        self._title_to_index = None
        self._build_model()

    # ------------------------------------------------------------------ #
    # Model building
    # ------------------------------------------------------------------ #
    def _build_model(self) -> None:
        """Fit TF-IDF on the genre text and pre-compute cosine similarity."""
        # Turn "Adventure|Animation|Comedy" into "adventure animation comedy"
        genre_soup = self.movies["genres_list"].apply(
            lambda genres: " ".join(g.lower().replace("-", "") for g in genres)
        )

        # Some movies have no genres at all -- TF-IDF needs a non-empty string
        genre_soup = genre_soup.replace("", "unknown")

        vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9]+")
        self._tfidf_matrix = vectorizer.fit_transform(genre_soup)

        # Cosine similarity between every pair of movies (fine for ~9.7k movies)
        self._similarity_matrix = cosine_similarity(self._tfidf_matrix)

        # Fast lookup: lowercase clean title -> row index
        self._title_to_index = pd.Series(
            self.movies.index, index=self.movies["clean_title"].str.lower()
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search_titles(self, query: str, limit: int = 15) -> List[str]:
        """Return a list of movie titles that contain the query (case-insensitive)."""
        if not query:
            return []
        mask = self.movies["clean_title"].str.contains(query, case=False, na=False, regex=False)
        return self.movies.loc[mask, "clean_title"].head(limit).tolist()

    # ------------------------------------------------------------------ #
    # Recommendation
    # ------------------------------------------------------------------ #
    def recommend(self, title: str, top_n: int = 10) -> List[Recommendation]:
        """
        Return the top_n movies most similar to `title` based on genre content.
        Raises ValueError if the title isn't found in the dataset.
        """
        key = title.lower()
        if key not in self._title_to_index:
            raise ValueError(f"'{title}' was not found in the dataset.")

        idx = self._title_to_index[key]
        # If duplicate titles exist, take the first match
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self._similarity_matrix[idx]))
        # Exclude the movie itself, sort by similarity descending
        sim_scores = [s for s in sim_scores if s[0] != idx]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = sim_scores[:top_n]

        results = []
        for movie_idx, score in top_scores:
            row = self.movies.iloc[movie_idx]
            results.append(
                Recommendation(
                    movie_id=int(row["movieId"]),
                    title=row["clean_title"],
                    year=int(row["year"]) if not np.isnan(row["year"]) else None,
                    genres=row["genres"],
                    similarity=round(float(score), 4),
                )
            )
        return results

    def attach_rating_stats(
        self, recommendations: List[Recommendation], ratings_stats: pd.DataFrame
    ) -> List[Recommendation]:
        """Optionally enrich recommendations with avg rating / rating count."""
        stats_lookup = ratings_stats.set_index("movieId")
        for rec in recommendations:
            if rec.movie_id in stats_lookup.index:
                rec.avg_rating = float(stats_lookup.loc[rec.movie_id, "avg_rating"])
                rec.rating_count = int(stats_lookup.loc[rec.movie_id, "rating_count"])
        return recommendations
