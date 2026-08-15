"""
app.py
-------
Streamlit front-end for a beginner-friendly, CONTENT-BASED movie
recommendation system built on the MovieLens "small" dataset.

Run with:
    streamlit run app.py

No API keys are required. If an OMDb API key is set (see .env.example),
poster images will be shown; otherwise the app shows a clean placeholder.
"""

import streamlit as st
from dotenv import load_dotenv

from models.recommender import ContentBasedRecommender
from services.poster_service import fetch_poster_url, is_poster_service_enabled, PLACEHOLDER_TEXT
from utils.data_loader import load_movies, load_ratings, compute_average_ratings

# Load variables from a local .env file (e.g. OMDB_API_KEY) if present.
# Safe to call even if no .env file exists.
load_dotenv()


# ---------------------------------------------------------------------- #
# Page config -- must be the first Streamlit call
# ---------------------------------------------------------------------- #
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------- #
# Cached data / model loading
# Streamlit re-runs the whole script on every interaction, so we cache
# the expensive parts (reading CSVs, building the TF-IDF model) so the
# app stays fast after the first load.
# ---------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_movies():
    return load_movies()


@st.cache_data(show_spinner=False)
def get_rating_stats():
    ratings = load_ratings()
    return compute_average_ratings(ratings)


@st.cache_resource(show_spinner=False)
def get_recommender(_movies):
    # Leading underscore tells Streamlit not to try to hash the DataFrame
    return ContentBasedRecommender(_movies)


def load_app_state():
    """Load data + model once, with a friendly error message on failure."""
    try:
        movies = get_movies()
        rating_stats = get_rating_stats()
        recommender = get_recommender(movies)
        return movies, rating_stats, recommender, None
    except FileNotFoundError as exc:
        return None, None, None, str(exc)


# ---------------------------------------------------------------------- #
# Small UI helpers
# ---------------------------------------------------------------------- #
def render_movie_card(rec, poster_enabled: bool):
    """Render one recommendation as a card inside the current column."""
    with st.container(border=True):
        poster_url = None
        if poster_enabled:
            poster_url = fetch_poster_url(rec.title, rec.year)

        if poster_url:
            st.image(poster_url)
        else:
            # Graceful fallback -- no crash, no API key required
            st.markdown(
                f"""
                <div style="
                    height:260px; display:flex; align-items:center; justify-content:center;
                    background-color:#2b2b2b; color:#bbbbbb; border-radius:8px;
                    text-align:center; padding:10px; font-size:14px;">
                    🎞️<br>{PLACEHOLDER_TEXT}
                </div>
                """,
                unsafe_allow_html=True,
            )

        title_display = rec.title + (f" ({rec.year})" if rec.year else "")
        st.markdown(f"**{title_display}**")
        st.caption(rec.genres.replace("|", " • "))

        st.progress(min(max(rec.similarity, 0.0), 1.0))
        st.write(f"Similarity score: **{rec.similarity:.2f}**")

        if rec.avg_rating is not None:
            st.write(f"⭐ {rec.avg_rating} / 5  ({rec.rating_count} ratings)")


# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
def render_sidebar(movies):
    st.sidebar.title("🎬 Movie Recommender")
    st.sidebar.write("Content-based recommendations using movie genres.")
    st.sidebar.divider()

    search_query = st.sidebar.text_input(
        "Search for a movie",
        placeholder="e.g. Toy Story",
        help="Type part of a movie title to search the MovieLens catalog.",
    )

    matching_titles = []
    if search_query:
        matching_titles = sorted(set(
            title for title in movies["clean_title"] if search_query.lower() in title.lower()
        ))

    selected_title = None
    if matching_titles:
        selected_title = st.sidebar.selectbox(
            "Select a movie",
            options=matching_titles,
            index=0,
        )
    elif search_query:
        st.sidebar.warning("No movies matched your search. Try a different title.")

    num_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.sidebar.divider()
    if is_poster_service_enabled():
        st.sidebar.success("OMDb poster lookup: enabled")
    else:
        st.sidebar.info("OMDb poster lookup: disabled\n\nAdd OMDB_API_KEY to a .env file to show posters.")

    return selected_title, num_recommendations


# ---------------------------------------------------------------------- #
# Main page
# ---------------------------------------------------------------------- #
def render_main(recommender, rating_stats, selected_title, num_recommendations):
    st.title("🎬 Content-Based Movie Recommendation System")
    st.write(
        "Pick a movie in the sidebar and get similar recommendations based on "
        "genre content -- powered by TF-IDF and cosine similarity, no user "
        "history or collaborative filtering involved."
    )

    if not selected_title:
        st.info("👈 Search for a movie in the sidebar to get started.")
        return

    st.subheader(f"Because you selected: {selected_title}")

    try:
        recommendations = recommender.recommend(selected_title, top_n=num_recommendations)
    except ValueError as exc:
        st.error(str(exc))
        return

    if not recommendations:
        st.warning("No similar movies were found.")
        return

    recommendations = recommender.attach_rating_stats(recommendations, rating_stats)

    poster_enabled = is_poster_service_enabled()

    # Display recommendation cards in a responsive grid (4 per row)
    cards_per_row = 4
    for start in range(0, len(recommendations), cards_per_row):
        row_items = recommendations[start:start + cards_per_row]
        columns = st.columns(len(row_items))
        for column, rec in zip(columns, row_items):
            with column:
                render_movie_card(rec, poster_enabled)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main():
    movies, rating_stats, recommender, error = load_app_state()

    if error:
        st.title("🎬 Movie Recommendation System")
        st.error(
            "Could not load the MovieLens dataset.\n\n"
            f"Details: {error}\n\n"
            "Make sure `movies.csv` and `ratings.csv` are present inside the `data/` folder."
        )
        return

    selected_title, num_recommendations = render_sidebar(movies)
    render_main(recommender, rating_stats, selected_title, num_recommendations)


if __name__ == "__main__":
    main()
