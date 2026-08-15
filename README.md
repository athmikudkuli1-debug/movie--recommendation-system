# 🎬 Movie Recommendation System (Content-Based Filtering)

A beginner-friendly movie recommender built with **Python** and **Streamlit**,
using **only content-based filtering** on the MovieLens `ml-latest-small`
dataset. No collaborative filtering, no `scikit-surprise`, no FAISS, no
sentence-transformers, no Anthropic/OpenAI APIs, and **no API key required**
to run the app.

---

## How it works

1. Each movie's genres (e.g. `Adventure|Animation|Comedy`) are converted
   into a text "soup" like `adventure animation comedy`.
2. **TF-IDF** (`scikit-learn`'s `TfidfVectorizer`) turns that text into a
   numeric vector for every movie.
3. **Cosine similarity** compares every movie's vector to every other
   movie's vector.
4. When you pick a movie, the app returns the N most similar movies by
   cosine similarity score (0 to 1, higher = more similar).

This is pure **content-based filtering** — recommendations come only from
each movie's own attributes (genres), never from other users' ratings.

---

## Project structure

```
movie_rec_system/
│
├── app.py                  # Streamlit UI (entry point)
├── requirements.txt
├── README.md
├── .env.example             # Copy to .env to enable poster images (optional)
│
├── data/
│   ├── movies.csv           # MovieLens ml-latest-small
│   └── ratings.csv          # MovieLens ml-latest-small
│
├── models/
│   ├── __init__.py
│   └── recommender.py       # TF-IDF + cosine similarity content-based model
│
├── services/
│   ├── __init__.py
│   └── poster_service.py    # Optional OMDb poster lookup (safe fallback)
│
└── utils/
    ├── __init__.py
    └── data_loader.py       # CSV loading / cleaning helpers
```

---

## Requirements

- Python **3.11**
- Works on **Windows without Visual Studio Build Tools** — every package
  in `requirements.txt` ships pre-built wheels, so nothing needs to be
  compiled locally.

Dependencies (see `requirements.txt`):
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `requests`
- `python-dotenv`

---

## Setup instructions

### 1. Get the project files
Make sure the folder structure above is intact, with `movies.csv` and
`ratings.csv` inside the `data/` folder. (This project already ships with
the MovieLens `ml-latest-small` dataset included.)

### 2. Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Enable movie posters
Posters are fetched from the free [OMDb API](https://www.omdbapi.com/apikey.aspx).
This is entirely optional — without a key, the app shows a clean
"Poster not available" placeholder instead of crashing.

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```
Then open `.env` and paste your key:
```
OMDB_API_KEY=your_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```
Streamlit will open the app automatically at `http://localhost:8501`.

---

## Using the app

1. Use the **sidebar** to search for a movie title (e.g. "Toy Story").
2. Select the exact movie from the dropdown that appears.
3. Adjust the **"Number of recommendations"** slider (5–20).
4. The main page displays recommendation cards showing:
   - Movie title (with release year)
   - Genres
   - Similarity score (0–1, progress bar + numeric value)
   - Average rating & rating count (from `ratings.csv`, for context only —
     not used by the similarity model)
   - Poster image if `OMDB_API_KEY` is configured, otherwise a placeholder

---

## Why content-based filtering only?

| Approach | Used here? | Notes |
|---|---|---|
| Content-based filtering (TF-IDF + cosine similarity on genres) | ✅ Yes | Works from day one, no user history needed, no cold-start problem for new users |
| Collaborative filtering | ❌ No | Explicitly excluded per requirements |
| `scikit-surprise` | ❌ No | Requires C++ build tools on Windows — excluded |
| FAISS | ❌ No | Excluded per requirements |
| Sentence-Transformers | ❌ No | Excluded per requirements |
| Anthropic / OpenAI APIs | ❌ No | Excluded per requirements — no API key needed to run the core app |

---

## Troubleshooting

- **"Could not load the MovieLens dataset"** — confirm `movies.csv` and
  `ratings.csv` exist inside the `data/` folder.
- **Posters never show up** — check that `.env` exists, contains a valid
  `OMDB_API_KEY`, and that you have an internet connection. The app will
  still work correctly without posters.
- **Slow first load** — the TF-IDF model and similarity matrix are built
  once and cached by Streamlit (`@st.cache_resource`); subsequent
  interactions are fast.

---

## Dataset credit

This project uses the **MovieLens "ml-latest-small"** dataset
(9,742 movies, 100,836 ratings from 610 users), created by
[GroupLens Research](https://grouplens.org/datasets/movielens/) at the
University of Minnesota. Citation:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
> History and Context. ACM Transactions on Interactive Intelligent
> Systems (TiiS) 5, 4, Article 19.

The dataset is used here for educational purposes only.
