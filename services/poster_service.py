"""
poster_service.py
-------------------
Optional poster lookup via the OMDb API (https://www.omdbapi.com/).

Design goals:
- The app must work perfectly with ZERO api key configured.
- Any network error, missing key, or "not found" response should degrade
  gracefully to a placeholder -- never crash the Streamlit app.
"""

import os
from functools import lru_cache
from typing import Optional

import requests

OMDB_URL = "https://www.omdbapi.com/"
REQUEST_TIMEOUT_SECONDS = 5
PLACEHOLDER_TEXT = "Poster not available"


def is_poster_service_enabled() -> bool:
    """True only if an OMDb API key has been configured via environment variables."""
    api_key = os.getenv("OMDB_API_KEY", "").strip()
    return bool(api_key)


@lru_cache(maxsize=512)
def fetch_poster_url(title: str, year: Optional[int] = None) -> Optional[str]:
    """
    Try to fetch a poster URL for `title` (optionally narrowed by `year`).

    Returns:
        A poster image URL (str) on success.
        None if no API key is set, the movie wasn't found, or a network
        error occurred -- callers should show PLACEHOLDER_TEXT in that case.
    """
    api_key = os.getenv("OMDB_API_KEY", "").strip()
    if not api_key:
        return None

    params = {"apikey": api_key, "t": title, "type": "movie"}
    if year:
        params["y"] = year

    try:
        response = requests.get(OMDB_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        # Network problem, timeout, or invalid JSON -- fail silently
        return None

    if data.get("Response") != "True":
        return None

    poster = data.get("Poster")
    if not poster or poster == "N/A":
        return None

    return poster
