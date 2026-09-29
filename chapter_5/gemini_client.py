"""Create the API client only when an exercise explicitly makes a request."""
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY before running chapter 5 API exercises.")
    from google import genai
    return genai.Client(api_key=key)
