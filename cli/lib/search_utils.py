import json

SCORE_PRECISION = 4


def load_movies(filepath: str = "data/movies.json") -> list[dict]:
    """Load movie documents from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("movies", [])


def format_search_result(doc_id, title, document, score, metadata=None):
    """
    Format a search result as a dictionary.
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }