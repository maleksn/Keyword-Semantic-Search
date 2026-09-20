"""
Tool Registry & Search Dispatcher for the AI Agent.
Exposes BM25, Semantic Chunked Search, Hybrid RRF/Weighted Search,
Multimodal Image Search, and Cross-Encoder Re-ranking as structured tools.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure cli/ directory is accessible for existing search modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CLI_DIR = os.path.join(PROJECT_ROOT, "cli")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from inverted_index import InvertedIndex
from lib.search_utils import load_movies
from lib.semantic_search import ChunkedSemanticSearch


class SearchTools:
    """Encapsulates and caches all retrieval and ranking components."""

    def __init__(self, data_path: str = "data/movies.json") -> None:
        self.data_path = os.path.join(PROJECT_ROOT, data_path)
        self._documents: Optional[List[Dict[str, Any]]] = None
        self._doc_map: Optional[Dict[int, Dict[str, Any]]] = None
        self._inverted_index: Optional[InvertedIndex] = None
        self._semantic_search: Optional[ChunkedSemanticSearch] = None
        self._multimodal_search = None
        self._cross_encoder = None

    @property
    def documents(self) -> List[Dict[str, Any]]:
        if self._documents is None:
            self._documents = load_movies(self.data_path)
        return self._documents

    @property
    def doc_map(self) -> Dict[int, Dict[str, Any]]:
        if self._doc_map is None:
            self._doc_map = {doc["id"]: doc for doc in self.documents}
        return self._doc_map

    @property
    def inverted_index(self) -> InvertedIndex:
        if self._inverted_index is None:
            idx = InvertedIndex()
            idx.load()
            self._inverted_index = idx
        return self._inverted_index

    @property
    def semantic_search_engine(self) -> ChunkedSemanticSearch:
        if self._semantic_search is None:
            engine = ChunkedSemanticSearch()
            engine.load_or_create_chunk_embeddings(self.documents)
            self._semantic_search = engine
        return self._semantic_search

    def bm25_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Execute BM25 exact/keyword search against the inverted index.
        Best for specific character names, exact titles, quotes, or keywords.
        """
        raw_results = self.inverted_index.bm25_search(query, limit=limit)
        results = []
        for doc_id, score in raw_results:
            doc = self.doc_map.get(doc_id, {})
            results.append({
                "id": doc_id,
                "title": doc.get("title", "Unknown"),
                "description": doc.get("description", "")[:250],
                "bm25_score": round(float(score), 4),
            })
        return results

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Execute dense vector semantic search across chunked descriptions.
        Best for broad themes, vibes, emotions, plots, or abstract queries.
        """
        raw_results = self.semantic_search_engine.search_chunks(query, limit=limit)
        results = []
        for res in raw_results:
            results.append({
                "id": res["id"],
                "title": res["title"],
                "description": res["document"][:250],
                "similarity_score": round(float(res["score"]), 4),
            })
        return results

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Execute Reciprocal Rank Fusion (RRF) combining BM25 and dense semantic search.
        This is the recommended default search tool for general movie queries.
        """
        fetch_limit = max(limit * 20, 50)
        bm25_raw = self.inverted_index.bm25_search(query, limit=fetch_limit)
        semantic_raw = self.semantic_search_engine.search_chunks(query, limit=fetch_limit)

        combined: Dict[int, Dict[str, Any]] = {}

        def get_entry(doc_id: int):
            if doc_id not in combined:
                doc = self.doc_map.get(doc_id, {})
                combined[doc_id] = {
                    "id": doc_id,
                    "title": doc.get("title", "Unknown"),
                    "description": doc.get("description", "")[:250],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }
            return combined[doc_id]

        for rank, item in enumerate(bm25_raw, start=1):
            doc_id = item[0] if isinstance(item, (list, tuple)) else item["id"]
            entry = get_entry(doc_id)
            entry["bm25_rank"] = rank
            entry["rrf_score"] += 1.0 / (k + rank)

        for rank, item in enumerate(semantic_raw, start=1):
            doc_id = item[0] if isinstance(item, (list, tuple)) else item["id"]
            entry = get_entry(doc_id)
            entry["semantic_rank"] = rank
            entry["rrf_score"] += 1.0 / (k + rank)

        ranked = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        for r in ranked[:limit]:
            r["rrf_score"] = round(r["rrf_score"], 5)
        return ranked[:limit]

    def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        """
        Retrieve full details and metadata for a specific movie by its numeric ID.
        """
        doc = self.doc_map.get(movie_id)
        if not doc:
            return {"error": f"Movie with ID {movie_id} not found in database."}
        return {
            "id": doc["id"],
            "title": doc.get("title", "Unknown"),
            "description": doc.get("description", ""),
            "metadata": {k: v for k, v in doc.items() if k not in ["id", "title", "description"]},
        }

    def multimodal_image_search(self, image_path: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search movies using an image/poster file via CLIP multimodal embeddings.
        """
        if self._multimodal_search is None:
            from lib.multimodal_search import MultimodalSearch
            self._multimodal_search = MultimodalSearch(self.documents)

        clean_path = os.path.abspath(image_path)
        if not os.path.exists(clean_path):
            return [{"error": f"Image file not found at path: {image_path}"}]

        raw_results = self._multimodal_search.search_with_image(clean_path, limit=limit)
        results = []
        for r in raw_results:
            results.append({
                "id": r["id"],
                "title": r["title"],
                "description": r["description"][:250],
                "similarity": round(float(r.get("similarity", 0.0)), 4),
            })
        return results

    def cross_encode_rerank(
        self,
        query: str,
        candidate_ids: List[int],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank a list of candidate movie IDs using a neural Cross-Encoder model.
        Useful when you have a list of search results and want fine-grained re-ordering.
        """
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            try:
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            except Exception:
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")

        candidates = [self.doc_map[cid] for cid in candidate_ids if cid in self.doc_map]
        if not candidates:
            return []

        pairs = [[query, f"{doc.get('title', '')} - {doc.get('description', '')}"] for doc in candidates]
        scores = self._cross_encoder.predict(pairs)

        ranked = []
        for doc, score in zip(candidates, scores):
            ranked.append({
                "id": doc["id"],
                "title": doc.get("title", "Unknown"),
                "description": doc.get("description", "")[:250],
                "cross_encoder_score": round(float(score), 4),
            })

        ranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return ranked[:top_k]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Route and execute a tool call by name with arguments."""
        if tool_name == "hybrid_search":
            return self.hybrid_search(
                query=arguments.get("query", ""),
                limit=int(arguments.get("limit", 5)),
            )
        elif tool_name == "bm25_search":
            return self.bm25_search(
                query=arguments.get("query", ""),
                limit=int(arguments.get("limit", 5)),
            )
        elif tool_name == "semantic_search":
            return self.semantic_search(
                query=arguments.get("query", ""),
                limit=int(arguments.get("limit", 5)),
            )
        elif tool_name == "get_movie_details":
            return self.get_movie_details(
                movie_id=int(arguments.get("movie_id", 0)),
            )
        elif tool_name == "multimodal_image_search":
            return self.multimodal_image_search(
                image_path=arguments.get("image_path", ""),
                limit=int(arguments.get("limit", 5)),
            )
        elif tool_name == "cross_encode_rerank":
            return self.cross_encode_rerank(
                query=arguments.get("query", ""),
                candidate_ids=arguments.get("candidate_ids", []),
                top_k=int(arguments.get("top_k", 5)),
            )
        else:
            return {"error": f"Unknown tool: '{tool_name}'"}


# ---------------------------------------------------------------------------
# OpenAI-Compatible Tool Definitions Schema
# ---------------------------------------------------------------------------

OPENAI_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": (
                "Search movies using hybrid fusion (BM25 + Semantic Vector Search via RRF). "
                "Recommended default search tool for almost all user queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, concepts, genre, or keywords.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of top results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bm25_search",
            "description": (
                "Lexical keyword and BM25 search. Use when the user asks for exact titles, "
                "specific actor/character names, or exact phrases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Exact keywords or names to search for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Dense vector similarity search using sentence embeddings. Best for abstract themes, "
                "moods, feelings, and conceptual descriptions without specific keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The thematic or conceptual query description.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": (
                "Retrieve full description and metadata for a specific movie by its ID. "
                "Use this after search when the user wants to know more about a specific movie."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "The unique integer ID of the movie.",
                    },
                },
                "required": ["movie_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multimodal_image_search",
            "description": (
                "Search movies based on an image file (poster, scene photo) using CLIP embeddings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "File path to the image.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5).",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_encode_rerank",
            "description": (
                "Re-rank a list of movie IDs using a neural cross-encoder for higher precision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query to rank against.",
                    },
                    "candidate_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of movie IDs retrieved from prior search to re-rank.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top re-ranked movies to return (default: 5).",
                    },
                },
                "required": ["query", "candidate_ids"],
            },
        },
    },
]
