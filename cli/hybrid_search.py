import os

from inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(os.path.join("cache", "index.pkl")):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [1.0 for _ in scores]

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def weighted_search(
        self,
        query: str,
        alpha: float,
        limit: int = 5,
    ) -> list[dict]:
        fetch_limit = limit * 500

        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(
            query,
            limit=fetch_limit,
        )

        document_map = {doc["id"]: doc for doc in self.documents}
        combined = {}

        def get_entry(doc_id):
            if doc_id not in combined:
                doc = document_map.get(doc_id, {})
                combined[doc_id] = {
                    "id": doc_id,
                    "title": doc.get("title", "Unknown"),
                    "document": doc.get("description", ""),
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            return combined[doc_id]

        bm25_pairs = []
        for result in bm25_results:
            if isinstance(result, dict):
                bm25_pairs.append((result["id"], result["score"]))
            else:
                bm25_pairs.append((result[0], result[1]))

        normalized_bm25_scores = self.normalize(
            [score for _, score in bm25_pairs]
        )

        for (doc_id, _), normalized_score in zip(
            bm25_pairs,
            normalized_bm25_scores,
        ):
            get_entry(doc_id)["bm25_score"] = normalized_score

        semantic_pairs = []
        for result in semantic_results:
            if isinstance(result, dict):
                semantic_pairs.append((result["id"], result["score"]))
            else:
                semantic_pairs.append((result[0], result[1]))

        normalized_semantic_scores = self.normalize(
            [score for _, score in semantic_pairs]
        )

        for (doc_id, _), normalized_score in zip(
            semantic_pairs,
            normalized_semantic_scores,
        ):
            get_entry(doc_id)["semantic_score"] = normalized_score

        def hybrid_score(
            bm25_score: float,
            semantic_score: float,
        ) -> float:
            return alpha * bm25_score + (1 - alpha) * semantic_score

        for entry in combined.values():
            entry["hybrid_score"] = hybrid_score(
                entry["bm25_score"],
                entry["semantic_score"],
            )

        return sorted(
            combined.values(),
            key=lambda entry: entry["hybrid_score"],
            reverse=True,
        )

    def rrf_search(
        self,
        query: str,
        k: int,
        limit: int = 5,
    ) -> list[dict]:
        print(f"RRF started: limit={limit}", flush=True)

        fetch_limit = limit * 500
        print(f"RRF fetch_limit={fetch_limit}", flush=True)

        print("Starting BM25...", flush=True)
        bm25_results = self._bm25_search(query, fetch_limit)
        print(
            f"BM25 done: {len(bm25_results)} results",
            flush=True,
        )

        print("Starting semantic search...", flush=True)
        semantic_results = self.semantic_search.search_chunks(
            query,
            limit=fetch_limit,
        )
        print(
            f"Semantic done: {len(semantic_results)} results",
            flush=True,
        )

        document_map = {doc["id"]: doc for doc in self.documents}
        combined = {}

        def get_entry(doc_id):
            if doc_id not in combined:
                doc = document_map.get(doc_id, {})
                combined[doc_id] = {
                    "id": doc_id,
                    "title": doc.get("title", "Unknown"),
                    "document": doc.get("description", ""),
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }
            return combined[doc_id]

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = (
                result["id"]
                if isinstance(result, dict)
                else result[0]
            )

            entry = get_entry(doc_id)
            entry["bm25_rank"] = rank
            entry["rrf_score"] += 1.0 / (k + rank)

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = (
                result["id"]
                if isinstance(result, dict)
                else result[0]
            )

            entry = get_entry(doc_id)
            entry["semantic_rank"] = rank
            entry["rrf_score"] += 1.0 / (k + rank)

        return sorted(
            combined.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:limit]