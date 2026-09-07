import json
import os
import re
from typing import TypedDict
import numpy as np
from sentence_transformers import SentenceTransformer
from lib.search_utils import format_search_result

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "cache/movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = "cache/chunk_embeddings.npy"
CHUNK_METADATA_PATH = "cache/chunk_metadata.json"


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        """Generate an embedding vector for a single text input."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        """Build embeddings for all documents and save to disk."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        """Load embeddings from disk if they match, else build."""
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for movies semantically similar to the query.
        Returns a list of dicts with keys: score, title, description.
        """
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        results = []

        for i, doc_emb in enumerate(self.embeddings):
            doc = self.documents[i]
            score = cosine_similarity(query_embedding, doc_emb)
            results.append((score, doc))

        # Sort descending by score
        results.sort(key=lambda x: x[0], reverse=True)

        top_results = []
        for score, doc in results[:limit]:
            top_results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })
        return top_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMetadata] = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Build embeddings for chunked document descriptions.
        """
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description or not description.strip():
                continue

            # Use semantic chunking: 4 sentences per chunk, overlap=0
            chunks = chunk_text_semantic(description, max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)

            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })

        # Encode all chunks (remove extra print to match expected output exactly)
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        # Save embeddings and metadata
        os.makedirs("cache", exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": chunk_metadata,
                "total_chunks": len(all_chunks)
            }, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Load chunk embeddings and metadata from cache, or build if missing.
        """
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search across chunk embeddings, returning the best matching movie results.
        """
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        # Step 1: Score each chunk
        chunk_scores = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_emb)
            meta = self.chunk_metadata[i]
            chunk_scores.append({
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": score
            })

        # Step 2: Per movie, keep only the highest chunk score
        movie_best_score = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            if movie_idx not in movie_best_score or cs["score"] > movie_best_score[movie_idx]["score"]:
                movie_best_score[movie_idx] = cs

        # Step 3: Sort movies by score descending
        sorted_movies = sorted(
            movie_best_score.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Step 4: Limit and format results
        results = []
        for item in sorted_movies[:limit]:
            doc = self.documents[item["movie_idx"]]
            result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc.get("description", ""),
                score=item["score"],
                metadata={"chunk_idx": item["chunk_idx"]}
            )
            results.append(result)

        return results


# ---------- top-level helper functions ----------

def verify_model():
    searcher = SemanticSearch()
    model = searcher.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str):
    searcher = SemanticSearch()
    embedding = searcher.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    searcher = SemanticSearch()
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    documents = data.get("movies", [])
    embeddings = searcher.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str):
    searcher = SemanticSearch()
    embedding = searcher.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    """
    Split text into chunks of a fixed number of words, with optional overlap.
    """
    words = text.split()
    total_words = len(words)

    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) must be less than chunk size ({chunk_size})."
        )

    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = 1

    start = 0
    while start < total_words:
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def chunk_text_semantic(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    # 1. Strip leading/trailing whitespace and return [] if empty
    text = text.strip()
    if not text:
        return []

    if overlap >= max_chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) must be less than max_chunk_size ({max_chunk_size})."
        )

    # 2. Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 3. Strip each sentence and filter out empties
    sentences = [s.strip() for s in sentences if s.strip()]

    total_sentences = len(sentences)
    chunks = []
    step = max_chunk_size - overlap
    if step <= 0:
        step = 1

    start = 0
    while start < total_sentences:
        chunk_sentences = sentences[start : start + max_chunk_size]
        # Only keep chunks with at least 2 sentences, unless the whole text is too short
        if len(chunk_sentences) >= 2 or total_sentences < max_chunk_size:
            chunks.append(" ".join(chunk_sentences))
        start += step

    return chunks