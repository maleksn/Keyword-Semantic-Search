import json
import math
import os
import pickle
import string
from collections import Counter
from nltk.stem import PorterStemmer

# BM25 constants
BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"


def tokenize_text(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    stemmer = PorterStemmer()

    with open("data/stopwords.txt", "r") as f:
        stopwords = set(
            w.lower().translate(translator) for w in f.read().splitlines()
        )

    tokens = text.lower().translate(translator).split()
    return [stemmer.stem(t) for t in tokens if t not in stopwords]


def tokenize_single_term(term: str) -> str:
    """
    Tokenize a single term using the same pipeline.
    Raises ValueError if the result is not exactly one token.
    """
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(
            f"Expected exactly one token after processing, but got {len(tokens)}: {tokens}"
        )
    return tokens[0]


class InvertedIndex:

    def __init__(self):
        self.index = {}            # token -> set(doc_ids)
        self.docmap = {}           # doc_id -> movie object
        self.term_frequencies = {} # doc_id -> Counter(token -> count)
        self.doc_lengths = {}      # doc_id -> number of tokens
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        # Update inverted index
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        # Update term frequencies
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(tokens)

        # Store document length (token count)
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term: str) -> list:
        return sorted(list(self.index.get(term, set())))

    def get_tf(self, doc_id, term: str) -> int:
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term, 0)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(term, set()))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        L = self.doc_lengths.get(doc_id, 0)
        avg = self.__get_avg_doc_length()

        if avg == 0:
            ratio = 0.0
        else:
            ratio = L / avg

        denominator = tf + k1 * (1 - b + b * ratio)
        return (tf * (k1 + 1)) / denominator

    def bm25(self, doc_id, term: str) -> float:
        """Full BM25 score for a document and a single token."""
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple]:
        """
        Rank all documents by BM25 score for the given query.
        Returns a list of (doc_id, score) sorted by score descending, limited to `limit`.
        """
        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = {}
        for doc_id in self.docmap:
            score = sum(self.bm25(doc_id, token) for token in tokens)
            if score > 0:  # skip irrelevant documents
                scores[doc_id] = score

        # Sort by score descending and take top `limit`
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:limit]

    def build(self):
        with open("data/movies.json", "r") as f:
            data = json.load(f)

        for movie in data.get("movies", []):
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            full_text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, full_text)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not (os.path.exists(os.path.join(CACHE_DIR, "index.pkl")) and
                os.path.exists(os.path.join(CACHE_DIR, "docmap.pkl")) and
                os.path.exists(os.path.join(CACHE_DIR, "term_frequencies.pkl")) and
                os.path.exists(self.doc_lengths_path)):
            raise FileNotFoundError(
                "Index files not found. Please run 'build' first."
            )
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as f:
            self.docmap = pickle.load(f)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    token = tokenize_single_term(term)
    return idx.get_bm25_idf(token)


def bm25_tf_command(doc_id, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    token = tokenize_single_term(term)
    return idx.get_bm25_tf(doc_id, token, k1, b)