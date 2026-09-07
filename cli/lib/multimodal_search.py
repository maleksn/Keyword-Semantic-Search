# File: ./cli/lib/multimodal_search.py
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_movies


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]
        self.text_embeddings = self.model.encode(
            self.texts, show_progress_bar=True
        )

    def embed_image(self, image_path: str):
        """Generate an embedding vector for a single image."""
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        """Search text documents using an image embedding."""
        image_embedding = self.embed_image(image_path)

        results = []
        for i, text_emb in enumerate(self.text_embeddings):
            dot_product = np.dot(image_embedding, text_emb)
            norm_img = np.linalg.norm(image_embedding)
            norm_txt = np.linalg.norm(text_emb)

            if norm_img == 0 or norm_txt == 0:
                similarity = 0.0
            else:
                similarity = float(dot_product / (norm_img * norm_txt))

            doc = self.documents[i]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": similarity,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]


def verify_image_embedding(image_path: str) -> None:
    """Load a CLIP model, generate an image embedding, and print its shape."""
    documents = load_movies()
    searcher = MultimodalSearch(documents)
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    """Load movies, create MultimodalSearch, and search with an image."""
    documents = load_movies()
    searcher = MultimodalSearch(documents)
    return searcher.search_with_image(image_path)