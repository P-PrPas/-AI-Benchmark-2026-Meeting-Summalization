import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import BGE_MODEL_PATH


class Embedder:
    """BGE-M3 embedding wrapper for paragraph and query encoding."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = BGE_MODEL_PATH

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BGE-M3 model from {model_path}...")
        self.model = SentenceTransformer(
            str(model_path),
            device=self.device,
            local_files_only=True,
        )
        print(f"Model loaded on {self.device}")

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        return self.encode([query])[0]

    def encode_paragraphs(self, paragraphs: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Encode paragraphs and return (embeddings, para_ids)."""
        texts = [p["text"] for p in paragraphs]
        para_ids = [p["para_id"] for p in paragraphs]
        embeddings = self.encode(texts)
        return embeddings, para_ids


class FAISSRetriever:
    """FAISS-based retrieval for fast similarity search."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.doc_index = {}  # doc_id -> FAISS index
        self.paragraphs = {}  # doc_id -> list of paragraph dicts

    def index_document(self, doc_id: str, paragraphs: List[Dict]):
        """Index a document's paragraphs for retrieval."""
        if not paragraphs:
            return

        embeddings, para_ids = self.embedder.encode_paragraphs(paragraphs)

        try:
            import faiss
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
            self.doc_index[doc_id] = index
            self.paragraphs[doc_id] = paragraphs
            print(f"Indexed {len(paragraphs)} paragraphs for {doc_id}")
        except ImportError:
            print("FAISS not installed, using numpy fallback")
            self.doc_index[doc_id] = embeddings
            self.paragraphs[doc_id] = paragraphs

    def retrieve(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k paragraphs for a query from a specific document."""
        if doc_id not in self.doc_index:
            return []

        query_emb = self.embedder.encode_query(query)

        try:
            import faiss
            index = self.doc_index[doc_id]
            scores, indices = index.search(query_emb.reshape(1, -1).astype(np.float32), top_k)
            scores = scores[0]
            indices = indices[0]
        except:
            embeddings = self.doc_index[doc_id]
            query_emb = query_emb / np.linalg.norm(query_emb)
            cosine_sim = np.dot(embeddings, query_emb)
            top_indices = np.argsort(cosine_sim)[-top_k:][::-1]
            scores = cosine_sim[top_indices]
            indices = top_indices

        results = []
        for i, idx in enumerate(indices):
            if idx < len(self.paragraphs[doc_id]):
                para = self.paragraphs[doc_id][idx]
                results.append({
                    "para_id": para["para_id"],
                    "text": para["text"],
                    "score": float(scores[i])
                })

        return results

    def batch_retrieve(self, doc_id: str, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """Retrieve for multiple queries at once."""
        return [self.retrieve(doc_id, q, top_k) for q in queries]


if __name__ == "__main__":
    embedder = Embedder()
    test_texts = ["ทดสอบการเข้ารหัส", "บันทึกการประชุม"]
    emb = embedder.encode(test_texts)
    print(f"Embedding shape: {emb.shape}")
