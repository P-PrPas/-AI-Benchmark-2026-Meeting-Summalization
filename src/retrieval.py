import numpy as np
from typing import List, Dict, Tuple
from .embedder import Embedder, FAISSRetriever


def retrieve_references(
    retriever: FAISSRetriever,
    doc_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.0
) -> List[str]:
    """Retrieve top-k paragraph IDs for a query.

    Args:
        retriever: FAISS retriever with indexed documents
        doc_id: Document ID to search in
        query: Query text
        top_k: Number of top paragraphs to retrieve
        threshold: Minimum similarity score threshold

    Returns:
        List of paragraph IDs (e.g., ["P3", "P5", "P1"])
    """
    results = retriever.retrieve(doc_id, query, top_k)

    if threshold > 0:
        results = [r for r in results if r["score"] >= threshold]

    return [r["para_id"] for r in results]


def get_top_references_by_score(
    retriever: FAISSRetriever,
    doc_id: str,
    query: str,
    n: int = 3
) -> List[Tuple[str, float]]:
    """Get top-n references with scores.

    Returns:
        List of (para_id, score) tuples
    """
    results = retriever.retrieve(doc_id, query, top_k=n*2)
    return [(r["para_id"], r["score"]) for r in results[:n]]


def cross_encode_rerank(
    query: str,
    paragraphs: List[Dict],
    embedder: Embedder,
    top_k: int = 5
) -> List[Dict]:
    """Rerank paragraphs using cross-encoding scores.

    For more accurate ranking, we compute cross-similarity
    between query and paragraphs.
    """
    if not paragraphs:
        return []

    query_emb = embedder.encode_query(query)
    para_embs, para_ids = embedder.encode_paragraphs(paragraphs)

    scores = np.dot(para_embs, query_emb)
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for idx in sorted_indices[:top_k]:
        results.append({
            "para_id": para_ids[idx],
            "text": paragraphs[idx]["text"],
            "score": float(scores[idx])
        })

    return results


def select_references_from_retrieved(
    retrieved: List[Dict],
    n: int = 3,
    score_threshold: float = 0.5
) -> List[str]:
    """Select top-n references from retrieved results.

    Args:
        retrieved: List of retrieved paragraphs with scores
        n: Number of references to select
        score_threshold: Minimum score to include

    Returns:
        List of paragraph IDs
    """
    selected = []
    for r in retrieved:
        if r["score"] >= score_threshold and len(selected) < n:
            selected.append(r["para_id"])

    if len(selected) < n:
        for r in retrieved[len(selected):]:
            if len(selected) >= n:
                break
            selected.append(r["para_id"])

    return selected


if __name__ == "__main__":
    print("Retrieval module loaded successfully")