from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import config
from .embedder import Embedder, FAISSRetriever


def tokenize_for_overlap(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def lexical_overlap_score(query: str, paragraph_text: str) -> float:
    query_tokens = tokenize_for_overlap(query)
    if not query_tokens:
        return 0.0
    para_tokens = set(tokenize_for_overlap(paragraph_text))
    if not para_tokens:
        return 0.0
    overlap = sum(1 for token in query_tokens if token in para_tokens)
    return overlap / max(1, len(set(query_tokens)))


def _normalize_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    if float(arr.max() - arr.min()) < 1e-8:
        return [1.0 for _ in values]
    normalized = (arr - arr.min()) / (arr.max() - arr.min())
    return normalized.tolist()


def hybrid_rerank(
    query: str,
    retrieved: Sequence[Dict],
    *,
    dense_weight: float = config.HYBRID_DENSE_WEIGHT,
    lexical_weight: float = config.HYBRID_LEXICAL_WEIGHT,
) -> List[Dict]:
    if not retrieved:
        return []
    dense_scores = [float(item.get("score", 0.0)) for item in retrieved]
    lexical_scores = [lexical_overlap_score(query, item.get("text", "")) for item in retrieved]
    dense_norm = _normalize_scores(dense_scores)
    lexical_norm = _normalize_scores(lexical_scores)

    reranked = []
    for item, dense_score, lexical_score, dense_scaled, lexical_scaled in zip(
        retrieved,
        dense_scores,
        lexical_scores,
        dense_norm,
        lexical_norm,
    ):
        hybrid_score = dense_weight * dense_scaled + lexical_weight * lexical_scaled
        reranked.append(
            {
                **item,
                "dense_score": dense_score,
                "lexical_score": lexical_score,
                "hybrid_score": hybrid_score,
                "score": hybrid_score,
            }
        )
    reranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return reranked


def retrieve_references(
    retriever: FAISSRetriever,
    doc_id: str,
    query: str,
    top_k: int = config.RETRIEVAL_CANDIDATE_K,
) -> List[str]:
    results = hybrid_rerank(query, retriever.retrieve(doc_id, query, top_k))
    return [r["para_id"] for r in results[:top_k]]


def get_top_references_by_score(
    retriever: FAISSRetriever,
    doc_id: str,
    query: str,
    n: int = config.REFERENCE_TOP_N,
) -> List[Tuple[str, float]]:
    results = hybrid_rerank(query, retriever.retrieve(doc_id, query, top_k=max(n * 2, n)))
    return [(r["para_id"], r["score"]) for r in results[:n]]


def cross_encode_rerank(
    query: str,
    paragraphs: List[Dict],
    embedder: Embedder,
    top_k: int = config.RETRIEVAL_CANDIDATE_K,
) -> List[Dict]:
    if not paragraphs:
        return []

    query_emb = embedder.encode_query(query)
    para_embs, para_ids = embedder.encode_paragraphs(paragraphs)

    dense_scores = np.dot(para_embs, query_emb)
    prelim = []
    for idx in np.argsort(dense_scores)[::-1][:top_k]:
        prelim.append(
            {
                "para_id": para_ids[idx],
                "text": paragraphs[idx]["text"],
                "score": float(dense_scores[idx]),
            }
        )
    return hybrid_rerank(query, prelim)


def select_references_from_retrieved(
    retrieved: Sequence[Dict],
    n: int = config.REFERENCE_TOP_N,
    score_threshold: float | None = None,
) -> List[str]:
    if not retrieved:
        return []
    selected = []
    for item in retrieved:
        if score_threshold is None or item["score"] >= score_threshold:
            selected.append(item["para_id"])
        if len(selected) >= n:
            break
    if len(selected) < n:
        for item in retrieved:
            if item["para_id"] in selected:
                continue
            selected.append(item["para_id"])
            if len(selected) >= n:
                break
    return selected


def compute_retrieval_metrics(
    gold_refs_list: Sequence[Sequence[str]],
    retrieved_list: Sequence[Sequence[Dict]],
) -> Dict[str, float]:
    total = max(1, len(gold_refs_list))
    hit1 = hit3 = hit10 = 0
    recall10 = []
    iou3 = []
    for gold_refs, retrieved in zip(gold_refs_list, retrieved_list):
        gold_set = set(gold_refs or [])
        top1 = {item["para_id"] for item in retrieved[:1]}
        top3 = {item["para_id"] for item in retrieved[:3]}
        top10 = {item["para_id"] for item in retrieved[:10]}
        if gold_set & top1:
            hit1 += 1
        if gold_set & top3:
            hit3 += 1
        if gold_set & top10:
            hit10 += 1
        if gold_set:
            recall10.append(len(gold_set & top10) / len(gold_set))
            iou3.append(len(gold_set & top3) / len(gold_set | top3) if (gold_set | top3) else 0.0)
    return {
        "hit_any_gold_at_1": hit1 / total,
        "hit_any_gold_at_3": hit3 / total,
        "hit_any_gold_at_10": hit10 / total,
        "mean_ref_recall_at_10": float(np.mean(recall10)) if recall10 else 0.0,
        "mean_iou_top_3": float(np.mean(iou3)) if iou3 else 0.0,
    }


if __name__ == "__main__":
    print("Retrieval module loaded successfully")
