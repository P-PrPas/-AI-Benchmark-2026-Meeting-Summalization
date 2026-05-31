from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from . import config
from .retrieval import lexical_overlap_score, tokenize_for_overlap


def _bm25_scores(query: str, paragraphs: Sequence[dict[str, Any]], *, field: str = "text") -> list[tuple[int, float]]:
    tokenized_docs = [tokenize_for_overlap(str(paragraph.get(field, paragraph.get("text", "")))) for paragraph in paragraphs]
    query_terms = tokenize_for_overlap(query)
    if not paragraphs or not query_terms:
        return []

    doc_freq = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))
    avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, len(tokenized_docs))
    k1 = 1.5
    b = 0.75
    results = []
    for idx, tokens in enumerate(tokenized_docs):
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (len(paragraphs) - df + 0.5) / (df + 0.5))
            denom = tf[term] + k1 * (1 - b + b * doc_len / max(avg_len, 1e-6))
            score += idf * (tf[term] * (k1 + 1) / max(denom, 1e-6))
        results.append((idx, float(score)))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def _quote_text(paragraph: dict[str, Any]) -> str:
    text = str(paragraph.get("text", ""))
    tokens = tokenize_for_overlap(text)
    first_tokens = " ".join(tokens[:24])
    return f"{first_tokens} {text}"


def expanded_retrieve_paragraphs(
    doc_embedding_index: dict[str, dict[str, Any]],
    doc_id: str,
    query: str,
    embedder: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    payload = doc_embedding_index[doc_id]
    paragraphs = list(payload["paragraphs"])
    query_embedding = embedder.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    dense_scores = payload["embeddings"] @ query_embedding
    dense_indices = np.argsort(dense_scores)[-config.EXPANDED_DENSE_TOP_K:][::-1].tolist()

    bm25 = _bm25_scores(query, paragraphs)[: config.EXPANDED_BM25_TOP_K]
    quote_paragraphs = [{**paragraph, "quote_text": _quote_text(paragraph)} for paragraph in paragraphs]
    quote = _bm25_scores(query, quote_paragraphs, field="quote_text")[: config.EXPANDED_QUOTE_TOP_K]

    rank_votes: dict[int, float] = defaultdict(float)
    raw_scores: dict[int, dict[str, float]] = defaultdict(dict)
    for rank, idx in enumerate(dense_indices, start=1):
        rank_votes[idx] += 1.0 / (60 + rank)
        raw_scores[idx]["dense_score"] = float(dense_scores[idx])
    for rank, (idx, score) in enumerate(bm25, start=1):
        rank_votes[idx] += 1.0 / (60 + rank)
        raw_scores[idx]["bm25_score"] = float(score)
    for rank, (idx, score) in enumerate(quote, start=1):
        rank_votes[idx] += 1.0 / (60 + rank)
        raw_scores[idx]["quote_score"] = float(score)

    ranked = sorted(rank_votes.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results = []
    for idx, fused_score in ranked:
        paragraph = paragraphs[idx]
        results.append(
            {
                "para_id": paragraph["para_id"],
                "text": paragraph["text"],
                "score": float(fused_score),
                "dense_score": raw_scores[idx].get("dense_score", float(dense_scores[idx])),
                "bm25_score": raw_scores[idx].get("bm25_score", 0.0),
                "quote_score": raw_scores[idx].get("quote_score", 0.0),
                "lexical_score": lexical_overlap_score(query, paragraph.get("text", "")),
            }
        )
    return results


def expanded_retrieve_from_dense(
    query: str,
    paragraphs: Sequence[dict[str, Any]],
    dense_candidates: Sequence[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Fuse runtime dense candidates with lexical candidate sources."""
    para_id_to_idx = {paragraph["para_id"]: idx for idx, paragraph in enumerate(paragraphs)}
    bm25 = _bm25_scores(query, paragraphs)[: config.EXPANDED_BM25_TOP_K]
    quote_paragraphs = [{**paragraph, "quote_text": _quote_text(paragraph)} for paragraph in paragraphs]
    quote = _bm25_scores(query, quote_paragraphs, field="quote_text")[: config.EXPANDED_QUOTE_TOP_K]

    rank_votes: dict[int, float] = defaultdict(float)
    raw_scores: dict[int, dict[str, float]] = defaultdict(dict)
    for rank, item in enumerate(dense_candidates[: config.EXPANDED_DENSE_TOP_K], start=1):
        idx = para_id_to_idx.get(item.get("para_id"))
        if idx is None:
            continue
        rank_votes[idx] += 1.0 / (60 + rank)
        raw_scores[idx]["dense_score"] = float(item.get("dense_score", item.get("score", 0.0)))
    for rank, (idx, score) in enumerate(bm25, start=1):
        rank_votes[idx] += 1.0 / (60 + rank)
        raw_scores[idx]["bm25_score"] = float(score)
    for rank, (idx, score) in enumerate(quote, start=1):
        rank_votes[idx] += 1.0 / (60 + rank)
        raw_scores[idx]["quote_score"] = float(score)

    ranked = sorted(rank_votes.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results = []
    for idx, fused_score in ranked:
        paragraph = paragraphs[idx]
        results.append(
            {
                "para_id": paragraph["para_id"],
                "text": paragraph["text"],
                "score": float(fused_score),
                "dense_score": raw_scores[idx].get("dense_score", 0.0),
                "bm25_score": raw_scores[idx].get("bm25_score", 0.0),
                "quote_score": raw_scores[idx].get("quote_score", 0.0),
                "lexical_score": lexical_overlap_score(query, paragraph.get("text", "")),
            }
        )
    return results
