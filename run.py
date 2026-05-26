#!/usr/bin/env python3
"""
CAMNET-P: Thai Parliamentary Meeting Summarization
Docker Model Submission Entry Point
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.generator import Generator
from src.retrieval import hybrid_rerank, select_references_from_retrieved, tokenize_for_overlap

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TEST_PATH = config.TEST_PATH
RESULT_DIR = config.OUTPUT_DIR
RESULT_PATH = RESULT_DIR / "submission.csv"
PROGRESS_LIB = config.PROGRESS_LIB
STARTUP_SLEEP_SECONDS = config.STARTUP_SLEEP_SECONDS


def log_runtime_context():
    print("[info] Runtime configuration")
    print(f"      TEST_PATH={TEST_PATH}")
    print(f"      RESULT_PATH={RESULT_PATH}")
    print(f"      EMBED_MODEL_PATH={config.EMBED_MODEL_PATH}")
    print(f"      LLM_MODEL_PATH={config.LLM_MODEL_PATH}")
    print(f"      RETRIEVAL_CANDIDATE_K={config.RETRIEVAL_CANDIDATE_K}")
    print(f"      REFERENCE_TOP_N={config.REFERENCE_TOP_N}")


def normalize_dataset(data: dict) -> dict:
    docs = data.get("docs") or data.get("documents") or []
    queries = data.get("queries") or data.get("questions") or []
    if not isinstance(docs, list) or not isinstance(queries, list):
        raise ValueError("Dataset must contain list-like docs and queries fields")
    return {"docs": docs, "queries": queries}


def fallback_retrieve(paragraphs: List[Dict], query: str, top_k: int) -> List[Dict]:
    query_tokens = set(tokenize_for_overlap(query))
    scored = []
    for paragraph in paragraphs:
        para_tokens = set(tokenize_for_overlap(paragraph.get("text", "")))
        score = float(len(query_tokens & para_tokens) / max(1, len(query_tokens)))
        scored.append(
            {
                "para_id": paragraph.get("para_id", ""),
                "text": paragraph.get("text", ""),
                "score": score,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def candidate_test_files(path: str) -> List[Path]:
    path = Path(path)
    candidates = [path]
    if path.parent.exists():
        siblings = sorted(path.parent.glob("*.json"))
        for sibling in siblings:
            if sibling not in candidates:
                candidates.append(sibling)
    return candidates


def call_progress(i: int):
    try:
        subprocess.run([str(PROGRESS_LIB), str(i)], check=True)
    except Exception as exc:
        print(f"[WARN] progress signal failed at {i}: {exc}")


def load_data(path: str) -> dict:
    print(f"[1/4] Loading test data from {path} ...")
    last_error = None
    for candidate in candidate_test_files(path):
        try:
            print(f"      trying {candidate}")
            with open(candidate, "r", encoding="utf-8") as handle:
                return normalize_dataset(json.load(handle))
        except Exception as exc:
            last_error = exc
            print(f"      [warn] Skipping {candidate}: {exc}")
    raise RuntimeError(f"Unable to load a valid test dataset from {path}: {last_error}")


def build_doc_index(data: dict) -> Dict[str, List[Dict]]:
    return {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}


def load_embedder():
    print(f"[2/4] Loading embedding model from {config.EMBED_MODEL_PATH} ...")
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            str(config.EMBED_MODEL_PATH),
            device=device,
            local_files_only=True,
        )
        print(f"      Embedder loaded on {device}")
        return model
    except Exception as exc:
        print(f"      [warn] Failed to load embedder: {exc}")
        print("      [warn] Falling back to lexical retrieval")
        return None


def encode(model, texts: List[str]) -> np.ndarray:
    if model is None:
        raise RuntimeError("Embedder is unavailable")
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_tensor=False,
        show_progress_bar=False,
    )


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve(
    faiss_index,
    paragraphs: List[Dict],
    query_emb,
    query_text: str,
    top_k: int,
) -> List[Dict]:
    if faiss_index is None or query_emb is None:
        return fallback_retrieve(paragraphs, query_text, top_k)
    scores, indices = faiss_index.search(query_emb.reshape(1, -1).astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(paragraphs):
            results.append(
                {
                    "para_id": paragraphs[idx]["para_id"],
                    "text": paragraphs[idx]["text"],
                    "score": float(score),
                }
            )
    return results


def load_generator() -> Generator:
    print(f"[3/4] Loading LLM from {config.LLM_MODEL_PATH} ...")
    generator = Generator()
    try:
        generator.load_model()
    except Exception as exc:
        print(f"      [warn] Failed to load LLM: {exc}")
        print("      [warn] Falling back to extractive no-answer generation")
    return generator


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_runtime_context()

    if STARTUP_SLEEP_SECONDS > 0:
        print(f"[0/4] Sleeping for {STARTUP_SLEEP_SECONDS} seconds before startup ...")
        time.sleep(STARTUP_SLEEP_SECONDS)

    data = load_data(TEST_PATH)
    doc_index = build_doc_index(data)
    queries = data["queries"]
    total = len(queries)
    print(f"      {len(data['docs'])} docs | {total} queries")

    embedder = load_embedder()
    generator = load_generator()

    print("[3/4] Indexing documents ...")
    faiss_indices = {}
    for doc in tqdm(data["docs"], desc="Indexing"):
        paragraphs = doc["paragraphs"]
        if embedder is None:
            faiss_indices[doc["doc_id"]] = None
            continue
        try:
            embeddings = encode(embedder, [paragraph["text"] for paragraph in paragraphs])
            faiss_indices[doc["doc_id"]] = build_faiss_index(embeddings)
        except Exception as exc:
            print(f"      [warn] Failed to index {doc['doc_id']}: {exc}")
            faiss_indices[doc["doc_id"]] = None

    print(f"[4/4] Running inference on {total} queries ...")
    rows = []
    for i, query_row in enumerate(tqdm(queries, desc="Predicting"), start=1):
        query_id = query_row["ID"]
        doc_id = query_row["doc_id"]
        query = query_row["query"]
        paragraphs = doc_index.get(doc_id, [])

        query_emb = None
        if embedder is not None:
            try:
                query_emb = encode(embedder, [query])[0]
            except Exception as exc:
                print(f"      [warn] Failed to encode query {query_id}: {exc}")

        retrieved = retrieve(
            faiss_indices.get(doc_id),
            paragraphs,
            query_emb,
            query,
            config.RETRIEVAL_CANDIDATE_K,
        )
        reranked = hybrid_rerank(query, retrieved)
        refs = select_references_from_retrieved(reranked, n=config.REFERENCE_TOP_N)
        abstractive = generator.generate(
            query,
            reranked,
            max_seq_len=config.GENERATOR_MAX_SEQ_LEN,
        )

        rows.append(
            {
                "ID": query_id,
                "abstractive": abstractive,
                "refs": ",".join(refs),
            }
        )
        call_progress(i)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {RESULT_PATH}")
    call_progress(total)
    print("Done.")


if __name__ == "__main__":
    main()
