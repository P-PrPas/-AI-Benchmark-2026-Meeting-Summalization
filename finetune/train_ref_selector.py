from __future__ import annotations

import argparse
import json
import pickle
import os
import sys
from pathlib import Path

from src.prompting import detect_answer_profile
from src.retrieval import compute_selected_reference_metrics, rerank_retrieved, retrieval_candidate_count
from src.ref_selector import FEATURE_ORDER, extract_selector_features
from src.reranker import load_reranker_if_available

from .common import (
    DEFAULT_EMBED_MODEL_PATH,
    LANTA_CACHE_ROOT,
    LANTA_PROJECT_ROOT,
    build_document_embedding_index,
    cache_dir_as_str,
    configure_cache_env,
    grouped_doc_split,
    load_training_data,
    resolve_model_source,
    resolve_path,
    retrieve_paragraphs,
    save_json,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train lightweight learned ref selector")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--rerank-model-name-or-path")
    parser.add_argument("--output-path")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-top-k", type=int, default=retrieval_candidate_count())
    return parser


def _labels_for_query(gold_refs: list[str], reranked: list[dict]) -> tuple[int, int]:
    gold_set = set(gold_refs or [])
    keep2 = int(len(reranked) >= 2 and reranked[1]["para_id"] in gold_set)
    keep3 = int(len(reranked) >= 3 and reranked[2]["para_id"] in gold_set)
    return keep2, keep3


def _predict_refs(query: str, reranked: list[dict], profile: str, keep2_model, keep3_model) -> list[str]:
    if not reranked:
        return []
    features = extract_selector_features(query, reranked, profile)
    vector = [[features[name] for name in FEATURE_ORDER]]
    selected = [reranked[0]["para_id"]]
    if len(reranked) >= 2 and float(keep2_model.predict_proba(vector)[0][1]) >= 0.5:
        selected.append(reranked[1]["para_id"])
    if len(reranked) >= 3 and profile in {"list", "synthesis"} and float(keep3_model.predict_proba(vector)[0][1]) >= 0.5:
        selected.append(reranked[2]["para_id"])
    return selected


def print_runtime_config(args: argparse.Namespace) -> None:
    print("Runtime configuration")
    print(f"  project_root={args.project_root}")
    print(f"  train_json_path={args.train_json_path}")
    print(f"  embed_model_name_or_path={resolve_model_source(args.embed_model_name_or_path, args.project_root)}")
    print(f"  rerank_model_name_or_path={args.rerank_model_name_or_path}")
    print(f"  output_path={args.output_path}")
    print(f"  cache_dir={args.cache_dir}")
    print(f"  val_doc_ratio={args.val_doc_ratio}")
    print(f"  seed={args.seed}")
    print(f"  retrieval_top_k={args.retrieval_top_k}")
    print(f"  python={sys.executable}")
    print(f"  cuda_available={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.project_root = resolve_path(args.project_root)
    args.train_json_path = resolve_path(args.train_json_path or (args.project_root / "data" / "train" / "train_set.json"), project_root=args.project_root)
    args.output_path = resolve_path(args.output_path or (args.project_root / "artifacts" / "ref_selector.pkl"), project_root=args.project_root)
    args.cache_dir = resolve_path(args.cache_dir, project_root=args.project_root)
    configure_cache_env(args.cache_dir, offline=True)

    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    print_runtime_config(args)
    print("[1/6] Loading training data ...")
    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    train_queries, val_queries, train_doc_ids, val_doc_ids = grouped_doc_split(queries, args.val_doc_ratio, args.seed)
    print(f"      docs={len(docs)} queries={len(queries)} train_queries={len(train_queries)} val_queries={len(val_queries)}")
    print("[2/6] Loading embedder ...")
    embedder = SentenceTransformer(
        resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root),
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    print("[3/6] Loading reranker ...")
    reranker = load_reranker_if_available(args.rerank_model_name_or_path)
    if reranker is not None:
        reranker.load_model()
        print("      reranker loaded")
    else:
        print("      reranker unavailable")

    print("[4/6] Building document indices ...")
    train_index = build_document_embedding_index([doc_lookup[doc_id] for doc_id in sorted(train_doc_ids)], embedder)
    val_index = build_document_embedding_index([doc_lookup[doc_id] for doc_id in sorted(val_doc_ids)], embedder)

    print("[5/6] Building selector dataset ...")
    train_x = []
    train_y2 = []
    train_y3 = []
    for index, query_row in enumerate(train_queries, start=1):
        dense = retrieve_paragraphs(train_index, query_row["doc_id"], query_row["query"], embedder, args.retrieval_top_k)
        reranked = rerank_retrieved(query_row["query"], dense, profile=detect_answer_profile(query_row["query"], dense), reranker=reranker)
        if not reranked:
            continue
        profile = detect_answer_profile(query_row["query"], reranked)
        features = extract_selector_features(query_row["query"], reranked, profile)
        keep2, keep3 = _labels_for_query(list(query_row.get("refs", [])), reranked)
        train_x.append([features[name] for name in FEATURE_ORDER])
        train_y2.append(keep2)
        train_y3.append(keep3 if profile in {"list", "synthesis"} else 0)
        if index % 200 == 0:
            print(f"      processed train queries={index}/{len(train_queries)}")

    print(f"      usable_train_rows={len(train_x)}")
    print("[6/6] Fitting selector models ...")
    keep2_model = LogisticRegression(max_iter=200, class_weight="balanced")
    keep3_model = LogisticRegression(max_iter=200, class_weight="balanced")
    keep2_model.fit(train_x, train_y2)
    keep3_model.fit(train_x, train_y3)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_bytes(pickle.dumps({"keep2_model": keep2_model, "keep3_model": keep3_model}))

    val_gold = []
    val_pred = []
    print("[7/7] Evaluating selector on held-out split ...")
    for index, query_row in enumerate(val_queries, start=1):
        dense = retrieve_paragraphs(val_index, query_row["doc_id"], query_row["query"], embedder, args.retrieval_top_k)
        reranked = rerank_retrieved(query_row["query"], dense, profile=detect_answer_profile(query_row["query"], dense), reranker=reranker)
        profile = detect_answer_profile(query_row["query"], reranked)
        val_gold.append(list(query_row.get("refs", [])))
        val_pred.append(_predict_refs(query_row["query"], reranked, profile, keep2_model, keep3_model))
        if index % 100 == 0:
            print(f"      processed val queries={index}/{len(val_queries)}")

    metrics = compute_selected_reference_metrics(val_gold, val_pred)
    save_json(
        args.output_path.with_suffix(".metrics.json"),
        {
            "train_size": len(train_x),
            "val_size": len(val_queries),
            **metrics,
        },
    )
    print(f"Saved selector model -> {args.output_path}")
    print(f"Saved selector metrics -> {args.output_path.with_suffix('.metrics.json')}")
    print(json.dumps({"model_path": str(args.output_path), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
