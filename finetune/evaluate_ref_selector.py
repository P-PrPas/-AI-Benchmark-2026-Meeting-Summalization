from __future__ import annotations

import argparse
import json
import os
import sys

from src.prompting import detect_answer_profile
from src.ref_selector import FEATURE_ORDER, LearnedRefSelector, extract_selector_features
from src.retrieval import compute_selected_reference_metrics, rerank_retrieved, retrieval_candidate_count, select_references_from_retrieved
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
    parser = argparse.ArgumentParser(description="Evaluate learned ref selector on grouped validation split")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--rerank-model-name-or-path")
    parser.add_argument("--selector-model-path")
    parser.add_argument("--output-path")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-top-k", type=int, default=retrieval_candidate_count())
    return parser


def _predict_refs(query: str, reranked: list[dict], profile: str, selector: LearnedRefSelector) -> list[str]:
    if not reranked:
        return []
    features = extract_selector_features(query, reranked, profile)
    vector = [[features[name] for name in FEATURE_ORDER]]
    selected = [reranked[0]["para_id"]]
    if len(reranked) >= 2 and float(selector.keep2_model.predict_proba(vector)[0][1]) >= 0.5:
        selected.append(reranked[1]["para_id"])
    if len(reranked) >= 3 and profile in {"list", "synthesis"} and float(selector.keep3_model.predict_proba(vector)[0][1]) >= 0.5:
        selected.append(reranked[2]["para_id"])
    return selected


def print_runtime_config(args: argparse.Namespace) -> None:
    print("Runtime configuration")
    print(f"  project_root={args.project_root}")
    print(f"  train_json_path={args.train_json_path}")
    print(f"  embed_model_name_or_path={resolve_model_source(args.embed_model_name_or_path, args.project_root)}")
    print(f"  rerank_model_name_or_path={args.rerank_model_name_or_path}")
    print(f"  selector_model_path={args.selector_model_path}")
    print(f"  output_path={args.output_path}")
    print(f"  cache_dir={args.cache_dir}")
    print(f"  val_doc_ratio={args.val_doc_ratio}")
    print(f"  seed={args.seed}")
    print(f"  retrieval_top_k={args.retrieval_top_k}")
    print(f"  python={sys.executable}")
    print(f"  cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.project_root = resolve_path(args.project_root)
    args.train_json_path = resolve_path(args.train_json_path or (args.project_root / "data" / "train" / "train_set.json"), project_root=args.project_root)
    args.selector_model_path = resolve_path(args.selector_model_path, project_root=args.project_root)
    args.output_path = resolve_path(args.output_path or args.selector_model_path.with_suffix(".eval.json"), project_root=args.project_root)
    args.cache_dir = resolve_path(args.cache_dir, project_root=args.project_root)
    configure_cache_env(args.cache_dir, offline=True)

    import torch
    from sentence_transformers import SentenceTransformer

    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    _, val_queries, _, val_doc_ids = grouped_doc_split(queries, args.val_doc_ratio, args.seed)
    embedder = SentenceTransformer(
        resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root),
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    reranker = load_reranker_if_available(args.rerank_model_name_or_path)
    if reranker is not None:
        reranker.load_model()
    selector = LearnedRefSelector(str(args.selector_model_path))
    selector.load_model()
    print_runtime_config(args)
    print("[1/4] Loading data and building indices ...")
    val_index = build_document_embedding_index([doc_lookup[doc_id] for doc_id in sorted(val_doc_ids)], embedder)

    gold_refs = []
    rule_refs = []
    selector_refs = []
    print("[2/4] Running selector on validation split ...")
    for index, query_row in enumerate(val_queries, start=1):
        dense = retrieve_paragraphs(val_index, query_row["doc_id"], query_row["query"], embedder, args.retrieval_top_k)
        reranked = rerank_retrieved(query_row["query"], dense, profile=detect_answer_profile(query_row["query"], dense), reranker=reranker)
        profile = detect_answer_profile(query_row["query"], reranked)
        gold_refs.append(list(query_row.get("refs", [])))
        rule_refs.append(select_references_from_retrieved(reranked, profile=profile))
        selector_refs.append(_predict_refs(query_row["query"], reranked, profile, selector))
        if index % 100 == 0:
            print(f"      processed val queries={index}/{len(val_queries)}")

    print("[3/4] Computing metrics ...")
    metrics = {
        "rule": compute_selected_reference_metrics(gold_refs, rule_refs),
        "selector": compute_selected_reference_metrics(gold_refs, selector_refs),
    }
    save_json(args.output_path, metrics)
    print(f"Saved selector metrics -> {args.output_path}")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
