from __future__ import annotations

import argparse
import os
import sys

from src import config as runtime_config
from src.prompting import detect_answer_profile
from src.reranker import load_reranker_if_available
from src.retrieval import compute_retrieval_metrics, rerank_retrieved, select_references_from_retrieved

from .common import (
    DEFAULT_EMBED_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RERANK_MODEL_PATH,
    LANTA_CACHE_ROOT,
    LANTA_PROJECT_ROOT,
    build_document_embedding_index,
    cache_dir_as_str,
    configure_cache_env,
    ensure_local_model_exists,
    ensure_path_exists,
    filter_queries_by_ids,
    load_json,
    load_training_data,
    resolve_model_source,
    resolve_path,
    save_json,
    retrieve_paragraphs,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate reranker on grouped validation split")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_RERANK_MODEL_PATH))
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--retrieval-top-k", type=int, default=runtime_config.RERANK_TOP_K)
    parser.add_argument("--reference-top-n", type=int, default=runtime_config.REFERENCE_TOP_N)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    project_root = resolve_path(args.project_root)
    assert project_root is not None
    args.project_root = project_root
    args.train_json_path = resolve_path(
        args.train_json_path or (project_root / "data" / "train" / "train_set.json"),
        project_root=project_root,
    )
    args.output_dir = resolve_path(args.output_dir, project_root=project_root)
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root)
    return args


def validate_args(args: argparse.Namespace) -> None:
    ensure_path_exists(args.train_json_path, "Train JSON")
    ensure_local_model_exists(args.model_name_or_path, "Reranker model", project_root=args.project_root)
    ensure_local_model_exists(args.embed_model_name_or_path, "Embed model", project_root=args.project_root)
    ensure_path_exists(args.output_dir / "split_metadata.json", "Split metadata")


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    validate_args(args)
    configure_cache_env(args.cache_dir, offline=True)

    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    split_metadata = load_json(args.output_dir / "split_metadata.json")
    val_query_ids = split_metadata.get("val_query_ids") or []
    val_queries = filter_queries_by_ids(queries, val_query_ids)
    val_doc_ids = sorted({query["doc_id"] for query in val_queries})
    val_docs = [doc_lookup[doc_id] for doc_id in val_doc_ids]

    embed_source = resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root)
    embedder = SentenceTransformer(
        embed_source,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    reranker = load_reranker_if_available(args.model_name_or_path)
    if reranker is None:
        raise RuntimeError(f"Unable to load reranker from {args.model_name_or_path}")
    reranker.load_model()
    doc_embedding_index = build_document_embedding_index(val_docs, embedder)

    prediction_rows = []
    diagnostics = []
    gold_refs_list = []
    reranked_retrievals = []
    for query in val_queries:
        dense_retrieved = retrieve_paragraphs(
            doc_embedding_index,
            query["doc_id"],
            query["query"],
            embedder,
            args.retrieval_top_k,
        )
        reranked = rerank_retrieved(
            query["query"],
            dense_retrieved,
            reranker=reranker,
            rerank_top_k=args.retrieval_top_k,
        )
        profile = detect_answer_profile(query["query"], reranked)
        predicted_refs = select_references_from_retrieved(reranked, profile=profile, n=args.reference_top_n)
        prediction_rows.append({"ID": query["ID"], "refs": ",".join(predicted_refs)})
        diagnostics.append(
            {
                "ID": query["ID"],
                "profile": profile,
                "gold_refs": query.get("refs", []),
                "pred_refs": predicted_refs,
                "top_candidates": [
                    {
                        "para_id": item["para_id"],
                        "score": float(item.get("score", 0.0)),
                        "dense_score": float(item.get("dense_score", 0.0)),
                        "rerank_score": float(item.get("rerank_score", item.get("score", 0.0))),
                    }
                    for item in reranked[:5]
                ],
            }
        )
        gold_refs_list.append(query.get("refs", []))
        reranked_retrievals.append(reranked)

    pred_df = pd.DataFrame(prediction_rows)
    metrics = compute_retrieval_metrics(gold_refs_list, reranked_retrievals)
    metrics["pred_ref_count_mean"] = float(pred_df["refs"].apply(lambda value: len([x for x in value.split(",") if x])).mean())
    save_json(args.output_dir / "reranker_validation_metrics.json", metrics)
    save_json(args.output_dir / "reranker_validation_diagnostics.json", {"rows": diagnostics})
    print(metrics)


if __name__ == "__main__":
    main()
