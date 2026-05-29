from __future__ import annotations

import argparse
import itertools
import json

from sentence_transformers import SentenceTransformer

from src.prompting import detect_answer_profile
from src.reranker import load_reranker_if_available
from src.retrieval import ReferenceSelectionConfig, compute_selected_reference_metrics, rerank_retrieved, retrieval_candidate_count, select_references_from_retrieved

from finetune.common import (
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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid-search profile-specific ref selection thresholds")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--rerank-model-name-or-path")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-top-k", type=int, default=retrieval_candidate_count())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.project_root = resolve_path(args.project_root)
    args.train_json_path = resolve_path(args.train_json_path or (args.project_root / "data" / "train" / "train_set.json"), project_root=args.project_root)
    args.cache_dir = resolve_path(args.cache_dir, project_root=args.project_root)
    configure_cache_env(args.cache_dir, offline=True)
    import torch

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
    val_index = build_document_embedding_index([doc_lookup[doc_id] for doc_id in sorted(val_doc_ids)], embedder)

    cached = []
    for query_row in val_queries:
        dense = retrieve_paragraphs(val_index, query_row["doc_id"], query_row["query"], embedder, args.retrieval_top_k)
        reranked = rerank_retrieved(query_row["query"], dense, profile=detect_answer_profile(query_row["query"], dense), reranker=reranker)
        profile = detect_answer_profile(query_row["query"], reranked)
        cached.append((query_row["query"], list(query_row.get("refs", [])), reranked, profile))

    best = None
    for fact_top2, list_top2, list_top3, synth_top2, synth_top3, fact_gap, list_gap, synth_gap in itertools.product(
        [0.28, 0.32, 0.35, 0.38],
        [0.28, 0.32, 0.35],
        [0.20, 0.24, 0.28],
        [0.28, 0.32, 0.35],
        [0.20, 0.24, 0.28],
        [0.06, 0.08, 0.10],
        [0.10, 0.12, 0.15],
        [0.10, 0.12, 0.15],
    ):
        cfg = ReferenceSelectionConfig(
            fact_top2_min=fact_top2,
            list_top2_min=list_top2,
            list_top3_min=list_top3,
            synthesis_top2_min=synth_top2,
            synthesis_top3_min=synth_top3,
            fact_max_gap=fact_gap,
            list_max_gap=list_gap,
            synthesis_max_gap=synth_gap,
        )
        gold_refs = []
        pred_refs = []
        for query, gold, reranked, profile in cached:
            gold_refs.append(gold)
            pred_refs.append(select_references_from_retrieved(reranked, profile=profile, calibration_config=cfg))
        metrics = compute_selected_reference_metrics(gold_refs, pred_refs)
        candidate = {"metrics": metrics, "config": cfg.__dict__}
        if best is None or metrics["selected_ref_iou"] > best["metrics"]["selected_ref_iou"]:
            best = candidate

    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
