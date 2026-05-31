from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from src import config as runtime_config
from src.evidence_set import EvidenceSetSelector, FEATURE_ORDER, extract_evidence_features
from src.prompting import detect_answer_profile
from src.reranker import load_reranker_if_available
from src.retrieval import (
    compute_selected_reference_metrics,
    compute_selected_reference_metrics_by_profile,
    rerank_retrieved,
)
from .common import (
    DEFAULT_EMBED_MODEL_PATH,
    DEFAULT_RERANK_MODEL_PATH,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train supervised evidence-set selector on reranked candidates")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--rerank-model-name-or-path", default=str(DEFAULT_RERANK_MODEL_PATH))
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    project_root = resolve_path(args.project_root)
    assert project_root is not None
    args.project_root = project_root
    args.train_json_path = resolve_path(
        args.train_json_path or (project_root / "data" / "train" / "train_set.json"),
        project_root=project_root,
    )
    args.output_path = resolve_path(args.output_path, project_root=project_root)
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root)
    return args


def _gold_refs(query: dict, doc_lookup: dict[str, dict]) -> list[str]:
    refs = query.get("relevant_paragraphs") or query.get("refs") or query.get("gold_refs") or []
    if refs:
        return list(refs)
    doc = doc_lookup[query["doc_id"]]
    ref_texts = set(query.get("relevant_sentences", []) or [])
    return [
        paragraph["para_id"]
        for paragraph in doc.get("paragraphs", [])
        if paragraph.get("text") in ref_texts
    ]


def _query_text(query: dict) -> str:
    return str(query.get("question") or query.get("query") or query.get("problem") or "")


def _answer_text(query: dict) -> str:
    return str(query.get("answer") or query.get("abstractive") or "")


def _build_rows(queries, doc_lookup, doc_embedding_index, embedder, reranker, retrieval_top_k):
    features = []
    labels = []
    cardinality_features = []
    cardinality_labels = []
    eval_payload = []
    for index, query in enumerate(queries, start=1):
        if index % 100 == 0:
            print(f"  built {index}/{len(queries)} queries")
        query_text = _query_text(query)
        gold_refs = _gold_refs(query, doc_lookup)
        dense = retrieve_paragraphs(doc_embedding_index, query["doc_id"], query_text, embedder, retrieval_top_k)
        profile = detect_answer_profile(query_text, dense)
        reranked = rerank_retrieved(
            query_text,
            dense,
            profile=profile,
            reranker=reranker,
            rerank_top_k=runtime_config.RERANK_TOP_K,
        )
        top = reranked[:20]
        gold_set = set(gold_refs)
        for candidate_index in range(len(top)):
            row = extract_evidence_features(query_text, top, candidate_index, profile)
            features.append([row[name] for name in FEATURE_ORDER])
            labels.append(1 if top[candidate_index]["para_id"] in gold_set else 0)
        if top:
            cardinality_features.append(features[-len(top)])
            cardinality_labels.append(min(3, max(1, len(gold_set))))
        eval_payload.append((query_text, gold_refs, reranked, profile, _answer_text(query)))
    return features, labels, cardinality_features, cardinality_labels, eval_payload


def _evaluate_selector(selector: EvidenceSetSelector, eval_payload):
    gold_refs_list = []
    pred_refs_list = []
    profiles = []
    for query_text, gold_refs, retrieved, profile, _ in eval_payload:
        prediction = selector.predict(query_text, retrieved, profile)
        gold_refs_list.append(gold_refs)
        pred_refs_list.append(prediction.refs)
        profiles.append(profile)
    metrics = compute_selected_reference_metrics(gold_refs_list, pred_refs_list)
    metrics.update(compute_selected_reference_metrics_by_profile(gold_refs_list, pred_refs_list, profiles))
    return metrics


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    configure_cache_env(args.cache_dir, offline=True)

    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import GradientBoostingClassifier

    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    train_queries, val_queries, train_doc_ids, val_doc_ids = grouped_doc_split(
        queries,
        args.val_doc_ratio,
        args.seed,
    )
    print(f"Train queries={len(train_queries)} val queries={len(val_queries)}")

    embedder = SentenceTransformer(
        resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root),
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    reranker = load_reranker_if_available(str(args.rerank_model_name_or_path))
    if reranker is not None:
        reranker.load_model()

    train_docs = [doc_lookup[doc_id] for doc_id in sorted(train_doc_ids)]
    val_docs = [doc_lookup[doc_id] for doc_id in sorted(val_doc_ids)]
    print("Building train embeddings")
    train_index = build_document_embedding_index(train_docs, embedder)
    print("Building val embeddings")
    val_index = build_document_embedding_index(val_docs, embedder)

    print("Building train rows")
    train_x, train_y, train_card_x, train_card_y, _ = _build_rows(
        train_queries,
        doc_lookup,
        train_index,
        embedder,
        reranker,
        args.retrieval_top_k,
    )
    print("Building val rows")
    _, _, _, _, val_payload = _build_rows(
        val_queries,
        doc_lookup,
        val_index,
        embedder,
        reranker,
        args.retrieval_top_k,
    )
    print(f"Membership rows={len(train_x)} positives={int(np.sum(train_y))}")

    membership_model = GradientBoostingClassifier(random_state=args.seed)
    membership_model.fit(train_x, train_y)
    cardinality_model = GradientBoostingClassifier(random_state=args.seed)
    cardinality_model.fit(train_card_x, train_card_y)

    payload = {
        "membership_model": membership_model,
        "cardinality_model": cardinality_model,
        "feature_order": FEATURE_ORDER,
        "train_size": len(train_queries),
        "val_size": len(val_queries),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_bytes(pickle.dumps(payload))

    selector = EvidenceSetSelector(str(args.output_path))
    selector.load_model()
    metrics = _evaluate_selector(selector, val_payload)
    save_json(args.output_path.with_suffix(".metrics.json"), metrics)
    print(f"Saved evidence-set selector to {args.output_path}")
    print(metrics)


if __name__ == "__main__":
    main()
