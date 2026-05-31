from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.prompting import detect_answer_profile
from src.reranker import load_reranker_if_available
from src.retrieval import rerank_retrieved, sentence_split
from .common import (
    DEFAULT_EMBED_MODEL_PATH,
    DEFAULT_RERANK_MODEL_PATH,
    LANTA_CACHE_ROOT,
    LANTA_PROJECT_ROOT,
    build_document_embedding_index,
    build_raw_samples,
    cache_dir_as_str,
    configure_cache_env,
    filter_queries_by_ids,
    grouped_doc_split,
    load_json,
    load_training_data,
    resolve_model_source,
    resolve_path,
    retrieve_paragraphs,
    run_evaluation,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure oracle ceilings for refs and extractive answers")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--rerank-model-name-or-path", default=str(DEFAULT_RERANK_MODEL_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--split-metadata-path")
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
    args.output_dir = resolve_path(args.output_dir, project_root=project_root)
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root)
    args.split_metadata_path = (
        resolve_path(args.split_metadata_path, project_root=project_root)
        if args.split_metadata_path
        else None
    )
    return args


def _best_sentence(query: str, answer: str, paragraphs: list[dict]) -> str:
    from rouge_score import rouge_scorer
    from rouge_score.tokenizers import Tokenizer
    from .common import tokenize_thai

    class ThaiSpaceTokenizer(Tokenizer):
        def tokenize(self, text: str) -> list[str]:
            return text.split(" ")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=ThaiSpaceTokenizer())
    gold = tokenize_thai(answer)
    candidates = []
    for paragraph in paragraphs:
        candidates.extend(sentence_split(paragraph.get("text", "")))
    if not candidates:
        return ""
    ranked = [
        (scorer.score(gold, tokenize_thai(candidate))["rougeL"].fmeasure, candidate)
        for candidate in candidates
        if candidate.strip()
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else ""


def _token_overlap(a: str, b: str) -> float:
    from src.retrieval import tokenize_for_overlap

    gold = set(tokenize_for_overlap(a))
    pred = set(tokenize_for_overlap(b))
    return len(gold & pred) / max(1, len(gold))


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    configure_cache_env(args.cache_dir, offline=True)

    from sentence_transformers import SentenceTransformer

    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    if args.split_metadata_path and args.split_metadata_path.exists():
        split_metadata = load_json(args.split_metadata_path)
        val_queries = filter_queries_by_ids(queries, split_metadata.get("val_query_ids", []))
    else:
        _, val_queries, _, _ = grouped_doc_split(queries, args.val_doc_ratio, args.seed)
    val_docs = [doc_lookup[doc_id] for doc_id in sorted({query["doc_id"] for query in val_queries})]
    val_samples, missing_refs = build_raw_samples(val_queries, doc_lookup)

    embedder = SentenceTransformer(
        resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root),
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    reranker = load_reranker_if_available(str(args.rerank_model_name_or_path), force=True)
    if reranker is not None:
        reranker.load_model()
    doc_embedding_index = build_document_embedding_index(val_docs, embedder)

    dense_hit20 = 0
    rerank_hit20 = 0
    best_sentence_rows = []
    extractability = []
    profiles = []
    for sample in val_samples:
        dense = retrieve_paragraphs(doc_embedding_index, sample["doc_id"], sample["query"], embedder, args.retrieval_top_k)
        profile = detect_answer_profile(sample["query"], dense)
        reranked = rerank_retrieved(sample["query"], dense, profile=profile, reranker=reranker)
        gold_set = set(sample["gold_refs"])
        dense_top20 = {item["para_id"] for item in dense[:20]}
        rerank_top20 = {item["para_id"] for item in reranked[:20]}
        dense_hit20 += 1 if gold_set & dense_top20 else 0
        rerank_hit20 += 1 if gold_set & rerank_top20 else 0
        doc = doc_lookup[sample["doc_id"]]
        gold_paragraphs = [p for p in doc["paragraphs"] if p["para_id"] in gold_set]
        best_sentence = _best_sentence(sample["query"], sample["answer"], gold_paragraphs)
        extractability.append(_token_overlap(sample["answer"], best_sentence))
        best_sentence_rows.append(
            {
                "ID": sample["ID"],
                "abstractive": best_sentence,
                "refs": ",".join(sample["gold_refs"]),
            }
        )
        profiles.append(profile)

    gold_df = pd.DataFrame(
        [
            {"ID": sample["ID"], "abstractive": sample["answer"], "refs": sample["gold_refs"]}
            for sample in val_samples
        ]
    )
    pred_df = pd.DataFrame(best_sentence_rows)
    sentence_metrics, _ = run_evaluation(gold_df, pred_df, embedder)
    total = max(1, len(val_samples))
    metrics = {
        "oracle_ref_iou": 1.0,
        "oracle_sentence_rouge": sentence_metrics["rougeL"],
        "oracle_sentence_ss": sentence_metrics["SS-score"],
        "oracle_sentence_score_with_gold_refs": sentence_metrics["score"],
        "dense_gold_coverage_at_20": dense_hit20 / total,
        "reranked_gold_coverage_at_20": rerank_hit20 / total,
        "answer_extractability_mean": float(sum(extractability) / max(1, len(extractability))),
        "missing_ref_rows": len(missing_refs),
        "val_samples": len(val_samples),
        "pred_profile_pct_fact": profiles.count("fact") / total,
        "pred_profile_pct_list": profiles.count("list") / total,
        "pred_profile_pct_synthesis": profiles.count("synthesis") / total,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_dir / "oracle_sentence_predictions.csv", index=False, encoding="utf-8")
    save_json(args.output_dir / "oracle_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
