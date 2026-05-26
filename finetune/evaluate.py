from __future__ import annotations

import argparse
import os
import sys

from src import config as runtime_config
from src.generator import Generator
from src.prompting import detect_answer_profile
from src.reranker import load_reranker_if_available
from src.retrieval import (
    build_generation_context,
    compute_retrieval_metrics,
    needs_query_refinement,
    rerank_retrieved,
    rewrite_query_heuristic,
    select_references_from_retrieved,
)
from .common import (
    DEFAULT_ARTIFACT_NAME,
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_EMBED_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    LANTA_CACHE_ROOT,
    LANTA_PROJECT_ROOT,
    SYSTEM_PROMPT,
    build_document_embedding_index,
    build_raw_samples,
    cache_dir_as_str,
    configure_cache_env,
    ensure_local_model_exists,
    ensure_path_exists,
    filter_queries_by_ids,
    load_json,
    load_training_data,
    resolve_model_source,
    resolve_path,
    run_evaluation,
    save_json,
    retrieve_paragraphs,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate finetuned typhoon2.5-qwen3-4b on held-out validation")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--rerank-model-name-or-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--max-seq-len", type=int, default=runtime_config.GENERATOR_MAX_SEQ_LEN)
    parser.add_argument("--retrieval-top-k", type=int, default=runtime_config.RETRIEVAL_CANDIDATE_K)
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
    args.output_dir = resolve_path(
        args.output_dir or (project_root / "artifacts" / DEFAULT_ARTIFACT_NAME),
        project_root=project_root,
    )
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root)
    args.model_name_or_path = args.model_name_or_path or str(args.output_dir / "final_merged")
    args.rerank_model_name_or_path = args.rerank_model_name_or_path or os.environ.get("CAMNET_RERANK_MODEL_PATH")
    args.adapter_path = (
        resolve_path(args.adapter_path, project_root=project_root)
        if args.adapter_path
        else None
    )
    return args


def validate_args(args: argparse.Namespace) -> None:
    ensure_path_exists(args.train_json_path, "Train JSON")
    ensure_path_exists(args.output_dir / "split_metadata.json", "Split metadata")
    ensure_local_model_exists(args.model_name_or_path, "Model", project_root=args.project_root)
    if args.adapter_path is not None:
        ensure_path_exists(args.adapter_path, "Adapter directory")
    ensure_local_model_exists(args.embed_model_name_or_path, "Embed model", project_root=args.project_root)


def print_runtime_config(args: argparse.Namespace) -> None:
    print("Runtime configuration")
    print(f"  project_root={args.project_root}")
    print(f"  train_json_path={args.train_json_path}")
    print(f"  model_name_or_path={resolve_model_source(args.model_name_or_path, args.project_root)}")
    print(f"  adapter_path={args.adapter_path}")
    print(f"  embed_model_name_or_path={resolve_model_source(args.embed_model_name_or_path, args.project_root)}")
    print(f"  rerank_model_name_or_path={args.rerank_model_name_or_path}")
    print(f"  output_dir={args.output_dir}")
    print(f"  cache_dir={args.cache_dir}")


def load_validation_queries(args: argparse.Namespace):
    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    split_metadata = load_json(args.output_dir / "split_metadata.json")
    val_query_ids = split_metadata.get("val_query_ids") or []
    val_queries = filter_queries_by_ids(queries, val_query_ids)
    if not val_queries:
        raise ValueError("No validation queries found from split metadata.")
    val_doc_ids = sorted({query["doc_id"] for query in val_queries})
    val_docs = [doc_lookup[doc_id] for doc_id in val_doc_ids]
    val_raw_samples, missing_val_refs = build_raw_samples(val_queries, doc_lookup)
    return docs, val_docs, val_queries, val_raw_samples, missing_val_refs


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    validate_args(args)
    configure_cache_env(args.cache_dir, offline=True)
    print_runtime_config(args)

    import pandas as pd
    import torch
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA-enabled server runtime.")

    _, val_docs, _, val_raw_samples, missing_val_refs = load_validation_queries(args)
    save_json(args.output_dir / "missing_eval_refs.json", {"rows": missing_val_refs})

    model_source = resolve_model_source(args.model_name_or_path, project_root=args.project_root)
    tokenizer_source = (
        args.adapter_path
        if args.adapter_path is not None and (args.adapter_path / "tokenizer_config.json").exists()
        else model_source
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=True,
        cache_dir=cache_dir_as_str(args.cache_dir),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    if args.adapter_path is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir_as_str(args.cache_dir),
        )
        model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir_as_str(args.cache_dir),
        )
    model.eval()
    model.config.use_cache = True

    embed_source = resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root)
    embedder = SentenceTransformer(
        embed_source,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    reranker = load_reranker_if_available(args.rerank_model_name_or_path)
    if reranker is not None:
        reranker.load_model()

    doc_embedding_index = build_document_embedding_index(val_docs, embedder)

    generator = Generator(system_prompt=SYSTEM_PROMPT)
    generator.model = model
    generator.tokenizer = tokenizer

    prediction_rows = []
    gold_rows = []
    reranked_retrievals = []
    predicted_profiles = {}
    retrieval_diagnostics = []
    for sample in val_raw_samples:
        dense_retrieved = retrieve_paragraphs(
            doc_embedding_index,
            sample["doc_id"],
            sample["query"],
            embedder,
            args.retrieval_top_k,
        )
        predicted_profile = detect_answer_profile(sample["query"], dense_retrieved)
        retrieved = rerank_retrieved(
            sample["query"],
            dense_retrieved,
            reranker=reranker,
            rerank_top_k=runtime_config.RERANK_TOP_K,
        )
        if needs_query_refinement(retrieved, predicted_profile):
            refined_query = rewrite_query_heuristic(sample["query"])
            if refined_query != sample["query"]:
                refined_dense = retrieve_paragraphs(
                    doc_embedding_index,
                    sample["doc_id"],
                    refined_query,
                    embedder,
                    args.retrieval_top_k,
                )
                refined_retrieved = rerank_retrieved(
                    refined_query,
                    refined_dense,
                    reranker=reranker,
                    rerank_top_k=runtime_config.RERANK_TOP_K,
                )
                if refined_retrieved:
                    retrieved = refined_retrieved
        reranked_retrievals.append(retrieved)
        predicted_profile = generator.detect_profile(sample["query"], retrieved)
        predicted_profiles[sample["ID"]] = predicted_profile
        predicted_refs = select_references_from_retrieved(
            retrieved,
            profile=predicted_profile,
            n=args.reference_top_n,
        )
        generation_paragraphs = build_generation_context(
            sample["query"],
            retrieved,
            predicted_refs,
            predicted_profile,
        )
        predicted_answer = generator.generate(
            sample["query"],
            generation_paragraphs,
            profile=predicted_profile,
            max_seq_len=args.max_seq_len,
        )
        prediction_rows.append(
            {
                "ID": sample["ID"],
                "abstractive": predicted_answer,
                "refs": ",".join(predicted_refs),
                "profile": predicted_profile,
            }
        )
        retrieval_diagnostics.append(
            {
                "ID": sample["ID"],
                "profile": predicted_profile,
                "predicted_refs": predicted_refs,
                "top_candidates": [
                    {
                        "para_id": item["para_id"],
                        "score": float(item.get("score", 0.0)),
                        "dense_score": float(item.get("dense_score", 0.0)),
                        "lexical_score": float(item.get("lexical_score", 0.0)),
                        "rerank_score": float(item.get("rerank_score", item.get("score", 0.0))),
                    }
                    for item in retrieved[:5]
                ],
            }
        )
        gold_rows.append(
            {
                "ID": sample["ID"],
                "abstractive": sample["answer"],
                "refs": sample["gold_refs"],
            }
        )

    pred_df = pd.DataFrame(prediction_rows)
    gold_df = pd.DataFrame(gold_rows)
    pred_df.to_csv(args.output_dir / "val_predictions.csv", index=False, encoding="utf-8")

    metrics, merged = run_evaluation(gold_df, pred_df, embedder)
    metrics.update(
        compute_retrieval_metrics(
            [sample["gold_refs"] for sample in val_raw_samples],
            reranked_retrievals,
        )
    )
    answer_lengths = [len(row["abstractive"]) for row in prediction_rows]
    malformed = [
        row for row in prediction_rows
        if "[P" in row["abstractive"]
        or row["abstractive"].strip().startswith(("คำตอบ:", "ตอบ:"))
        or row["abstractive"].strip().endswith(("...", "…"))
    ]
    metrics["pred_answer_length_median"] = float(pd.Series(answer_lengths).median()) if answer_lengths else 0.0
    metrics["pred_answer_length_mean"] = float(pd.Series(answer_lengths).mean()) if answer_lengths else 0.0
    metrics["format_error_rate"] = len(malformed) / max(1, len(prediction_rows))
    save_json(args.output_dir / "validation_metrics.json", metrics)
    save_json(args.output_dir / "retrieval_diagnostics.json", {"rows": retrieval_diagnostics})

    failure_rows = []
    pred_lookup = {row["ID"]: row for row in prediction_rows}
    merged_lookup = merged.set_index("ID").to_dict("index")
    failure_counts = {}
    for sample, retrieved in zip(val_raw_samples, reranked_retrievals):
        pred = pred_lookup[sample["ID"]]
        merged_row = merged_lookup[sample["ID"]]
        predicted_refs = set(pred["refs"].split(",")) if pred["refs"] else set()
        gold_refs = set(sample["gold_refs"])
        answer = pred["abstractive"]
        detected_profile = predicted_profiles[sample["ID"]]
        rouge_score = float(merged_row["rougeL"])
        ss_score = float(merged_row["SS-score"])
        if not gold_refs.intersection({item["para_id"] for item in retrieved[:10]}):
            failure_type = "retrieval_miss"
        elif ss_score > 0.8 and rouge_score < 0.4:
            failure_type = "high_ss_low_rouge"
        elif sample["profile"] == "fact" and detected_profile == "synthesis":
            failure_type = "fact_routed_to_synthesis"
        elif "[P" in answer or answer.strip().startswith(("คำตอบ:", "ตอบ:")):
            failure_type = "formatting"
        elif len(answer) > runtime_config.FACT_MAX_ANSWER_CHARS and "\n" not in answer:
            failure_type = "overlong_answer"
        elif predicted_refs == gold_refs and gold_refs:
            failure_type = "reference_correct_answer_style_mismatch"
        elif predicted_refs != gold_refs and gold_refs:
            failure_type = "reference_mismatch"
        else:
            failure_type = "answer_style_or_semantics"
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        failure_rows.append(
            {
                "ID": sample["ID"],
                "failure_type": failure_type,
                "query": sample["query"],
                "gold_answer": sample["answer"],
                "pred_answer": answer,
                "gold_refs": sample["gold_refs"],
                "pred_refs": sorted(predicted_refs),
                "top_retrieved": [item["para_id"] for item in retrieved[:5]],
                "gold_profile": sample["profile"],
                "pred_profile": detected_profile,
                "rougeL": rouge_score,
                "SS-score": ss_score,
            }
        )
    failure_rows.sort(key=lambda row: (row["failure_type"], row["rougeL"]))
    save_json(
        args.output_dir / "failure_analysis.json",
        {
            "summary": failure_counts,
            "rows": failure_rows[:50],
        },
    )

    print(f"Validation samples={len(val_raw_samples)}")
    print(f"Saved predictions to {args.output_dir / 'val_predictions.csv'}")
    print(f"Saved metrics to {args.output_dir / 'validation_metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
