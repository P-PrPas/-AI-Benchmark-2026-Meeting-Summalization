from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

from src import config as runtime_config
from src.answer_candidates import select_heuristic_candidate
from src.answer_ranker import load_answer_ranker_if_available
from src.generator import Generator
from src.prompting import detect_answer_profile
from src.ref_selector import load_ref_selector_if_available
from src.evidence_set import load_evidence_set_selector_if_available
from src.reranker import load_reranker_if_available
from src.retrieval import (
    build_generation_context,
    compute_arbiter_metrics,
    compute_evidence_set_metrics,
    compute_selected_reference_metrics,
    compute_selected_reference_metrics_by_profile,
    compute_selector_metrics,
    compute_retrieval_metrics,
    needs_query_refinement,
    rerank_retrieved,
    rewrite_query_heuristic,
    select_references_with_diagnostics,
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
    build_split_metadata,
    build_raw_samples,
    cache_dir_as_str,
    configure_cache_env,
    ensure_local_model_exists,
    ensure_path_exists,
    grouped_doc_split,
    filter_queries_by_ids,
    load_json,
    load_training_data,
    resolve_model_source,
    resolve_path,
    run_evaluation,
    save_json,
    tokenize_thai,
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
    parser.add_argument("--split-metadata-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--max-seq-len", type=int, default=runtime_config.GENERATOR_MAX_SEQ_LEN)
    parser.add_argument("--retrieval-top-k", type=int, default=runtime_config.RETRIEVAL_CANDIDATE_K)
    parser.add_argument("--reference-top-n", type=int, default=runtime_config.REFERENCE_TOP_N)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
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
    args.split_metadata_path = (
        resolve_path(args.split_metadata_path, project_root=project_root)
        if args.split_metadata_path
        else None
    )
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
    if args.split_metadata_path is not None:
        ensure_path_exists(args.split_metadata_path, "Split metadata")
    elif (args.output_dir / "split_metadata.json").exists():
        pass
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
    print(f"  use_reranker={runtime_config.USE_RERANKER}")
    print(f"  expanded_candidates={runtime_config.ENABLE_EXPANDED_CANDIDATES}")
    print(f"  evidence_set_selector={runtime_config.ENABLE_EVIDENCE_SET_SELECTOR}")
    print(f"  semi_extractive_composer={runtime_config.ENABLE_SEMI_EXTRACTIVE_COMPOSER}")
    print(f"  answer_candidates={runtime_config.ENABLE_ANSWER_CANDIDATES}")
    print(f"  answer_candidate_selection_mode={runtime_config.ANSWER_CANDIDATE_SELECTION_MODE}")
    print(f"  answer_ranker_model_path={runtime_config.ANSWER_RANKER_MODEL_PATH}")
    print(f"  output_dir={args.output_dir}")
    print(f"  cache_dir={args.cache_dir}")


def load_validation_queries(args: argparse.Namespace):
    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    split_metadata_path = args.split_metadata_path or (args.output_dir / "split_metadata.json")
    if split_metadata_path.exists():
        split_metadata = load_json(split_metadata_path)
    else:
        train_queries, val_queries, train_doc_ids, val_doc_ids = grouped_doc_split(
            queries,
            args.val_doc_ratio,
            args.seed,
        )
        split_metadata = build_split_metadata(
            seed=args.seed,
            val_ratio=args.val_doc_ratio,
            train_doc_ids=train_doc_ids,
            val_doc_ids=val_doc_ids,
            train_queries=train_queries,
            val_queries=val_queries,
            dropped_train=[],
            dropped_val=[],
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_json(split_metadata_path, split_metadata)
        except OSError:
            pass
    val_query_ids = split_metadata.get("val_query_ids") or []
    val_queries = filter_queries_by_ids(queries, val_query_ids)
    if not val_queries:
        raise ValueError("No validation queries found from split metadata.")
    val_doc_ids = sorted({query["doc_id"] for query in val_queries})
    val_docs = [doc_lookup[doc_id] for doc_id in val_doc_ids]
    val_raw_samples, missing_val_refs = build_raw_samples(val_queries, doc_lookup)
    return docs, val_docs, val_queries, val_raw_samples, missing_val_refs


def generate_prepared_rows_in_batches(
    generator: Generator,
    prepared_rows: list[dict],
    *,
    max_seq_len: int,
    answer_ranker=None,
) -> list[dict]:
    oracle_scorer = None
    if runtime_config.ENABLE_ANSWER_CANDIDATES and runtime_config.ANSWER_CANDIDATE_SELECTION_MODE == "oracle":
        from rouge_score import rouge_scorer
        from rouge_score.tokenizers import Tokenizer

        class ThaiSpaceTokenizer(Tokenizer):
            def tokenize(self, text: str) -> list[str]:
                return text.split(" ")

        oracle_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=ThaiSpaceTokenizer())

    ordered_rows = {row["ID"]: row for row in prepared_rows}
    for profile in ("fact", "list", "synthesis"):
        profile_rows = [row for row in prepared_rows if row["profile"] == profile]
        if not profile_rows:
            continue
        print(
            f"Generating {len(profile_rows)} {profile} validation answers "
            f"in batches of {runtime_config.GENERATOR_BATCH_SIZE}"
        )
        for start in range(0, len(profile_rows), runtime_config.GENERATOR_BATCH_SIZE):
            batch = profile_rows[start:start + runtime_config.GENERATOR_BATCH_SIZE]
            queries = [row["query"] for row in batch]
            paragraph_batches = [row["generation_paragraphs"] for row in batch]
            if runtime_config.ENABLE_ANSWER_CANDIDATES:
                candidate_batches = generator.batch_generate_candidates(
                    queries,
                    paragraph_batches,
                    profile=profile,
                    variants=runtime_config.ANSWER_CANDIDATE_VARIANTS,
                    max_seq_len=max_seq_len,
                )
                for row, candidates in zip(batch, candidate_batches):
                    mode = runtime_config.ANSWER_CANDIDATE_SELECTION_MODE
                    if mode == "oracle" and oracle_scorer is not None:
                        gold = tokenize_thai(row["gold_answer"])
                        chosen = max(
                            candidates,
                            key=lambda item: oracle_scorer.score(gold, tokenize_thai(item["answer"]))["rougeL"].fmeasure,
                        )
                    elif mode == "heuristic":
                        chosen = select_heuristic_candidate(
                            row["query"],
                            candidates,
                            row["generation_paragraphs"],
                            profile,
                        )
                    elif mode == "ranker" and answer_ranker is not None:
                        prediction = answer_ranker.select(
                            row["query"],
                            candidates,
                            row["generation_paragraphs"],
                            profile,
                        )
                        chosen = {"variant": prediction.variant, "answer": prediction.answer}
                        ordered_rows[row["ID"]]["answer_ranker_score"] = prediction.score
                    else:
                        chosen = next((item for item in candidates if item["variant"] == "base"), candidates[0])
                    ordered_rows[row["ID"]]["abstractive"] = chosen["answer"]
                    ordered_rows[row["ID"]]["answer_variant"] = chosen["variant"]
                    ordered_rows[row["ID"]]["answer_candidates"] = candidates
            else:
                outputs = generator.batch_generate(
                    queries,
                    paragraph_batches,
                    profile=profile,
                    max_seq_len=max_seq_len,
                )
                for row, answer in zip(batch, outputs):
                    ordered_rows[row["ID"]]["abstractive"] = answer
                    ordered_rows[row["ID"]]["answer_variant"] = "base"
    return [ordered_rows[row["ID"]] for row in prepared_rows]


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
    tokenizer.padding_side = "left"

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
    ref_selector = load_ref_selector_if_available()
    if ref_selector is not None and runtime_config.ENABLE_LEARNED_REF_SELECTOR:
        ref_selector.load_model()
    evidence_selector = load_evidence_set_selector_if_available()
    if evidence_selector is not None and runtime_config.ENABLE_EVIDENCE_SET_SELECTOR:
        evidence_selector.load_model()
    answer_ranker = load_answer_ranker_if_available()
    if answer_ranker is not None and runtime_config.ANSWER_CANDIDATE_SELECTION_MODE == "ranker":
        answer_ranker.load_model()

    doc_embedding_index = build_document_embedding_index(val_docs, embedder)

    generator = Generator(system_prompt=SYSTEM_PROMPT)
    generator.model = model
    generator.tokenizer = tokenizer

    prepared_rows = []
    gold_rows = []
    dense_retrievals = []
    reranked_retrievals = []
    predicted_refs_list = []
    rule_refs_list = []
    selection_results = []
    predicted_profiles = {}
    selection_by_id = {}
    retrieval_diagnostics = []
    effective_retrieval_top_k = max(
        args.retrieval_top_k,
        runtime_config.RERANK_TOP_K if reranker is not None else 0,
    )
    for sample in val_raw_samples:
        dense_retrieved = retrieve_paragraphs(
            doc_embedding_index,
            sample["doc_id"],
            sample["query"],
            embedder,
            effective_retrieval_top_k,
        )
        dense_retrievals.append(dense_retrieved)
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
                    effective_retrieval_top_k,
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
        selection_result = select_references_with_diagnostics(
            sample["query"],
            retrieved,
            profile=predicted_profile,
            n=args.reference_top_n,
            mode="dynamic_rules_then_llm_arbiter" if runtime_config.ENABLE_LLM_REF_ARBITER else None,
            ref_selector=ref_selector if runtime_config.ENABLE_LEARNED_REF_SELECTOR else None,
            evidence_selector=evidence_selector if runtime_config.ENABLE_EVIDENCE_SET_SELECTOR else None,
            generator=generator,
        )
        predicted_refs = selection_result.selected_refs
        rule_refs = selection_result.rule_refs
        predicted_refs_list.append(predicted_refs)
        rule_refs_list.append(rule_refs)
        selection_results.append(selection_result)
        selection_by_id[sample["ID"]] = selection_result
        generation_paragraphs = build_generation_context(
            sample["query"],
            retrieved,
            predicted_refs,
            predicted_profile,
        )
        prepared_rows.append(
            {
                "ID": sample["ID"],
                "query": sample["query"],
                "generation_paragraphs": generation_paragraphs,
                "abstractive": None,
                "gold_answer": sample["answer"],
                "refs": ",".join(predicted_refs),
                "profile": predicted_profile,
            }
        )
        retrieval_diagnostics.append(
            {
                "ID": sample["ID"],
                "profile": predicted_profile,
                "predicted_refs": predicted_refs,
                "rule_refs": rule_refs,
                "selector_refs": selection_result.selector_refs,
                "evidence_refs": selection_result.evidence_refs,
                "arbiter_refs": selection_result.arbiter_refs,
                "selector_used": selection_result.selector_used,
                "evidence_used": selection_result.evidence_used,
                "arbiter_triggered": selection_result.arbiter_triggered,
                "arbiter_used": selection_result.arbiter_used,
                "arbiter_fallback": selection_result.arbiter_fallback,
                "dense_top_candidates": [
                    {
                        "para_id": item["para_id"],
                        "score": float(item.get("score", 0.0)),
                    }
                    for item in dense_retrieved[:5]
                ],
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

    prediction_rows = generate_prepared_rows_in_batches(
        generator,
        prepared_rows,
        max_seq_len=args.max_seq_len,
        answer_ranker=answer_ranker,
    )
    pred_df = pd.DataFrame(prediction_rows)
    gold_df = pd.DataFrame(gold_rows)
    pred_df[["ID", "abstractive", "refs", "profile", "answer_variant"]].to_csv(
        args.output_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    if runtime_config.ENABLE_ANSWER_CANDIDATES:
        save_json(
            args.output_dir / "answer_candidates.json",
            {
                "rows": [
                    {
                        "ID": row["ID"],
                        "profile": row["profile"],
                        "chosen_variant": row.get("answer_variant", "base"),
                        "query": row["query"],
                        "gold_answer": row.get("gold_answer", ""),
                        "selected_refs": row.get("refs", ""),
                        "evidence": row.get("generation_paragraphs", []),
                        "candidates": row.get("answer_candidates", []),
                    }
                    for row in prediction_rows
                ]
            },
        )

    metrics, merged = run_evaluation(gold_df, pred_df, embedder)
    gold_refs_list = [sample["gold_refs"] for sample in val_raw_samples]
    dense_metrics = compute_retrieval_metrics(gold_refs_list, dense_retrievals)
    reranked_metrics = compute_retrieval_metrics(gold_refs_list, reranked_retrievals)
    selected_metrics = compute_selected_reference_metrics(gold_refs_list, predicted_refs_list)
    rule_selected_metrics = compute_selected_reference_metrics(gold_refs_list, rule_refs_list)
    selected_profile_metrics = compute_selected_reference_metrics_by_profile(
        gold_refs_list,
        predicted_refs_list,
        [predicted_profiles[sample["ID"]] for sample in val_raw_samples],
    )
    arbiter_metrics = compute_arbiter_metrics(selection_results)
    selector_metrics = compute_selector_metrics(selection_results)
    evidence_set_metrics = compute_evidence_set_metrics(selection_results)
    arbiter_used_indices = [
        index for index, result in enumerate(selection_results)
        if result.arbiter_used and not result.arbiter_fallback
    ]
    if arbiter_used_indices:
        arbiter_selected_metrics = compute_selected_reference_metrics(
            [gold_refs_list[index] for index in arbiter_used_indices],
            [predicted_refs_list[index] for index in arbiter_used_indices],
        )
        metrics["arbiter_selected_ref_iou"] = arbiter_selected_metrics["selected_ref_iou"]
    else:
        metrics["arbiter_selected_ref_iou"] = 0.0
    selector_used_indices = [index for index, result in enumerate(selection_results) if result.selector_used]
    if selector_used_indices:
        selector_selected_metrics = compute_selected_reference_metrics(
            [gold_refs_list[index] for index in selector_used_indices],
            [predicted_refs_list[index] for index in selector_used_indices],
        )
        metrics["selector_selected_ref_iou"] = selector_selected_metrics["selected_ref_iou"]
    else:
        metrics["selector_selected_ref_iou"] = 0.0
    evidence_used_indices = [index for index, result in enumerate(selection_results) if result.evidence_used]
    if evidence_used_indices:
        evidence_selected_metrics = compute_selected_reference_metrics(
            [gold_refs_list[index] for index in evidence_used_indices],
            [predicted_refs_list[index] for index in evidence_used_indices],
        )
        metrics["evidence_set_selected_ref_iou"] = evidence_selected_metrics["selected_ref_iou"]
    else:
        metrics["evidence_set_selected_ref_iou"] = 0.0
    metrics.update({f"dense_{key}": value for key, value in dense_metrics.items()})
    metrics.update({f"reranked_{key}": value for key, value in reranked_metrics.items()})
    metrics.update(selected_metrics)
    metrics.update({f"rule_{key}": value for key, value in rule_selected_metrics.items()})
    metrics.update(selected_profile_metrics)
    metrics.update(selector_metrics)
    metrics.update(evidence_set_metrics)
    metrics.update(arbiter_metrics)
    metrics.update(reranked_metrics)
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
    metrics["effective_retrieval_top_k"] = float(effective_retrieval_top_k)
    profile_counts = Counter(predicted_profiles.values())
    total_profiles = max(1, len(predicted_profiles))
    metrics["pred_profile_pct_fact"] = profile_counts.get("fact", 0) / total_profiles
    metrics["pred_profile_pct_list"] = profile_counts.get("list", 0) / total_profiles
    metrics["pred_profile_pct_synthesis"] = profile_counts.get("synthesis", 0) / total_profiles
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
        selection_result = selection_by_id[sample["ID"]]
        rouge_score = float(merged_row["rougeL"])
        ss_score = float(merged_row["SS-score"])
        if not gold_refs.intersection({item["para_id"] for item in retrieved[:10]}):
            failure_type = "retrieval_miss"
        elif selection_result.arbiter_fallback:
            failure_type = "arbiter_fallback"
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
                "rule_refs": selection_result.rule_refs,
                "selector_refs": selection_result.selector_refs,
                "evidence_refs": selection_result.evidence_refs,
                "arbiter_refs": selection_result.arbiter_refs,
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
