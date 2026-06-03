from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src import config as runtime_config
from src.llm_only import (
    build_llm_only_prompt,
    fallback_refs_by_answer_overlap,
    normalize_prompt_mode,
    paragraph_ids,
    parse_llm_only_output,
    truncate_paragraphs_by_chars,
)

from .common import (
    DEFAULT_EMBED_MODEL_PATH,
    LANTA_CACHE_ROOT,
    LANTA_MODEL_ROOT,
    LANTA_PROJECT_ROOT,
    build_raw_samples,
    build_split_metadata,
    cache_dir_as_str,
    configure_cache_env,
    ensure_local_model_exists,
    ensure_path_exists,
    filter_queries_by_ids,
    grouped_doc_split,
    load_json,
    load_training_data,
    resolve_model_source,
    resolve_path,
    run_evaluation,
    save_json,
)


DEFAULT_LLM_ONLY_MODEL_PATH = LANTA_MODEL_ROOT / "Qwen3.5-9B-finetuned-bf16"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LLM-only full-document QA on held-out validation")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--model-name-or-path", default=os.environ.get("CAMNET_LLM_ONLY_MODEL_PATH", str(DEFAULT_LLM_ONLY_MODEL_PATH)))
    parser.add_argument("--semantic-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--split-metadata-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--prompt-mode", default=runtime_config.LLM_ONLY_PROMPT_MODE)
    parser.add_argument("--max-seq-len", type=int, default=runtime_config.LLM_ONLY_MAX_SEQ_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=runtime_config.LLM_ONLY_MAX_NEW_TOKENS)
    parser.add_argument("--max-doc-chars", type=int, default=runtime_config.LLM_ONLY_MAX_DOC_CHARS)
    parser.add_argument("--batch-size", type=int, default=runtime_config.LLM_ONLY_BATCH_SIZE)
    parser.add_argument("--repetition-penalty", type=float, default=runtime_config.LLM_ONLY_REPETITION_PENALTY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-ref-fallback", action="store_true", default=runtime_config.LLM_ONLY_REF_FALLBACK)
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
        args.output_dir or (project_root / "artifacts" / f"llm_only_{args.prompt_mode}"),
        project_root=project_root,
    )
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root)
    args.split_metadata_path = (
        resolve_path(args.split_metadata_path, project_root=project_root)
        if args.split_metadata_path
        else None
    )
    args.prompt_mode = normalize_prompt_mode(args.prompt_mode)
    args.use_chat_template = not args.no_chat_template and runtime_config.LLM_ONLY_USE_CHAT_TEMPLATE
    return args


def validate_args(args: argparse.Namespace) -> None:
    ensure_path_exists(args.train_json_path, "Train JSON")
    ensure_local_model_exists(args.model_name_or_path, "LLM-only model", project_root=args.project_root)
    ensure_local_model_exists(args.semantic_model_name_or_path, "Semantic metric model", project_root=args.project_root)
    if args.split_metadata_path is not None:
        ensure_path_exists(args.split_metadata_path, "Split metadata")


def print_runtime_config(args: argparse.Namespace) -> None:
    print("Runtime configuration")
    print(f"  project_root={args.project_root}")
    print(f"  train_json_path={args.train_json_path}")
    print(f"  model_name_or_path={resolve_model_source(args.model_name_or_path, args.project_root)}")
    print(f"  semantic_model_name_or_path={resolve_model_source(args.semantic_model_name_or_path, args.project_root)}")
    print(f"  prompt_mode={args.prompt_mode}")
    print(f"  output_dir={args.output_dir}")
    print(f"  cache_dir={args.cache_dir}")
    print(f"  max_seq_len={args.max_seq_len}")
    print(f"  max_new_tokens={args.max_new_tokens}")
    print(f"  max_doc_chars={args.max_doc_chars}")
    print(f"  batch_size={args.batch_size}")
    print(f"  load_in_4bit={args.load_in_4bit}")
    print(f"  use_chat_template={args.use_chat_template}")
    print(f"  enable_ref_fallback={args.enable_ref_fallback}")


def load_validation_payload(args: argparse.Namespace):
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
        save_json(split_metadata_path, split_metadata)

    val_queries = filter_queries_by_ids(queries, split_metadata.get("val_query_ids") or [])
    if not val_queries:
        raise ValueError("No validation queries found from split metadata.")
    val_raw_samples, missing_val_refs = build_raw_samples(val_queries, doc_lookup)
    return docs, doc_lookup, val_raw_samples, missing_val_refs


def render_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    use_chat_template: bool,
) -> list[str]:
    if not use_chat_template:
        return list(prompts)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def batch_generate_raw(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
    repetition_penalty: float,
    use_chat_template: bool,
) -> list[str]:
    import torch

    outputs: list[str] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        batch_prompts = prompts[start:start + max(1, batch_size)]
        rendered = render_prompts(tokenizer, batch_prompts, use_chat_template=use_chat_template)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded.input_ids.shape[1]
        outputs.extend(
            tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            for row in generated
        )
        print(f"Generated {min(start + len(batch_prompts), len(prompts))}/{len(prompts)}")
    return outputs


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    validate_args(args)
    configure_cache_env(args.cache_dir, offline=True)
    print_runtime_config(args)

    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("LLM-only evaluation requires a CUDA-enabled server runtime.")

    _, doc_lookup, val_raw_samples, missing_val_refs = load_validation_payload(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "missing_eval_refs.json", {"rows": missing_val_refs})

    model_source = resolve_model_source(args.model_name_or_path, project_root=args.project_root)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        cache_dir=cache_dir_as_str(args.cache_dir),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "cache_dir": cache_dir_as_str(args.cache_dir),
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True

    prepared: list[dict[str, Any]] = []
    prompts: list[str] = []
    for sample in val_raw_samples:
        doc = doc_lookup[sample["doc_id"]]
        paragraphs = truncate_paragraphs_by_chars(doc["paragraphs"], max_doc_chars=args.max_doc_chars)
        valid_refs = paragraph_ids(paragraphs)
        prompt = build_llm_only_prompt(
            sample["query"],
            paragraphs,
            mode=args.prompt_mode,
            max_doc_chars=args.max_doc_chars,
        )
        prompts.append(prompt)
        prepared.append(
            {
                **sample,
                "paragraphs": paragraphs,
                "valid_refs": valid_refs,
                "prompt_chars": len(prompt),
            }
        )

    print(f"Validation samples={len(prepared)}")
    raw_outputs = batch_generate_raw(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        use_chat_template=args.use_chat_template,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    prediction_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    parse_errors = 0
    invalid_ref_rows = 0
    empty_ref_rows = 0
    for sample, raw_response in zip(prepared, raw_outputs):
        parsed = parse_llm_only_output(raw_response, sample["valid_refs"])
        refs = parsed.refs
        if not refs and args.enable_ref_fallback:
            refs = fallback_refs_by_answer_overlap(parsed.abstractive, sample["paragraphs"], max_refs=1)
        parse_errors += int(parsed.parse_error)
        invalid_ref_rows += int(bool(parsed.invalid_refs))
        empty_ref_rows += int(not refs)
        prediction_rows.append(
            {
                "ID": sample["ID"],
                "abstractive": parsed.abstractive,
                "refs": ",".join(refs),
            }
        )
        diagnostic_rows.append(
            {
                "ID": sample["ID"],
                "doc_id": sample["doc_id"],
                "query": sample["query"],
                "gold_answer": sample["answer"],
                "gold_refs": sample["gold_refs"],
                "pred_answer": parsed.abstractive,
                "pred_refs": refs,
                "raw_response": raw_response,
                "parse_error": parsed.parse_error,
                "invalid_refs": parsed.invalid_refs,
                "prompt_chars": sample["prompt_chars"],
                "available_ref_count": len(sample["valid_refs"]),
            }
        )

    pred_df = pd.DataFrame(prediction_rows)
    gold_df = pd.DataFrame(
        [
            {
                "ID": sample["ID"],
                "abstractive": sample["answer"],
                "refs": sample["gold_refs"],
            }
            for sample in val_raw_samples
        ]
    )
    pred_df.to_csv(args.output_dir / "val_predictions.csv", index=False, encoding="utf-8")

    semantic_source = resolve_model_source(args.semantic_model_name_or_path, project_root=args.project_root)
    semantic_model = SentenceTransformer(
        semantic_source,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    metrics, merged = run_evaluation(gold_df, pred_df, semantic_model)
    answer_lengths = [len(row["abstractive"]) for row in prediction_rows]
    ref_counts = [len(row["refs"].split(",")) if row["refs"] else 0 for row in prediction_rows]
    metrics["parse_error_rate"] = parse_errors / max(1, len(prediction_rows))
    metrics["invalid_ref_rate"] = invalid_ref_rows / max(1, len(prediction_rows))
    metrics["empty_ref_rate"] = empty_ref_rows / max(1, len(prediction_rows))
    metrics["pred_ref_count_mean"] = float(pd.Series(ref_counts).mean()) if ref_counts else 0.0
    metrics["pred_ref_count_pct_1"] = sum(1 for count in ref_counts if count == 1) / max(1, len(ref_counts))
    metrics["pred_ref_count_pct_2"] = sum(1 for count in ref_counts if count == 2) / max(1, len(ref_counts))
    metrics["pred_ref_count_pct_3_plus"] = sum(1 for count in ref_counts if count >= 3) / max(1, len(ref_counts))
    metrics["pred_answer_length_median"] = float(pd.Series(answer_lengths).median()) if answer_lengths else 0.0
    metrics["pred_answer_length_mean"] = float(pd.Series(answer_lengths).mean()) if answer_lengths else 0.0
    metrics["prompt_mode"] = args.prompt_mode
    metrics["use_chat_template"] = args.use_chat_template
    save_json(args.output_dir / "validation_metrics.json", metrics)
    save_json(args.output_dir / "llm_only_diagnostics.json", {"rows": diagnostic_rows})

    per_row = merged[["ID", "rougeL", "SS-score", "IoU"]].to_dict("records")
    profile_counts = Counter(sample["profile"] for sample in val_raw_samples)
    save_json(
        args.output_dir / "llm_only_failure_analysis.json",
        {
            "profile_counts": dict(profile_counts),
            "worst_rows": sorted(
                [
                    {
                        **row,
                        "query": next(item["query"] for item in diagnostic_rows if item["ID"] == row["ID"]),
                        "gold_refs": next(item["gold_refs"] for item in diagnostic_rows if item["ID"] == row["ID"]),
                        "pred_refs": next(item["pred_refs"] for item in diagnostic_rows if item["ID"] == row["ID"]),
                        "pred_answer": next(item["pred_answer"] for item in diagnostic_rows if item["ID"] == row["ID"]),
                    }
                    for row in per_row
                ],
                key=lambda item: (item["rougeL"], item["IoU"]),
            )[:50],
        },
    )

    print(f"Saved predictions to {args.output_dir / 'val_predictions.csv'}")
    print(f"Saved metrics to {args.output_dir / 'validation_metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
