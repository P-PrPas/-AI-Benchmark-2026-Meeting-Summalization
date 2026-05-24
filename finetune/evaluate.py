from __future__ import annotations

import argparse

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
    build_user_prompt,
    cache_dir_as_str,
    configure_cache_env,
    ensure_local_model_exists,
    ensure_path_exists,
    filter_queries_by_ids,
    get_model_device,
    load_json,
    load_training_data,
    resolve_model_source,
    resolve_path,
    run_evaluation,
    save_json,
    select_top_refs,
    retrieve_paragraphs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate finetuned typhoon2.5-qwen3-4b on held-out validation")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_OUTPUT_DIR / "final_merged"))
    parser.add_argument("--adapter-path")
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--reference-top-n", type=int, default=3)
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

    doc_embedding_index = build_document_embedding_index(val_docs, embedder)

    @torch.inference_mode()
    def generate_answer(query: str, paragraphs: list[dict[str, str]]) -> str:
        context = "\n".join(f"[{paragraph['para_id']}] {paragraph['text']}" for paragraph in paragraphs)
        if not context.strip():
            context = "(ไม่มีข้อมูลอ้างอิง)"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(context, query)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        )
        device = get_model_device(model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer or "ไม่พบข้อมูลในเอกสาร"

    prediction_rows = []
    gold_rows = []
    for sample in val_raw_samples:
        retrieved = retrieve_paragraphs(
            doc_embedding_index,
            sample["doc_id"],
            sample["query"],
            embedder,
            args.retrieval_top_k,
        )
        predicted_refs = select_top_refs(retrieved, args.reference_top_n)
        predicted_answer = generate_answer(sample["query"], retrieved[: args.retrieval_top_k])
        prediction_rows.append(
            {
                "ID": sample["ID"],
                "abstractive": predicted_answer,
                "refs": ",".join(predicted_refs),
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

    metrics, _ = run_evaluation(gold_df, pred_df, embedder)
    save_json(args.output_dir / "validation_metrics.json", metrics)

    print(f"Validation samples={len(val_raw_samples)}")
    print(f"Saved predictions to {args.output_dir / 'val_predictions.csv'}")
    print(f"Saved metrics to {args.output_dir / 'validation_metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
