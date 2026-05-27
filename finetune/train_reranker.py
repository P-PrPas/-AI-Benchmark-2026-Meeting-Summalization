from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from src import config as runtime_config
from src.reranker import RERANK_ASSISTANT_SUFFIX, RERANK_SYSTEM_PREFIX, format_reranker_pair

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
    grouped_doc_split,
    load_training_data,
    resolve_model_source,
    resolve_path,
    retrieve_paragraphs,
    save_json,
    set_global_seed,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 reranker with official causal-LM yes/no scoring")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_RERANK_MODEL_PATH))
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--retrieval-top-k", type=int, default=runtime_config.RERANK_TOP_K)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--instruction", default=runtime_config.RERANK_INSTRUCTION)
    parser.add_argument("--skip-merge", action="store_true")
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
        args.output_dir or (project_root / "artifacts" / "qwen3_reranker_4b_qlora"),
        project_root=project_root,
    )
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root)
    return args


def validate_args(args: argparse.Namespace) -> None:
    ensure_path_exists(args.train_json_path, "Train JSON")
    ensure_local_model_exists(args.model_name_or_path, "Reranker model", project_root=args.project_root)
    ensure_local_model_exists(args.embed_model_name_or_path, "Embed model", project_root=args.project_root)


def build_pair_rows(
    queries: Sequence[dict[str, Any]],
    doc_lookup: dict[str, dict[str, Any]],
    doc_embedding_index: dict[str, dict[str, Any]],
    embedder,
    *,
    retrieval_top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        gold_refs = set(query.get("refs", []))
        doc = doc_lookup[query["doc_id"]]
        dense_retrieved = retrieve_paragraphs(
            doc_embedding_index,
            query["doc_id"],
            query["query"],
            embedder,
            retrieval_top_k,
        )
        candidates: dict[str, dict[str, Any]] = {}
        for item in dense_retrieved:
            candidates[item["para_id"]] = {
                "para_id": item["para_id"],
                "text": item["text"],
                "source": "dense_topk",
            }
        for paragraph in doc["paragraphs"]:
            if paragraph["para_id"] in gold_refs:
                candidates[paragraph["para_id"]] = {
                    "para_id": paragraph["para_id"],
                    "text": paragraph["text"],
                    "source": "gold",
                }
        for para_id, item in candidates.items():
            rows.append(
                {
                    "ID": query["ID"],
                    "doc_id": query["doc_id"],
                    "query": query["query"],
                    "para_id": para_id,
                    "text": item["text"],
                    "label": 1 if para_id in gold_refs else 0,
                    "source": item["source"],
                }
            )
    return rows


def tokenize_rows(
    rows: Sequence[dict[str, Any]],
    tokenizer,
    *,
    instruction: str,
    max_seq_len: int,
) -> tuple[Any, list[dict[str, Any]]]:
    from datasets import Dataset

    encoded_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for reranker fine-tuning.")

    for row in rows:
        pair_text = format_reranker_pair(row["query"], row["text"], instruction=instruction)
        prompt_text = RERANK_SYSTEM_PREFIX + pair_text + RERANK_ASSISTANT_SUFFIX
        target_text = "yes" if row["label"] else "no"

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [eos_token_id]
        full_ids = prompt_ids + target_ids
        if len(full_ids) > max_seq_len:
            dropped_rows.append(
                {
                    "ID": row["ID"],
                    "para_id": row["para_id"],
                    "label": row["label"],
                    "reason": f"overlength:{len(full_ids)}",
                }
            )
            continue
        encoded_rows.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": ([-100] * len(prompt_ids)) + target_ids,
            }
        )
    if not encoded_rows:
        raise ValueError("No usable reranker training rows remain after tokenization.")
    return Dataset.from_list(encoded_rows), dropped_rows


class CausalLMDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id for batching reranker samples.")

        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_length = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    validate_args(args)
    configure_cache_env(args.cache_dir, offline=True)
    set_global_seed(args.seed)

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("Reranker training requires a CUDA-enabled server runtime.")

    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    train_queries, val_queries, train_doc_ids, val_doc_ids = grouped_doc_split(
        queries,
        args.val_doc_ratio,
        args.seed,
    )

    embed_source = resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root)
    embedder = SentenceTransformer(
        embed_source,
        device="cuda",
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    train_docs = [doc_lookup[doc_id] for doc_id in sorted(train_doc_ids)]
    val_docs = [doc_lookup[doc_id] for doc_id in sorted(val_doc_ids)]
    train_doc_embedding_index = build_document_embedding_index(train_docs, embedder)
    val_doc_embedding_index = build_document_embedding_index(val_docs, embedder)

    train_rows = build_pair_rows(
        train_queries,
        doc_lookup,
        train_doc_embedding_index,
        embedder,
        retrieval_top_k=args.retrieval_top_k,
    )
    val_rows = build_pair_rows(
        val_queries,
        doc_lookup,
        val_doc_embedding_index,
        embedder,
        retrieval_top_k=args.retrieval_top_k,
    )

    model_source = resolve_model_source(args.model_name_or_path, project_root=args.project_root)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        cache_dir=cache_dir_as_str(args.cache_dir),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must define pad_token or eos_token for reranker fine-tuning.")

    train_dataset, dropped_train_rows = tokenize_rows(
        train_rows,
        tokenizer,
        instruction=args.instruction,
        max_seq_len=args.max_seq_len,
    )
    val_dataset, dropped_val_rows = tokenize_rows(
        val_rows,
        tokenizer,
        instruction=args.instruction,
        max_seq_len=args.max_seq_len,
    )

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir_as_str(args.cache_dir),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )

    effective_batch_size = max(1, args.train_batch_size * args.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    warmup_steps = max(0, int(max(1, steps_per_epoch * args.num_train_epochs) * args.warmup_ratio))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "split_metadata.json",
        {
            "train_doc_ids": sorted(train_doc_ids),
            "val_doc_ids": sorted(val_doc_ids),
            "train_query_ids": [query["ID"] for query in train_queries],
            "val_query_ids": [query["ID"] for query in val_queries],
            "seed": args.seed,
            "val_doc_ratio": args.val_doc_ratio,
        },
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=CausalLMDataCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    final_model_dir = args.output_dir / "final_model"
    if not args.skip_merge:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir_as_str(args.cache_dir),
        )
        merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = merged_model.merge_and_unload()
        final_model_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(final_model_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(final_model_dir)

    save_json(
        args.output_dir / "reranker_training_metadata.json",
        {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_rows_after_tokenization": len(train_dataset),
            "val_rows_after_tokenization": len(val_dataset),
            "dropped_train_rows": dropped_train_rows[:200],
            "dropped_val_rows": dropped_val_rows[:200],
            "train_query_count": len(train_queries),
            "val_query_count": len(val_queries),
            "train_doc_ids": sorted(train_doc_ids),
            "val_doc_ids": sorted(val_doc_ids),
            "model_name_or_path": model_source,
            "embed_model_name_or_path": embed_source,
            "output_dir": str(args.output_dir),
            "instruction": args.instruction,
        },
    )
    print(f"Saved adapter to {adapter_dir}")
    if not args.skip_merge:
        print(f"Saved merged model to {final_model_dir}")


if __name__ == "__main__":
    main()
