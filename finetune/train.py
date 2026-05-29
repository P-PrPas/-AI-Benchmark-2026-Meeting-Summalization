from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import sys
from pathlib import Path

from src import config as runtime_config
from src.generator import Generator
from src.reranker import load_reranker_if_available

from .common import (
    DEFAULT_ARTIFACT_NAME,
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_EMBED_MODEL_PATH,
    LANTA_CACHE_ROOT,
    LANTA_PROJECT_ROOT,
    SupervisedDataCollator,
    build_augmented_training_samples,
    build_document_embedding_index,
    build_raw_samples,
    build_split_metadata,
    build_tokenized_dataset,
    cache_dir_as_str,
    configure_cache_env,
    ensure_local_model_exists,
    ensure_path_exists,
    grouped_doc_split,
    load_training_data,
    resolve_model_source,
    resolve_path,
    save_json,
    set_global_seed,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


class JsonlLoggingCallback:
    """Persist Trainer log events to disk while training is running."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = dict(logs)
        payload.setdefault("step", state.global_step)
        payload.setdefault("epoch", state.epoch)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune typhoon2.5-qwen3-4b for Thai RAG QA")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_BASE_MODEL_PATH))
    parser.add_argument("--embed-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--rerank-model-name-or-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--max-seq-len", type=int, default=runtime_config.GENERATOR_MAX_SEQ_LEN)
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument(
        "--merge-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument("--debug-max-train-samples", type=int)
    parser.add_argument("--debug-max-val-samples", type=int)
    parser.add_argument("--disable-training-augmentation", action="store_true")
    parser.add_argument("--oracle-fraction", type=float, default=0.85)
    parser.add_argument("--noisy-fraction", type=float, default=0.15)
    parser.add_argument("--synthetic-fraction", type=float, default=0.0)
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
    args.rerank_model_name_or_path = args.rerank_model_name_or_path or os.environ.get("CAMNET_RERANK_MODEL_PATH")
    return args


def validate_args(args: argparse.Namespace) -> None:
    ensure_path_exists(args.train_json_path, "Train JSON")
    ensure_local_model_exists(args.model_name_or_path, "Base model", project_root=args.project_root)
    ensure_local_model_exists(args.embed_model_name_or_path, "Embed model", project_root=args.project_root)
    if args.rerank_model_name_or_path:
        ensure_local_model_exists(args.rerank_model_name_or_path, "Rerank model", project_root=args.project_root)


def print_runtime_config(args: argparse.Namespace) -> None:
    print("Runtime configuration")
    print(f"  project_root={args.project_root}")
    print(f"  train_json_path={args.train_json_path}")
    print(f"  model_name_or_path={resolve_model_source(args.model_name_or_path, args.project_root)}")
    print(f"  embed_model_name_or_path={resolve_model_source(args.embed_model_name_or_path, args.project_root)}")
    print(f"  rerank_model_name_or_path={args.rerank_model_name_or_path}")
    print(f"  use_reranker={runtime_config.USE_RERANKER}")
    print(f"  enable_llm_ref_arbiter={runtime_config.ENABLE_LLM_REF_ARBITER}")
    print(f"  ref_arbiter_trigger_mode={runtime_config.REF_ARBITER_TRIGGER_MODE}")
    print(f"  enable_fact_answer_rewrite={runtime_config.ENABLE_FACT_ANSWER_REWRITE}")
    print(f"  output_dir={args.output_dir}")
    print(f"  cache_dir={args.cache_dir}")
    print(f"  artifact_name={DEFAULT_ARTIFACT_NAME}")
    print(f"  skip_merge={args.skip_merge}")
    print(f"  merge_dtype={args.merge_dtype}")
    print(f"  disable_training_augmentation={args.disable_training_augmentation}")
    print(
        "  training_mix="
        f"oracle:{args.oracle_fraction} noisy:{args.noisy_fraction} synthetic:{args.synthetic_fraction}"
    )


def resolve_merge_dtype(torch_module, merge_dtype: str):
    if merge_dtype == "float16":
        return torch_module.float16
    if merge_dtype == "bfloat16":
        return torch_module.bfloat16
    if merge_dtype == "float32":
        return torch_module.float32
    if torch_module.cuda.is_available():
        return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16
    return torch_module.float32


def merge_and_save_model(
    *,
    model_source: str,
    adapter_dir: Path,
    merged_dir: Path,
    tokenizer,
    cache_dir,
    torch_module,
    merge_dtype,
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    merged_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_merge_dtype(torch_module, merge_dtype)
    print(f"Loading base model for merge from {model_source}")
    print(f"Merging adapter from {adapter_dir} into full model at dtype={dtype}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        device_map="auto" if torch_module.cuda.is_available() else None,
        trust_remote_code=True,
        cache_dir=cache_dir_as_str(cache_dir),
        low_cpu_mem_usage=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(merged_dir)
    print(f"Saved merged model to {merged_dir}")


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    validate_args(args)
    configure_cache_env(args.cache_dir, offline=True)
    print_runtime_config(args)
    set_global_seed(args.seed)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA-enabled server runtime.")

    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    train_queries, val_queries, train_doc_ids, val_doc_ids = grouped_doc_split(
        queries,
        args.val_doc_ratio,
        args.seed,
    )

    if args.debug_max_train_samples is not None:
        train_queries = train_queries[: args.debug_max_train_samples]
    if args.debug_max_val_samples is not None:
        val_queries = val_queries[: args.debug_max_val_samples]

    embed_source = resolve_model_source(args.embed_model_name_or_path, project_root=args.project_root)
    embedder = SentenceTransformer(
        embed_source,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=cache_dir_as_str(args.cache_dir),
    )
    reranker = load_reranker_if_available(args.rerank_model_name_or_path)
    if reranker is not None:
        reranker.load_model()
    train_docs = [doc_lookup[doc_id] for doc_id in sorted(train_doc_ids)]
    train_doc_embedding_index = build_document_embedding_index(train_docs, embedder)

    if args.disable_training_augmentation:
        train_raw_samples, train_missing_refs = build_raw_samples(train_queries, doc_lookup)
        augmentation_counts = {"oracle": len(train_raw_samples), "noisy_retrieved": 0, "synthetic_style": 0}
    else:
        retrieval_generator = None
        if runtime_config.ENABLE_LLM_REF_ARBITER:
            retrieval_generator = Generator(model_path=args.model_name_or_path)
            retrieval_generator.load_model()
        train_raw_samples, train_missing_refs, augmentation_counts = build_augmented_training_samples(
            train_queries,
            doc_lookup,
            train_doc_embedding_index,
            embedder,
            reranker=reranker,
            generator=retrieval_generator,
            seed=args.seed,
            oracle_fraction=args.oracle_fraction,
            noisy_fraction=args.noisy_fraction,
            synthetic_fraction=args.synthetic_fraction,
        )
        if retrieval_generator is not None:
            del retrieval_generator
            retrieval_generator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    val_raw_samples, val_missing_refs = build_raw_samples(val_queries, doc_lookup)

    model_source = resolve_model_source(args.model_name_or_path, project_root=args.project_root)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        cache_dir=cache_dir_as_str(args.cache_dir),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, dropped_train = build_tokenized_dataset(
        train_raw_samples,
        tokenizer,
        args.max_seq_len,
    )
    val_dataset, dropped_val = build_tokenized_dataset(
        val_raw_samples,
        tokenizer,
        args.max_seq_len,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = build_split_metadata(
        seed=args.seed,
        val_ratio=args.val_doc_ratio,
        train_doc_ids=train_doc_ids,
        val_doc_ids=val_doc_ids,
        train_queries=train_queries,
        val_queries=val_queries,
        dropped_train=dropped_train,
        dropped_val=dropped_val,
    )
    save_json(args.output_dir / "split_metadata.json", split_metadata)
    save_json(args.output_dir / "missing_train_refs.json", {"rows": train_missing_refs})
    save_json(args.output_dir / "missing_val_refs.json", {"rows": val_missing_refs})
    save_json(args.output_dir / "training_mix_counts.json", augmentation_counts)
    save_json(
        args.output_dir / "runtime_paths.json",
        {
            "project_root": str(args.project_root),
            "output_dir": str(args.output_dir),
            "model_name_or_path": model_source,
            "embed_model_name_or_path": embed_source,
            "rerank_model_name_or_path": args.rerank_model_name_or_path,
            "cache_dir": str(args.cache_dir),
        },
    )

    print(f"Loaded docs={len(docs)} queries={len(queries)}")
    print(f"Train docs={len(train_doc_ids)} val docs={len(val_doc_ids)}")
    print(f"Train samples={len(train_raw_samples)} val samples={len(val_raw_samples)}")
    print(f"Training mix counts={augmentation_counts}")
    print(f"Tokenized train rows={len(train_dataset)} dropped={len(dropped_train)}")
    print(f"Tokenized val rows={len(val_dataset)} dropped={len(dropped_val)}")

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
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    bf16 = torch.cuda.is_bf16_supported()
    if args.warmup_steps is None:
        effective_batch_size = max(1, args.train_batch_size * args.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch_size))
        total_training_steps = steps_per_epoch * max(1, math.ceil(args.num_train_epochs))
        args.warmup_steps = max(0, int(total_training_steps * args.warmup_ratio))
        print(
            "Derived warmup_steps from warmup_ratio: "
            f"warmup_ratio={args.warmup_ratio} total_training_steps={total_training_steps} "
            f"warmup_steps={args.warmup_steps}"
        )
    training_args_kwargs = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": True,
        "lr_scheduler_type": "cosine",
        "bf16": bf16,
        "fp16": not bf16,
        "report_to": "none",
        "remove_unused_columns": False,
        "optim": "paged_adamw_8bit",
        "seed": args.seed,
        "dataloader_pin_memory": True,
    }

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_args_signature.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "This transformers build does not expose evaluation_strategy or eval_strategy "
            "on TrainingArguments."
        )

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": SupervisedDataCollator(tokenizer),
        "callbacks": [JsonlLoggingCallback(args.output_dir / "trainer_logs.jsonl")],
    }
    trainer_signature = set(Trainer.__init__.__code__.co_varnames)
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_state()
    save_json(args.output_dir / "train_metrics.json", train_result.metrics)
    save_json(args.output_dir / "trainer_log_history.json", {"rows": trainer.state.log_history})

    final_adapter_dir = args.output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Saved adapter to {final_adapter_dir}")

    final_merged_dir = args.output_dir / "final_merged"
    if args.skip_merge:
        print("Skipping merged-model export because --skip-merge was provided")
    else:
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        merge_and_save_model(
            model_source=model_source,
            adapter_dir=final_adapter_dir,
            merged_dir=final_merged_dir,
            tokenizer=tokenizer,
            cache_dir=args.cache_dir,
            torch_module=torch,
            merge_dtype=args.merge_dtype,
        )

    print(f"Saved split metadata to {args.output_dir / 'split_metadata.json'}")
    print(f"Saved train metrics to {args.output_dir / 'train_metrics.json'}")
    print(f"Saved trainer logs to {args.output_dir / 'trainer_logs.jsonl'}")


if __name__ == "__main__":
    main()
