from __future__ import annotations

import argparse
import gc
import inspect
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from src import config as runtime_config
from src.llm_only import (
    build_llm_only_prompt,
    build_llm_only_target,
    normalize_prompt_mode,
    truncate_paragraphs_by_chars,
)

from .common import (
    LANTA_CACHE_ROOT,
    LANTA_MODEL_ROOT,
    LANTA_PROJECT_ROOT,
    SupervisedDataCollator,
    build_raw_samples,
    build_split_metadata,
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
from .train import JsonlLoggingCallback, merge_and_save_model


DEFAULT_LLM_ONLY_MODEL_PATH = LANTA_MODEL_ROOT / "Qwen3.5-9B-finetuned-bf16"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continue fine-tuning LLM-only full-document QA")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--train-json-path")
    parser.add_argument("--model-name-or-path", default=os.environ.get("CAMNET_LLM_ONLY_MODEL_PATH", str(DEFAULT_LLM_ONLY_MODEL_PATH)))
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(LANTA_CACHE_ROOT))
    parser.add_argument("--prompt-mode", default=runtime_config.LLM_ONLY_PROMPT_MODE)
    parser.add_argument("--max-seq-len", type=int, default=runtime_config.LLM_ONLY_MAX_SEQ_LEN)
    parser.add_argument("--max-doc-chars", type=int, default=runtime_config.LLM_ONLY_MAX_DOC_CHARS)
    parser.add_argument("--val-doc-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--merge-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--debug-max-train-samples", type=int)
    parser.add_argument("--debug-max-val-samples", type=int)
    parser.add_argument("--no-chat-template", action="store_true")
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
        args.output_dir or (project_root / "artifacts" / f"llm_only_ft_{args.prompt_mode}"),
        project_root=project_root,
    )
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root)
    args.prompt_mode = normalize_prompt_mode(args.prompt_mode)
    args.use_chat_template = not args.no_chat_template and runtime_config.LLM_ONLY_USE_CHAT_TEMPLATE
    return args


def validate_args(args: argparse.Namespace) -> None:
    ensure_path_exists(args.train_json_path, "Train JSON")
    ensure_local_model_exists(args.model_name_or_path, "LLM-only base model", project_root=args.project_root)


def print_runtime_config(args: argparse.Namespace) -> None:
    print("Runtime configuration")
    print(f"  project_root={args.project_root}")
    print(f"  train_json_path={args.train_json_path}")
    print(f"  model_name_or_path={resolve_model_source(args.model_name_or_path, args.project_root)}")
    print(f"  prompt_mode={args.prompt_mode}")
    print(f"  output_dir={args.output_dir}")
    print(f"  cache_dir={args.cache_dir}")
    print(f"  max_seq_len={args.max_seq_len}")
    print(f"  max_doc_chars={args.max_doc_chars}")
    print(f"  learning_rate={args.learning_rate}")
    print(f"  num_train_epochs={args.num_train_epochs}")
    print(f"  use_chat_template={args.use_chat_template}")
    print(f"  skip_merge={args.skip_merge}")


def tokenize_llm_only_sample(
    sample: dict[str, Any],
    doc_lookup: dict[str, dict[str, Any]],
    tokenizer: Any,
    *,
    prompt_mode: str,
    max_seq_len: int,
    max_doc_chars: int,
    use_chat_template: bool,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    doc = doc_lookup[sample["doc_id"]]
    paragraphs = truncate_paragraphs_by_chars(doc["paragraphs"], max_doc_chars=max_doc_chars)
    prompt = build_llm_only_prompt(
        sample["query"],
        paragraphs,
        mode=prompt_mode,
        max_doc_chars=max_doc_chars,
    )
    target = build_llm_only_target(sample["answer"], sample["gold_refs"], mode=prompt_mode)

    if use_chat_template:
        prompt_messages = [{"role": "user", "content": prompt}]
        full_messages = prompt_messages + [{"role": "assistant", "content": target}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt_text = f"{prompt}\n"
        full_text = f"{prompt_text}{target}"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        return None, {"ID": sample["ID"], "reason": f"overlength:{len(full_ids)}"}
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None, {"ID": sample["ID"], "reason": "prompt_alignment_failed"}
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return (
        {
            "ID": sample["ID"],
            "doc_id": sample["doc_id"],
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        },
        None,
    )


def build_llm_only_dataset(
    samples: Sequence[dict[str, Any]],
    doc_lookup: dict[str, dict[str, Any]],
    tokenizer: Any,
    *,
    prompt_mode: str,
    max_seq_len: int,
    max_doc_chars: int,
    use_chat_template: bool,
):
    from datasets import Dataset

    encoded_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, str]] = []
    for sample in samples:
        encoded, dropped = tokenize_llm_only_sample(
            sample,
            doc_lookup,
            tokenizer,
            prompt_mode=prompt_mode,
            max_seq_len=max_seq_len,
            max_doc_chars=max_doc_chars,
            use_chat_template=use_chat_template,
        )
        if encoded is not None:
            encoded_rows.append(encoded)
        if dropped is not None:
            dropped_rows.append(dropped)
    if not encoded_rows:
        raise ValueError("No usable LLM-only samples remain after tokenization.")
    return Dataset.from_list(encoded_rows), dropped_rows


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    validate_args(args)
    configure_cache_env(args.cache_dir, offline=True)
    print_runtime_config(args)
    set_global_seed(args.seed)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("LLM-only QLoRA training requires a CUDA-enabled server runtime.")

    docs, queries, doc_lookup = load_training_data(args.train_json_path)
    train_queries, val_queries, train_doc_ids, val_doc_ids = grouped_doc_split(queries, args.val_doc_ratio, args.seed)
    if args.debug_max_train_samples is not None:
        train_queries = train_queries[: args.debug_max_train_samples]
    if args.debug_max_val_samples is not None:
        val_queries = val_queries[: args.debug_max_val_samples]

    train_raw_samples, train_missing_refs = build_raw_samples(train_queries, doc_lookup)
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

    train_dataset, dropped_train = build_llm_only_dataset(
        train_raw_samples,
        doc_lookup,
        tokenizer,
        prompt_mode=args.prompt_mode,
        max_seq_len=args.max_seq_len,
        max_doc_chars=args.max_doc_chars,
        use_chat_template=args.use_chat_template,
    )
    val_dataset, dropped_val = build_llm_only_dataset(
        val_raw_samples,
        doc_lookup,
        tokenizer,
        prompt_mode=args.prompt_mode,
        max_seq_len=args.max_seq_len,
        max_doc_chars=args.max_doc_chars,
        use_chat_template=args.use_chat_template,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "split_metadata.json",
        build_split_metadata(
            seed=args.seed,
            val_ratio=args.val_doc_ratio,
            train_doc_ids=train_doc_ids,
            val_doc_ids=val_doc_ids,
            train_queries=train_queries,
            val_queries=val_queries,
            dropped_train=dropped_train,
            dropped_val=dropped_val,
        ),
    )
    save_json(args.output_dir / "missing_train_refs.json", {"rows": train_missing_refs})
    save_json(args.output_dir / "missing_val_refs.json", {"rows": val_missing_refs})
    save_json(
        args.output_dir / "runtime_paths.json",
        {
            "project_root": str(args.project_root),
            "output_dir": str(args.output_dir),
            "model_name_or_path": model_source,
            "cache_dir": str(args.cache_dir),
            "prompt_mode": args.prompt_mode,
            "max_doc_chars": args.max_doc_chars,
        },
    )

    print(f"Loaded docs={len(docs)} queries={len(queries)}")
    print(f"Train docs={len(train_doc_ids)} val docs={len(val_doc_ids)}")
    print(f"Train samples={len(train_raw_samples)} val samples={len(val_raw_samples)}")
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
        total_training_steps = max(1, int(math.ceil(steps_per_epoch * args.num_train_epochs)))
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
        raise RuntimeError("This transformers build does not expose evaluation_strategy or eval_strategy.")
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
