from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.prompting import NO_CONTEXT_TEXT, SYSTEM_PROMPT, build_user_prompt


LANTA_PROJECT_ROOT = Path("/project/zz991000-zdeva/zz991011/CAMNET_P")
LANTA_MODEL_ROOT = Path("/project/zz991000-zdeva/zz991011/models")
LANTA_CACHE_ROOT = Path("/project/zz991000-zdeva/zz991011/.cache")

DEFAULT_ARTIFACT_NAME = "typhoon25_qwen3_4b_rag_qa_qlora"
DEFAULT_OUTPUT_DIR = LANTA_PROJECT_ROOT / "artifacts" / DEFAULT_ARTIFACT_NAME
DEFAULT_TRAIN_JSON_PATH = LANTA_PROJECT_ROOT / "data" / "train" / "train_set.json"
DEFAULT_BASE_MODEL_PATH = LANTA_MODEL_ROOT / "typhoon2.5-qwen3-4b"
DEFAULT_EMBED_MODEL_PATH = LANTA_MODEL_ROOT / "bge-m3"


def set_global_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def looks_like_local_path(value: str | Path) -> bool:
    if isinstance(value, Path):
        return True
    raw = str(value)
    candidate = Path(raw).expanduser()
    return (
        raw.startswith((".", "/", "~"))
        or re.match(r"^[A-Za-z]:[\\/]", raw) is not None
        or candidate.exists()
    )


def resolve_path(value: str | Path | None, project_root: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = value if isinstance(value, Path) else Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if project_root is None:
        return path.resolve()
    return (project_root / path).resolve()


def resolve_model_source(value: str | Path, project_root: Path | None = None) -> str:
    if isinstance(value, Path) or looks_like_local_path(value):
        resolved = resolve_path(value, project_root=project_root)
        assert resolved is not None
        return str(resolved)
    return str(value)


def maybe_resolve_local_path(value: str | Path, project_root: Path | None = None) -> Path | None:
    if isinstance(value, Path) or looks_like_local_path(value):
        return resolve_path(value, project_root=project_root)
    return None


def ensure_path_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_local_model_exists(
    value: str | Path,
    label: str,
    project_root: Path | None = None,
) -> Path | None:
    resolved = maybe_resolve_local_path(value, project_root=project_root)
    if resolved is not None:
        ensure_path_exists(resolved, label)
    return resolved


def cache_dir_as_str(cache_dir: Path | None) -> str | None:
    return None if cache_dir is None else str(cache_dir)


def configure_cache_env(cache_dir: Path | None, offline: bool = True) -> None:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_CACHE"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_training_data(
    train_json_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = load_json(train_json_path)
    docs = payload.get("docs") or []
    queries = payload.get("queries") or []
    if not docs or not queries:
        raise ValueError("Training JSON must contain non-empty 'docs' and 'queries'.")
    doc_lookup = {doc["doc_id"]: doc for doc in docs}
    return docs, queries, doc_lookup


def grouped_doc_split(
    queries: Sequence[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    doc_ids = sorted({query["doc_id"] for query in queries})
    if len(doc_ids) < 2:
        raise ValueError("Need at least 2 unique doc_id values for grouped validation split.")
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    val_doc_count = max(1, int(round(len(doc_ids) * val_ratio)))
    val_doc_count = min(val_doc_count, len(doc_ids) - 1)
    val_doc_ids = set(doc_ids[:val_doc_count])
    train_doc_ids = set(doc_ids[val_doc_count:])
    train_queries = [query for query in queries if query["doc_id"] in train_doc_ids]
    val_queries = [query for query in queries if query["doc_id"] in val_doc_ids]
    return train_queries, val_queries, train_doc_ids, val_doc_ids


def ordered_ref_paragraphs(
    doc: dict[str, Any],
    refs: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    ref_set = set(refs or [])
    ordered = [paragraph for paragraph in doc["paragraphs"] if paragraph["para_id"] in ref_set]
    found = {paragraph["para_id"] for paragraph in ordered}
    missing = [ref for ref in refs if ref not in found]
    return ordered, missing


def build_raw_samples(
    queries: Sequence[dict[str, Any]],
    doc_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    missing_ref_records: list[dict[str, Any]] = []
    for query in queries:
        doc = doc_lookup.get(query["doc_id"])
        if doc is None:
            raise KeyError(f"doc_id {query['doc_id']} from query {query['ID']} is missing from docs")
        ordered_refs, missing_refs = ordered_ref_paragraphs(doc, query.get("refs", []))
        context_lines = [
            f"[{paragraph['para_id']}] {paragraph['text'].strip()}"
            for paragraph in ordered_refs
            if paragraph.get("text", "").strip()
        ]
        if missing_refs:
            missing_ref_records.append({"ID": query["ID"], "missing_refs": missing_refs})
        samples.append(
            {
                "ID": query["ID"],
                "doc_id": query["doc_id"],
                "query": query["query"].strip(),
                "answer": query["abstractive"].strip(),
                "context": "\n".join(context_lines) if context_lines else NO_CONTEXT_TEXT,
                "gold_refs": list(query.get("refs", [])),
            }
        )
    return samples, missing_ref_records


def tokenize_supervised_sample(
    sample: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sample["context"], sample["query"])},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": sample["answer"]}]

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


def build_tokenized_dataset(
    raw_samples: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
):
    from datasets import Dataset

    encoded_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, str]] = []
    for sample in raw_samples:
        encoded, dropped = tokenize_supervised_sample(sample, tokenizer, max_seq_len)
        if encoded is not None:
            encoded_rows.append(encoded)
        if dropped is not None:
            dropped_rows.append(dropped)
    if not encoded_rows:
        raise ValueError("No usable samples remain after tokenization.")
    return Dataset.from_list(encoded_rows), dropped_rows


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        base_features = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(base_features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            padded = feature["labels"] + ([-100] * (max_len - len(feature["labels"])))
            labels.append(padded)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def build_split_metadata(
    *,
    seed: int,
    val_ratio: float,
    train_doc_ids: Iterable[str],
    val_doc_ids: Iterable[str],
    train_queries: Sequence[dict[str, Any]],
    val_queries: Sequence[dict[str, Any]],
    dropped_train: Sequence[dict[str, str]],
    dropped_val: Sequence[dict[str, str]],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_doc_ids": sorted(train_doc_ids),
        "val_doc_ids": sorted(val_doc_ids),
        "train_query_ids": [query["ID"] for query in train_queries],
        "val_query_ids": [query["ID"] for query in val_queries],
        "dropped_train": list(dropped_train),
        "dropped_val": list(dropped_val),
    }


def filter_queries_by_ids(
    queries: Sequence[dict[str, Any]],
    query_ids: Iterable[str],
) -> list[dict[str, Any]]:
    allowed = set(query_ids)
    return [query for query in queries if query["ID"] in allowed]


def parse_refs(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if value != value:
            return []
    except Exception:
        pass
    text = str(value).strip()
    return [item.strip() for item in text.split(",") if item.strip()] if text else []


def calculate_iou(pred_refs: Any, gold_refs: Any) -> float:
    pred_set = set(parse_refs(pred_refs))
    gold_set = set(parse_refs(gold_refs))
    if not gold_set:
        return 0.0
    return len(pred_set & gold_set) / len(pred_set | gold_set)


def calculate_final_score(metrics: dict[str, float]) -> float:
    return 0.45 * metrics["SS-score"] + 0.35 * metrics["rougeL"] + 0.20 * metrics["IoU"]


def tokenize_thai(text: str) -> str:
    from pythainlp.tokenize import word_tokenize

    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    return " ".join(tokens)


def run_evaluation(sol_df: Any, pred_df: Any, semantic_model: Any):
    import numpy as np
    from rouge_score import rouge_scorer
    from rouge_score.tokenizers import Tokenizer

    if len(sol_df) != len(pred_df):
        raise ValueError(f"Solution has {len(sol_df)} rows, prediction has {len(pred_df)} rows")

    merged = sol_df.merge(pred_df, on="ID", suffixes=("_sol", "_pred"))
    merged["IoU"] = merged.apply(
        lambda row: calculate_iou(row["refs_pred"], row["refs_sol"]),
        axis=1,
    )

    class ThaiSpaceTokenizer(Tokenizer):
        def tokenize(self, text: str) -> list[str]:
            return text.split(" ")

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
        tokenizer=ThaiSpaceTokenizer(),
    )
    sol_tok = merged["abstractive_sol"].apply(tokenize_thai)
    pred_tok = merged["abstractive_pred"].apply(tokenize_thai)
    merged["rougeL"] = [
        scorer.score(gold, pred)["rougeL"].fmeasure
        for gold, pred in zip(sol_tok, pred_tok)
    ]

    texts = merged["abstractive_sol"].tolist() + merged["abstractive_pred"].tolist()
    embeddings = semantic_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    midpoint = len(texts) // 2
    ref_embeddings = embeddings[:midpoint]
    pred_embeddings = embeddings[midpoint:]
    merged["SS-score"] = (ref_embeddings * pred_embeddings).sum(axis=1)

    metrics = merged[["rougeL", "SS-score", "IoU"]].mean().to_dict()
    metrics["score"] = calculate_final_score(metrics)
    return metrics, merged


def build_document_embedding_index(
    docs: Sequence[dict[str, Any]],
    embedder: Any,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for doc in docs:
        paragraphs = doc["paragraphs"]
        paragraph_texts = [paragraph["text"] for paragraph in paragraphs]
        embeddings = embedder.encode(
            paragraph_texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        index[doc["doc_id"]] = {
            "paragraphs": paragraphs,
            "embeddings": embeddings,
        }
    return index


def retrieve_paragraphs(
    doc_embedding_index: dict[str, dict[str, Any]],
    doc_id: str,
    query: str,
    embedder: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    import numpy as np

    payload = doc_embedding_index[doc_id]
    query_embedding = embedder.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = payload["embeddings"] @ query_embedding
    top_indices = np.argsort(scores)[-top_k:][::-1].tolist()
    return [
        {
            "para_id": payload["paragraphs"][idx]["para_id"],
            "text": payload["paragraphs"][idx]["text"],
            "score": float(scores[idx]),
        }
        for idx in top_indices
    ]


def select_top_refs(retrieved: Sequence[dict[str, Any]], top_n: int) -> list[str]:
    return [item["para_id"] for item in retrieved[:top_n]]


def get_model_device(model: Any):
    return next(model.parameters()).device
