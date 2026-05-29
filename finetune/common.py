from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from src import config
from src.prompting import (
    ANSWER_PROFILE_FACT,
    ANSWER_PROFILE_LIST,
    ANSWER_PROFILE_SYNTHESIS,
    NO_ANSWER_TEXT,
    NO_CONTEXT_TEXT,
    SYSTEM_PROMPT,
    build_user_prompt,
    context_limit_for_profile,
    detect_answer_profile,
    format_ranked_context,
)
from src.retrieval import (
    build_generation_context,
    rerank_retrieved,
    retrieval_candidate_count,
    select_references_with_diagnostics,
    tokenize_for_overlap,
)


LANTA_PROJECT_ROOT = Path("/project/zz991000-zdeva/zz991011/CAMNET_P")
LANTA_MODEL_ROOT = Path(
    os.environ.get("CAMNET_MODEL_DIR", "/project/zz991000-zdeva/zz991011/models")
).expanduser()
LANTA_CACHE_ROOT = Path("/project/zz991000-zdeva/zz991011/.cache")

DEFAULT_ARTIFACT_NAME = os.environ.get("CAMNET_FINETUNE_ARTIFACT_NAME", "typhoon25_qwen3_4b_rag_qa_qlora")
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "CAMNET_FINETUNE_OUTPUT_DIR",
        str(LANTA_PROJECT_ROOT / "artifacts" / DEFAULT_ARTIFACT_NAME),
    )
).expanduser()
DEFAULT_TRAIN_JSON_PATH = LANTA_PROJECT_ROOT / "data" / "train" / "train_set.json"
DEFAULT_BASE_MODEL_PATH = LANTA_MODEL_ROOT / "typhoon2.5-qwen3-4b"
DEFAULT_EMBED_MODEL_PATH = LANTA_MODEL_ROOT / "bge-m3"
DEFAULT_RERANK_MODEL_PATH = LANTA_MODEL_ROOT / "Qwen3-Reranker-4B"


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


def build_ranked_context_from_paragraphs(
    query: str,
    paragraphs: Sequence[dict[str, Any]],
    *,
    profile: str | None = None,
) -> str:
    profile = profile or detect_answer_profile(query, paragraphs)
    return format_ranked_context(
        paragraphs,
        primary_count=context_limit_for_profile(profile),
    )


def _sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[\.\?!…])\s+|\n+", text or "") if part.strip()]


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(tokenize_for_overlap(left))
    right_tokens = set(tokenize_for_overlap(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def build_source_anchored_fact_answer(
    query: str,
    gold_answer: str,
    paragraphs: Sequence[dict[str, Any]],
) -> str | None:
    best_sentence = None
    best_score = -1.0
    for paragraph in paragraphs:
        for sentence in _sentence_split(paragraph.get("text", "")):
            score = (0.65 * _token_overlap_ratio(gold_answer, sentence)) + (0.35 * _token_overlap_ratio(query, sentence))
            if score > best_score:
                best_score = score
                best_sentence = sentence
    if best_sentence is None or best_score < config.SOURCE_ANCHORED_FACT_MIN_OVERLAP:
        return None
    return best_sentence.strip()


def _query_subject_prefix(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", query).strip().rstrip(" ?")
    suffixes = (
        "คืออะไร",
        "คือใคร",
        "คือที่ใด",
        "คือที่ไหน",
        "คือเมื่อใด",
        "ได้แก่อะไรบ้าง",
        "ได้แก่ใครบ้าง",
        "มีอะไรบ้าง",
        "มีใครบ้าง",
        "อย่างไร",
        "เมื่อใด",
        "ที่ใด",
        "ที่ไหน",
        "หรือไม่",
        "เพราะเหตุใด",
        "เหตุใด",
        "ทำไม",
    )
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)].strip()
    return cleaned


def synthesize_style_variants(sample: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    answer = sample["answer"].strip()
    query = sample["query"].strip()
    if not answer or answer == NO_ANSWER_TEXT:
        return variants

    if re.search(r"(?m)\b1\.", answer):
        normalized_lines = []
        for item in re.split(r"(?=\b\d+\.)", answer.replace("\n", " ")):
            item = item.strip()
            if item:
                normalized_lines.append(item)
        rewritten = "\n".join(normalized_lines)
        if rewritten and rewritten != answer:
            variants.append(
                {
                    **sample,
                    "ID": f"{sample['ID']}::synthetic_list",
                    "answer": rewritten,
                    "mode": "synthetic_style",
                    "profile": ANSWER_PROFILE_LIST,
                }
            )

    if "\n" not in answer and len(answer) > 260:
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        rewritten = "\n".join(part.strip() for part in sentences if part.strip())
        if rewritten and rewritten != answer:
            variants.append(
                {
                    **sample,
                    "ID": f"{sample['ID']}::synthetic_synthesis",
                    "answer": rewritten,
                    "mode": "synthetic_style",
                    "profile": ANSWER_PROFILE_SYNTHESIS,
                }
            )

    subject_prefix = _query_subject_prefix(query)
    if (
        subject_prefix
        and "\n" not in answer
        and len(answer) <= 220
        and not answer.startswith(subject_prefix)
        and not re.search(r"(?m)\b1\.", answer)
    ):
        rewritten = f"{subject_prefix} {answer}".strip()
        if rewritten != answer:
            variants.append(
                {
                    **sample,
                    "ID": f"{sample['ID']}::synthetic_fact",
                    "answer": rewritten,
                    "mode": "synthetic_style",
                    "profile": ANSWER_PROFILE_FACT,
                }
            )

    return variants


def synthesize_no_answer_samples(
    queries: Sequence[dict[str, Any]],
    doc_lookup: dict[str, dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)
    doc_ids = sorted(doc_lookup)
    results: list[dict[str, Any]] = []
    for query in shuffled:
        foreign_doc_ids = [doc_id for doc_id in doc_ids if doc_id != query["doc_id"]]
        if not foreign_doc_ids:
            continue
        doc_id = foreign_doc_ids[rng.randrange(len(foreign_doc_ids))]
        doc = doc_lookup[doc_id]
        paragraphs = doc.get("paragraphs", [])[: config.GENERATOR_CONTEXT_K_FACT]
        context = build_ranked_context_from_paragraphs(
            query["query"].strip(),
            paragraphs,
            profile=ANSWER_PROFILE_FACT,
        )
        results.append(
            {
                "ID": f"{query['ID']}::synthetic_no_answer",
                "doc_id": doc_id,
                "query": query["query"].strip(),
                "answer": NO_ANSWER_TEXT,
                "context": context,
                "gold_refs": [],
                "mode": "synthetic_style",
                "profile": ANSWER_PROFILE_FACT,
            }
        )
        if len(results) >= limit:
            break
    return results


def build_raw_samples(
    queries: Sequence[dict[str, Any]],
    doc_lookup: dict[str, dict[str, Any]],
    *,
    use_source_anchored_fact_targets: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    missing_ref_records: list[dict[str, Any]] = []
    for query in queries:
        doc = doc_lookup.get(query["doc_id"])
        if doc is None:
            raise KeyError(f"doc_id {query['doc_id']} from query {query['ID']} is missing from docs")
        ordered_refs, missing_refs = ordered_ref_paragraphs(doc, query.get("refs", []))
        profile = detect_answer_profile(query["query"], ordered_refs)
        gold_answer = query["abstractive"].strip()
        if use_source_anchored_fact_targets and profile == ANSWER_PROFILE_FACT and ordered_refs:
            source_anchored = build_source_anchored_fact_answer(query["query"], gold_answer, ordered_refs)
            if source_anchored:
                gold_answer = source_anchored
        if missing_refs:
            missing_ref_records.append({"ID": query["ID"], "missing_refs": missing_refs})
        samples.append(
            {
                "ID": query["ID"],
                "doc_id": query["doc_id"],
                "query": query["query"].strip(),
                "answer": gold_answer,
                "context": build_ranked_context_from_paragraphs(
                    query["query"].strip(),
                    ordered_refs,
                    profile=profile,
                )
                if ordered_refs
                else NO_CONTEXT_TEXT,
                "gold_refs": list(query.get("refs", [])),
                "mode": "oracle",
                "profile": profile,
            }
        )
    return samples, missing_ref_records


def build_augmented_training_samples(
    queries: Sequence[dict[str, Any]],
    doc_lookup: dict[str, dict[str, Any]],
    doc_embedding_index: dict[str, dict[str, Any]],
    embedder: Any,
    reranker: Any | None = None,
    ref_selector: Any | None = None,
    generator: Any | None = None,
    *,
    seed: int,
    oracle_fraction: float = 0.85,
    noisy_fraction: float = 0.15,
    synthetic_fraction: float = 0.0,
    use_source_anchored_fact_targets: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    oracle_samples, missing_refs = build_raw_samples(
        queries,
        doc_lookup,
        use_source_anchored_fact_targets=use_source_anchored_fact_targets,
    )
    total_oracle = len(oracle_samples)
    if total_oracle == 0:
        return oracle_samples, missing_refs, {"oracle": 0, "noisy_retrieved": 0, "synthetic_style": 0}

    noisy_target = int(round(total_oracle * (noisy_fraction / max(oracle_fraction, 1e-6))))
    synthetic_target = int(round(total_oracle * (synthetic_fraction / max(oracle_fraction, 1e-6))))

    noisy_candidates: list[dict[str, Any]] = []
    for sample in oracle_samples:
        retrieved = retrieve_paragraphs(
            doc_embedding_index,
            sample["doc_id"],
            sample["query"],
            embedder,
            retrieval_candidate_count(),
        )
        reranked = rerank_retrieved(sample["query"], retrieved, reranker=reranker, rerank_top_k=config.RERANK_TOP_K)
        if not reranked:
            continue
        gold_refs = set(sample["gold_refs"])
        if gold_refs:
            existing = {item["para_id"] for item in reranked}
            doc = doc_lookup[sample["doc_id"]]
            missing_gold = [
                paragraph for paragraph in doc["paragraphs"]
                if paragraph["para_id"] in gold_refs and paragraph["para_id"] not in existing
            ]
            reranked = reranked + [
                {"para_id": paragraph["para_id"], "text": paragraph["text"], "score": -1.0}
                for paragraph in missing_gold
            ]
        profile = detect_answer_profile(sample["query"], reranked)
        selected_refs = select_references_with_diagnostics(
            sample["query"],
            reranked,
            profile=profile,
            mode="dynamic_rules_then_llm_arbiter" if config.ENABLE_LLM_REF_ARBITER else None,
            ref_selector=ref_selector if config.ENABLE_LEARNED_REF_SELECTOR else None,
            generator=generator,
        ).selected_refs
        noisy_candidates.append(
            {
                **sample,
                "ID": f"{sample['ID']}::noisy",
                "context": build_ranked_context_from_paragraphs(
                    sample["query"],
                    build_generation_context(sample["query"], reranked, selected_refs, profile),
                    profile=profile,
                ),
                "mode": "noisy_retrieved",
                "profile": profile,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(noisy_candidates)
    noisy_samples = noisy_candidates[:noisy_target]

    synthetic_candidates: list[dict[str, Any]] = []
    for sample in oracle_samples:
        synthetic_candidates.extend(synthesize_style_variants(sample))
    rng.shuffle(synthetic_candidates)
    synthetic_samples = synthetic_candidates[:synthetic_target]

    max_no_answer = max(1, int(round(synthetic_target * 0.05))) if synthetic_target > 0 else 0
    remaining_synthetic_slots = max(0, synthetic_target - len(synthetic_samples))
    no_answer_limit = min(max_no_answer, remaining_synthetic_slots)
    synthetic_samples.extend(
        synthesize_no_answer_samples(
            queries,
            doc_lookup,
            limit=no_answer_limit,
            seed=seed,
        )
    )

    all_samples = oracle_samples + noisy_samples + synthetic_samples
    counts = {
        "oracle": len(oracle_samples),
        "noisy_retrieved": len(noisy_samples),
        "synthetic_style": len(synthetic_samples),
    }
    return all_samples, missing_refs, counts


def tokenize_supervised_sample(
    sample: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sample["context"], sample["query"], profile=sample["profile"])},
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
