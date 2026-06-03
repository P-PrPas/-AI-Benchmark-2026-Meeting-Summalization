#!/usr/bin/env python3
"""
CAMNET-P: Thai Parliamentary Meeting Summarization
Docker Model Submission Entry Point
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.generator import Generator
from src.candidate_expansion import expanded_retrieve_from_dense
from src.evidence_set import load_evidence_set_selector_if_available
from src.llm_only import (
    build_llm_only_prompt,
    fallback_refs_by_answer_overlap,
    paragraph_ids,
    parse_llm_only_output,
    truncate_paragraphs_by_chars,
)
from src.prompting import detect_answer_profile
from src.ref_selector import load_ref_selector_if_available
from src.reranker import load_reranker_if_available
from src.retrieval import (
    build_generation_context,
    needs_query_refinement,
    rerank_retrieved,
    retrieval_candidate_count,
    rewrite_query_heuristic,
    select_references_with_diagnostics,
    tokenize_for_overlap,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TEST_PATH = config.TEST_PATH
RESULT_DIR = config.OUTPUT_DIR
RESULT_PATH = RESULT_DIR / "submission.csv"
PROGRESS_LIB = config.PROGRESS_LIB
STARTUP_SLEEP_SECONDS = config.STARTUP_SLEEP_SECONDS


def log_runtime_context():
    print("[info] Runtime configuration")
    print(f"      PIPELINE_MODE={config.PIPELINE_MODE}")
    print(f"      TEST_PATH={TEST_PATH}")
    print(f"      RESULT_PATH={RESULT_PATH}")
    print(f"      EMBED_MODEL_PATH={config.EMBED_MODEL_PATH}")
    print(f"      RERANK_MODEL_PATH={config.RERANK_MODEL_PATH}")
    print(f"      LLM_MODEL_PATH={config.LLM_MODEL_PATH}")
    print(f"      RETRIEVAL_CANDIDATE_K={config.RETRIEVAL_CANDIDATE_K}")
    print(f"      EFFECTIVE_RETRIEVAL_CANDIDATE_K={retrieval_candidate_count()}")
    print(f"      USE_RERANKER={config.USE_RERANKER}")
    print(f"      ENABLE_DYNAMIC_REF_SELECTION={config.ENABLE_DYNAMIC_REF_SELECTION}")
    print(f"      ENABLE_LEARNED_REF_SELECTOR={config.ENABLE_LEARNED_REF_SELECTOR}")
    print(f"      ENABLE_LLM_REF_ARBITER={config.ENABLE_LLM_REF_ARBITER}")
    print(f"      REF_ARBITER_TRIGGER_MODE={config.REF_ARBITER_TRIGGER_MODE}")
    print(f"      REF_ARBITER_MAX_CANDIDATES={config.REF_ARBITER_MAX_CANDIDATES}")
    print(f"      ENABLE_QUERY_REFINEMENT={config.ENABLE_QUERY_REFINEMENT}")
    print(f"      ENABLE_EVIDENCE_COMPRESSION={config.ENABLE_EVIDENCE_COMPRESSION}")
    print(f"      ENABLE_FACT_ANSWER_REWRITE={config.ENABLE_FACT_ANSWER_REWRITE}")
    print(f"      ENABLE_EXPANDED_CANDIDATES={config.ENABLE_EXPANDED_CANDIDATES}")
    print(f"      ENABLE_EVIDENCE_SET_SELECTOR={config.ENABLE_EVIDENCE_SET_SELECTOR}")
    print(f"      ENABLE_SEMI_EXTRACTIVE_COMPOSER={config.ENABLE_SEMI_EXTRACTIVE_COMPOSER}")
    print(f"      REF_SELECTOR_MODEL_PATH={config.REF_SELECTOR_MODEL_PATH}")
    print(f"      EVIDENCE_SET_MODEL_PATH={config.EVIDENCE_SET_MODEL_PATH}")
    print(f"      REFERENCE_TOP_N={config.REFERENCE_TOP_N}")
    print(f"      EMBED_BATCH_SIZE={config.EMBED_BATCH_SIZE}")
    print(f"      RERANK_BATCH_SIZE={config.RERANK_BATCH_SIZE}")
    print(f"      RERANK_MAX_LENGTH={config.RERANK_MAX_LENGTH}")
    print(f"      ENABLE_FACT_FEW_SHOT={config.ENABLE_FACT_FEW_SHOT}")
    print(f"      GENERATOR_BATCH_SIZE={config.GENERATOR_BATCH_SIZE}")
    print(f"      LLM_ONLY_PROMPT_MODE={config.LLM_ONLY_PROMPT_MODE}")
    print(f"      LLM_ONLY_MAX_SEQ_LEN={config.LLM_ONLY_MAX_SEQ_LEN}")
    print(f"      LLM_ONLY_MAX_NEW_TOKENS={config.LLM_ONLY_MAX_NEW_TOKENS}")
    print(f"      LLM_ONLY_BATCH_SIZE={config.LLM_ONLY_BATCH_SIZE}")
    print(f"      LLM_ONLY_MAX_DOC_CHARS={config.LLM_ONLY_MAX_DOC_CHARS}")
    print(f"      PROGRESS_UPDATE_EVERY={config.PROGRESS_UPDATE_EVERY}")


def normalize_dataset(data: dict) -> dict:
    docs = data.get("docs") or data.get("documents") or []
    queries = data.get("queries") or data.get("questions") or []
    if not isinstance(docs, list) or not isinstance(queries, list):
        raise ValueError("Dataset must contain list-like docs and queries fields")
    return {"docs": docs, "queries": queries}


def fallback_retrieve(paragraphs: List[Dict], query: str, top_k: int) -> List[Dict]:
    query_tokens = set(tokenize_for_overlap(query))
    scored = []
    for paragraph in paragraphs:
        para_tokens = set(tokenize_for_overlap(paragraph.get("text", "")))
        score = float(len(query_tokens & para_tokens) / max(1, len(query_tokens)))
        scored.append(
            {
                "para_id": paragraph.get("para_id", ""),
                "text": paragraph.get("text", ""),
                "score": score,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def candidate_test_files(path: str) -> List[Path]:
    path = Path(path)
    candidates = [path]
    if path.parent.exists():
        siblings = sorted(path.parent.glob("*.json"))
        for sibling in siblings:
            if sibling not in candidates:
                candidates.append(sibling)
    return candidates


def call_progress(i: int):
    try:
        subprocess.run([str(PROGRESS_LIB), str(i)], check=True)
    except Exception as exc:
        print(f"[WARN] progress signal failed at {i}: {exc}")


def precompute_query_embeddings(embedder, queries: List[Dict]) -> Dict[str, np.ndarray]:
    if embedder is None:
        return {}

    grouped: Dict[str, List[Dict]] = {}
    for query_row in queries:
        grouped.setdefault(query_row["doc_id"], []).append(query_row)

    encoded: Dict[str, np.ndarray] = {}
    for doc_id, doc_queries in grouped.items():
        texts = [row["query"] for row in doc_queries]
        try:
            embeddings = encode(embedder, texts)
        except Exception as exc:
            print(f"      [warn] Failed to batch-encode queries for {doc_id}: {exc}")
            continue
        for query_row, embedding in zip(doc_queries, embeddings):
            encoded[query_row["ID"]] = embedding
    return encoded


def generate_rows_in_batches(generator: Generator, prepared_rows: List[Dict], total: int) -> List[Dict]:
    completed = 0
    ordered_rows = {row["ID"]: row for row in prepared_rows}
    for profile in ("fact", "list", "synthesis"):
        profile_rows = [row for row in prepared_rows if row["profile"] == profile]
        if not profile_rows:
            continue
        print(
            f"      Generating {len(profile_rows)} {profile} answers "
            f"in batches of {config.GENERATOR_BATCH_SIZE}"
        )
        for start in range(0, len(profile_rows), config.GENERATOR_BATCH_SIZE):
            batch = profile_rows[start:start + config.GENERATOR_BATCH_SIZE]
            outputs = generator.batch_generate(
                [row["query"] for row in batch],
                [row["generation_paragraphs"] for row in batch],
                profile=profile,
                max_seq_len=config.GENERATOR_MAX_SEQ_LEN,
            )
            for row, abstractive in zip(batch, outputs):
                ordered_rows[row["ID"]]["abstractive"] = abstractive
                completed += 1
                if completed % max(1, config.PROGRESS_UPDATE_EVERY) == 0 or completed == total:
                    call_progress(completed)
    return [ordered_rows[row["ID"]] for row in prepared_rows]


def load_data(path: str) -> dict:
    print(f"[1/4] Loading test data from {path} ...")
    last_error = None
    for candidate in candidate_test_files(path):
        try:
            print(f"      trying {candidate}")
            with open(candidate, "r", encoding="utf-8") as handle:
                return normalize_dataset(json.load(handle))
        except Exception as exc:
            last_error = exc
            print(f"      [warn] Skipping {candidate}: {exc}")
    raise RuntimeError(f"Unable to load a valid test dataset from {path}: {last_error}")


def build_doc_index(data: dict) -> Dict[str, List[Dict]]:
    return {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}


def load_embedder():
    print(f"[2/4] Loading embedding model from {config.EMBED_MODEL_PATH} ...")
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            str(config.EMBED_MODEL_PATH),
            device=device,
            local_files_only=True,
        )
        print(f"      Embedder loaded on {device}")
        return model
    except Exception as exc:
        print(f"      [warn] Failed to load embedder: {exc}")
        print("      [warn] Falling back to lexical retrieval")
        return None


def encode(model, texts: List[str]) -> np.ndarray:
    if model is None:
        raise RuntimeError("Embedder is unavailable")
    return model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_tensor=False,
        show_progress_bar=False,
    )


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve(
    faiss_index,
    paragraphs: List[Dict],
    query_emb,
    query_text: str,
    top_k: int,
) -> List[Dict]:
    if faiss_index is None or query_emb is None:
        return fallback_retrieve(paragraphs, query_text, top_k)
    scores, indices = faiss_index.search(query_emb.reshape(1, -1).astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(paragraphs):
            results.append(
                {
                    "para_id": paragraphs[idx]["para_id"],
                    "text": paragraphs[idx]["text"],
                    "score": float(score),
                }
            )
    return results


def load_generator() -> Generator:
    print(f"[3/4] Loading LLM from {config.LLM_MODEL_PATH} ...")
    generator = Generator()
    try:
        generator.load_model()
    except Exception as exc:
        print(f"      [warn] Failed to load LLM: {exc}")
        print("      [warn] Falling back to extractive no-answer generation")
    return generator


def load_llm_only_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(config.LLM_ONLY_MODEL_PATH).expanduser() if config.LLM_ONLY_MODEL_PATH else config.LLM_MODEL_PATH
    print(f"[2/4] Loading LLM-only model from {model_path} ...")
    if not model_path.exists():
        available = []
        if model_path.parent.exists():
            available = sorted(path.name for path in model_path.parent.iterdir() if path.is_dir())
        raise FileNotFoundError(
            f"LLM-only model path not found: {model_path}. Available model directories: {available}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    print("      LLM-only model loaded successfully")
    return tokenizer, model


def render_llm_only_prompts(tokenizer, prompts: List[str]) -> List[str]:
    if not config.LLM_ONLY_USE_CHAT_TEMPLATE:
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def batch_generate_llm_only(tokenizer, model, prompts: List[str]) -> List[str]:
    import torch

    outputs: List[str] = []
    batch_size = max(1, config.LLM_ONLY_BATCH_SIZE)
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        rendered = render_llm_only_prompts(tokenizer, batch_prompts)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.LLM_ONLY_MAX_SEQ_LEN,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=config.LLM_ONLY_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=config.LLM_ONLY_REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded.input_ids.shape[1]
        outputs.extend(
            tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            for row in generated
        )
        completed = min(start + len(batch_prompts), len(prompts))
        if completed % max(1, config.PROGRESS_UPDATE_EVERY) == 0 or completed == len(prompts):
            call_progress(completed)
            print(f"      Generated {completed}/{len(prompts)}")
    return outputs


def run_llm_only(data: dict) -> None:
    doc_index = build_doc_index(data)
    queries = data["queries"]
    total = len(queries)
    print(f"      {len(data['docs'])} docs | {total} queries")
    tokenizer, model = load_llm_only_model()

    prepared_rows: List[Dict] = []
    prompts: List[str] = []
    print(f"[3/4] Building full-document prompts for {total} queries ...")
    for query_row in tqdm(queries, desc="Preparing"):
        paragraphs = truncate_paragraphs_by_chars(
            doc_index.get(query_row["doc_id"], []),
            max_doc_chars=config.LLM_ONLY_MAX_DOC_CHARS,
        )
        prompts.append(
            build_llm_only_prompt(
                query_row["query"],
                paragraphs,
                mode=config.LLM_ONLY_PROMPT_MODE,
                max_doc_chars=config.LLM_ONLY_MAX_DOC_CHARS,
            )
        )
        prepared_rows.append(
            {
                "ID": query_row["ID"],
                "paragraphs": paragraphs,
                "valid_refs": paragraph_ids(paragraphs),
            }
        )

    print(f"[4/4] Generating LLM-only answers on {total} queries ...")
    raw_outputs = batch_generate_llm_only(tokenizer, model, prompts)
    rows = []
    parse_errors = 0
    invalid_ref_rows = 0
    empty_ref_rows = 0
    for row, raw_output in zip(prepared_rows, raw_outputs):
        parsed = parse_llm_only_output(raw_output, row["valid_refs"])
        refs = parsed.refs
        if not refs and config.LLM_ONLY_REF_FALLBACK:
            refs = fallback_refs_by_answer_overlap(parsed.abstractive, row["paragraphs"], max_refs=1)
        parse_errors += int(parsed.parse_error)
        invalid_ref_rows += int(bool(parsed.invalid_refs))
        empty_ref_rows += int(not refs)
        rows.append(
            {
                "ID": row["ID"],
                "abstractive": parsed.abstractive,
                "refs": ",".join(refs),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8")
    print(
        "      LLM-only diagnostics: "
        f"parse_error_rate={parse_errors / max(1, total):.4f} "
        f"invalid_ref_rate={invalid_ref_rows / max(1, total):.4f} "
        f"empty_ref_rate={empty_ref_rows / max(1, total):.4f}"
    )
    print(f"\nSaved {len(df)} rows -> {RESULT_PATH}")
    call_progress(total)
    print("Done.")


def load_reranker():
    print(f"[3/4] Loading reranker from {config.RERANK_MODEL_PATH} ...")
    reranker = load_reranker_if_available()
    if reranker is None:
        print("      [warn] Reranker unavailable, falling back to hybrid reranking only")
        return None
    try:
        reranker.load_model()
        print("      Reranker loaded successfully")
        return reranker
    except Exception as exc:
        print(f"      [warn] Failed to load reranker: {exc}")
        print("      [warn] Falling back to hybrid reranking only")
        return None


def load_ref_selector():
    print(f"[3/4] Loading ref selector from {config.REF_SELECTOR_MODEL_PATH} ...")
    selector = load_ref_selector_if_available()
    if selector is None or not config.ENABLE_LEARNED_REF_SELECTOR:
        print("      [info] Learned ref selector disabled")
        return None


def load_evidence_selector():
    print(f"[3/4] Loading evidence-set selector from {config.EVIDENCE_SET_MODEL_PATH} ...")
    selector = load_evidence_set_selector_if_available()
    if selector is None or not config.ENABLE_EVIDENCE_SET_SELECTOR:
        print("      [info] Evidence-set selector disabled")
        return None
    try:
        selector.load_model()
        print("      Evidence-set selector loaded successfully")
        return selector
    except Exception as exc:
        print(f"      [warn] Failed to load evidence-set selector: {exc}")
        return None
    try:
        selector.load_model()
        print("      Learned ref selector loaded successfully")
        return selector
    except Exception as exc:
        print(f"      [warn] Failed to load ref selector: {exc}")
        return None


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_runtime_context()

    if STARTUP_SLEEP_SECONDS > 0:
        print(f"[0/4] Sleeping for {STARTUP_SLEEP_SECONDS} seconds before startup ...")
        time.sleep(STARTUP_SLEEP_SECONDS)

    data = load_data(TEST_PATH)
    if config.PIPELINE_MODE == "llm_only":
        run_llm_only(data)
        return
    if config.PIPELINE_MODE != "rag":
        raise ValueError("CAMNET_PIPELINE_MODE must be either 'rag' or 'llm_only'")
    doc_index = build_doc_index(data)
    queries = data["queries"]
    total = len(queries)
    print(f"      {len(data['docs'])} docs | {total} queries")

    embedder = load_embedder()
    generator = load_generator()
    reranker = load_reranker()
    ref_selector = load_ref_selector()
    evidence_selector = load_evidence_selector()

    print("[3/4] Indexing documents ...")
    faiss_indices = {}
    for doc in tqdm(data["docs"], desc="Indexing"):
        paragraphs = doc["paragraphs"]
        if embedder is None:
            faiss_indices[doc["doc_id"]] = None
            continue
        try:
            embeddings = encode(embedder, [paragraph["text"] for paragraph in paragraphs])
            faiss_indices[doc["doc_id"]] = build_faiss_index(embeddings)
        except Exception as exc:
            print(f"      [warn] Failed to index {doc['doc_id']}: {exc}")
            faiss_indices[doc["doc_id"]] = None

    print("[3/4] Encoding query embeddings ...")
    query_embeddings = precompute_query_embeddings(embedder, queries)

    print(f"[4/4] Preparing inference inputs for {total} queries ...")
    rows = []
    effective_retrieval_top_k = retrieval_candidate_count(use_reranker=reranker is not None)
    dense_retrieval_top_k = max(
        effective_retrieval_top_k,
        config.EXPANDED_DENSE_TOP_K if config.ENABLE_EXPANDED_CANDIDATES else 0,
    )
    for query_row in tqdm(queries, desc="Retrieving"):
        query_id = query_row["ID"]
        doc_id = query_row["doc_id"]
        query = query_row["query"]
        paragraphs = doc_index.get(doc_id, [])

        query_emb = query_embeddings.get(query_id)
        if query_emb is None and embedder is not None:
            try:
                query_emb = encode(embedder, [query])[0]
            except Exception as exc:
                print(f"      [warn] Failed to encode query {query_id}: {exc}")

        dense_retrieved = retrieve(
            faiss_indices.get(doc_id),
            paragraphs,
            query_emb,
            query,
            dense_retrieval_top_k,
        )
        if config.ENABLE_EXPANDED_CANDIDATES:
            dense_retrieved = expanded_retrieve_from_dense(
                query,
                paragraphs,
                dense_retrieved,
                effective_retrieval_top_k,
            )
        initial_profile = detect_answer_profile(query, dense_retrieved)
        reranked = rerank_retrieved(
            query,
            dense_retrieved,
            profile=initial_profile,
            reranker=reranker,
            rerank_top_k=config.RERANK_TOP_K,
        )
        if needs_query_refinement(reranked, initial_profile):
            refined_query = rewrite_query_heuristic(query)
            if refined_query != query:
                refined_dense = retrieve(
                    faiss_indices.get(doc_id),
                    paragraphs,
                    encode(embedder, [refined_query])[0] if embedder is not None else None,
                    refined_query,
                    dense_retrieval_top_k,
                )
                if config.ENABLE_EXPANDED_CANDIDATES:
                    refined_dense = expanded_retrieve_from_dense(
                        refined_query,
                        paragraphs,
                        refined_dense,
                        effective_retrieval_top_k,
                    )
                refined_reranked = rerank_retrieved(
                    refined_query,
                    refined_dense,
                    profile=initial_profile,
                    reranker=reranker,
                    rerank_top_k=config.RERANK_TOP_K,
                )
                if refined_reranked:
                    reranked = refined_reranked
        profile = detect_answer_profile(query, reranked)
        ref_selection = select_references_with_diagnostics(
            query,
            reranked,
            profile=profile,
            mode="dynamic_rules_then_llm_arbiter" if config.ENABLE_LLM_REF_ARBITER else None,
            ref_selector=ref_selector if config.ENABLE_LEARNED_REF_SELECTOR else None,
            evidence_selector=evidence_selector if config.ENABLE_EVIDENCE_SET_SELECTOR else None,
            generator=generator,
        )
        refs = ref_selection.selected_refs
        generation_paragraphs = build_generation_context(query, reranked, refs, profile)

        rows.append(
            {
                "ID": query_id,
                "query": query,
                "profile": profile,
                "generation_paragraphs": generation_paragraphs,
                "refs": refs,
                "abstractive": None,
            }
        )

    print(f"[4/4] Generating answers on {total} queries ...")
    rows = generate_rows_in_batches(generator, rows, total)

    df = pd.DataFrame(
        [
            {
                "ID": row["ID"],
                "abstractive": row["abstractive"],
                "refs": ",".join(row["refs"]),
            }
            for row in rows
        ]
    )
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {RESULT_PATH}")
    call_progress(total)
    print("Done.")


if __name__ == "__main__":
    main()
