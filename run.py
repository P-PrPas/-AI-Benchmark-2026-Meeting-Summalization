#!/usr/bin/env python3
"""
CAMNET-P: Thai Parliamentary Meeting Summarization
Docker Model Submission Entry Point

Input  : READ  test set from config.TEST_PATH
Output : WRITE submission.csv to config.OUTPUT_DIR
Signal : CALL  progress helper from config.PROGRESS_LIB when available
"""

import os
import sys
import json
import subprocess
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from src import config
from src.prompting import NO_ANSWER_TEXT, NO_CONTEXT_TEXT, SYSTEM_PROMPT as SHARED_SYSTEM_PROMPT, build_user_prompt, sanitize_generated_answer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# ──────────────────────────────────────────────
# Paths are configurable via src.config and Docker env vars.
# ──────────────────────────────────────────────
TEST_PATH = config.TEST_PATH
RESULT_DIR = config.OUTPUT_DIR
RESULT_PATH = RESULT_DIR / "submission.csv"
PROGRESS_LIB = config.PROGRESS_LIB

# ──────────────────────────────────────────────
# Model paths are configurable via src.config and Docker env vars.
# ──────────────────────────────────────────────
BGE_MODEL_PATH = config.BGE_MODEL_PATH
LLM_MODEL_PATH = config.LLM_MODEL_PATH

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
RETRIEVAL_TOP_K  = 10   # paragraphs retrieved for LLM context
REFERENCE_TOP_N  = 3    # paragraphs reported as refs in submission
SCORE_THRESHOLD  = 0.3  # minimum cosine score to count as ref
STARTUP_SLEEP_SECONDS = config.STARTUP_SLEEP_SECONDS


def log_runtime_context():
    print("[info] Runtime configuration")
    print(f"      TEST_PATH={TEST_PATH}")
    print(f"      RESULT_PATH={RESULT_PATH}")
    print(f"      BGE_MODEL_PATH={BGE_MODEL_PATH}")
    print(f"      LLM_MODEL_PATH={LLM_MODEL_PATH}")
    print(f"      PROGRESS_LIB={PROGRESS_LIB}")
    if TEST_PATH.parent.exists():
        try:
            entries = sorted(path.name for path in TEST_PATH.parent.iterdir())
            print(f"      /model/test contents={entries}")
        except Exception as exc:
            print(f"      [warn] Unable to list test directory: {exc}")


def normalize_dataset(data: dict) -> dict:
    docs = data.get("docs") or data.get("documents") or []
    queries = data.get("queries") or data.get("questions") or []
    if not isinstance(docs, list) or not isinstance(queries, list):
        raise ValueError("Dataset must contain list-like docs and queries fields")
    return {"docs": docs, "queries": queries}


def _tokenize_for_overlap(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def fallback_retrieve(paragraphs: List[Dict], query: str, top_k: int) -> List[Dict]:
    query_tokens = set(_tokenize_for_overlap(query))
    scored = []
    for paragraph in paragraphs:
        para_tokens = set(_tokenize_for_overlap(paragraph.get("text", "")))
        score = float(len(query_tokens & para_tokens))
        scored.append({
            "para_id": paragraph.get("para_id", ""),
            "text": paragraph.get("text", ""),
            "score": score,
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def fallback_generate(query: str, paragraphs: List[Dict]) -> str:
    if not paragraphs:
        return "ไม่พบข้อมูลในเอกสาร"
    best = next((p.get("text", "").strip() for p in paragraphs if p.get("text", "").strip()), "")
    return best or "ไม่พบข้อมูลในเอกสาร"


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
    """Signal progress to the benchmark server."""
    try:
        subprocess.run([str(PROGRESS_LIB), str(i)], check=True)
    except Exception as e:
        print(f"[WARN] progress signal failed at {i}: {e}")


def load_data(path: str) -> dict:
    print(f"[1/4] Loading test data from {path} ...")
    last_error = None
    for candidate in candidate_test_files(path):
        try:
            print(f"      trying {candidate}")
            with open(candidate, "r", encoding="utf-8") as f:
                return normalize_dataset(json.load(f))
        except Exception as exc:
            last_error = exc
            print(f"      [warn] Skipping {candidate}: {exc}")
    raise RuntimeError(f"Unable to load a valid test dataset from {path}: {last_error}")


def build_doc_index(data: dict) -> Dict[str, List[Dict]]:
    """Build {doc_id: [paragraph, ...]} index."""
    return {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}


# ──────────────────────────────────────────────
# RETRIEVER  (BGE-M3 + FAISS)
# ──────────────────────────────────────────────
def load_embedder():
    print(f"[2/4] Loading BGE-M3 embedder from {BGE_MODEL_PATH} ...")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            str(BGE_MODEL_PATH),
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
        batch_size=32,
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
    query_emb: Optional[np.ndarray],
    query_text: str,
    top_k: int,
) -> List[Dict]:
    if faiss_index is None or query_emb is None:
        return fallback_retrieve(paragraphs, query_text, top_k)
    scores, indices = faiss_index.search(
        query_emb.reshape(1, -1).astype(np.float32), top_k
    )
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(paragraphs):
            results.append({
                "para_id": paragraphs[idx]["para_id"],
                "text":    paragraphs[idx]["text"],
                "score":   float(score),
            })
    return results


def select_refs(retrieved: List[Dict], n: int, threshold: float) -> List[str]:
    """Select top-n refs; fall back to top-n by rank if too few pass threshold."""
    above = [r["para_id"] for r in retrieved if r["score"] >= threshold]
    if len(above) >= n:
        return above[:n]
    # fill with next-best
    below = [r["para_id"] for r in retrieved if r["score"] < threshold]
    combined = above + below
    return combined[:n]


# ──────────────────────────────────────────────
# GENERATOR  (LLM)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยสรุปการประชุมรัฐสภาไทย "
    "ให้ตอบคำถามอย่างกระชับและถูกต้องตามเอกสารที่ให้มา "
    "ตอบเป็นประโยคสมบูรณ์ภาษาไทย ไม่ต้องมีคำนำหรือสรุปท้าย"
)

FEW_SHOT = """ตัวอย่าง:
เอกสาร:
[P5] ณ ห้องประชุมกรรมาธิการ N 404 ชั้น 4 อาคารรัฐสภา
คำถาม: การประชุมจัดที่สถานที่ใด
ตอบ: การประชุมจัดขึ้น ณ ห้องประชุมกรรมาธิการ N 404 ชั้น 4 อาคารรัฐสภา

"""

USER_TEMPLATE = (
    FEW_SHOT
    + "เอกสาร:\n{context}\n\nคำถาม: {query}\nตอบ: "
)
SYSTEM_PROMPT = SHARED_SYSTEM_PROMPT
USER_TEMPLATE = None


def load_generator():
    print(f"[3/4] Loading LLM from {LLM_MODEL_PATH} ...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not LLM_MODEL_PATH.exists():
            available = []
            if LLM_MODEL_PATH.parent.exists():
                available = sorted(
                    path.name for path in LLM_MODEL_PATH.parent.iterdir() if path.is_dir()
                )
            raise FileNotFoundError(
                f"LLM model path not found: {LLM_MODEL_PATH}. "
                f"Available model directories: {available}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            str(LLM_MODEL_PATH), trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(LLM_MODEL_PATH),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("      LLM loaded successfully")
        return tokenizer, model
    except Exception as exc:
        print(f"      [warn] Failed to load LLM: {exc}")
        print("      [warn] Falling back to extractive generation")
        return None, None


def fallback_generate(query: str, paragraphs: List[Dict]) -> str:
    if not paragraphs:
        return NO_ANSWER_TEXT
    best = next((p.get("text", "").strip() for p in paragraphs if p.get("text", "").strip()), "")
    return sanitize_generated_answer(best or NO_ANSWER_TEXT)


def generate(tokenizer, model, query: str, paragraphs: List[Dict]) -> str:
    if tokenizer is None or model is None:
        return fallback_generate(query, paragraphs)
    import torch

    context = "\n".join(f"[{p['para_id']}] {p['text']}" for p in paragraphs)
    prompt = build_user_prompt(context or NO_CONTEXT_TEXT, query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return sanitize_generated_answer(response)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_runtime_context()

    if STARTUP_SLEEP_SECONDS > 0:
        print(f"[0/4] Sleeping for {STARTUP_SLEEP_SECONDS} seconds before startup ...")
        time.sleep(STARTUP_SLEEP_SECONDS)

    # 1. Load data
    data       = load_data(TEST_PATH)
    doc_index  = build_doc_index(data)
    queries    = data["queries"]
    total      = len(queries)
    print(f"      {len(data['docs'])} docs | {total} queries")

    # 2. Load models
    embedder              = load_embedder()
    tokenizer, llm_model  = load_generator()

    # 3. Pre-encode all documents (per doc_id, once)
    print("[3/4] Indexing documents ...")
    faiss_indices = {}   # doc_id -> faiss index
    for doc in tqdm(data["docs"], desc="Indexing"):
        paras = doc["paragraphs"]
        if embedder is None:
            faiss_indices[doc["doc_id"]] = None
            continue
        try:
            texts = [p["text"] for p in paras]
            embs  = encode(embedder, texts)
            faiss_indices[doc["doc_id"]] = build_faiss_index(embs)
        except Exception as exc:
            print(f"      [warn] Failed to index {doc['doc_id']}: {exc}")
            faiss_indices[doc["doc_id"]] = None

    # 4. Inference
    print(f"[4/4] Running inference on {total} queries ...")
    rows = []
    for i, q in enumerate(tqdm(queries, desc="Predicting"), start=1):
        qid    = q["ID"]
        doc_id = q["doc_id"]
        query  = q["query"]

        paragraphs = doc_index.get(doc_id, [])

        # --- Retrieve ---
        query_emb = None
        if embedder is not None:
            try:
                query_emb = encode(embedder, [query])[0]
            except Exception as exc:
                print(f"      [warn] Failed to encode query {qid}: {exc}")
        retrieved = retrieve(
            faiss_indices.get(doc_id), paragraphs, query_emb, query, RETRIEVAL_TOP_K
        )
        refs = select_refs(retrieved, REFERENCE_TOP_N, SCORE_THRESHOLD)

        # --- Generate ---
        try:
            abstractive = generate(tokenizer, llm_model, query, retrieved[:RETRIEVAL_TOP_K])
        except Exception as exc:
            print(f"      [warn] Failed to generate answer for {qid}: {exc}")
            abstractive = fallback_generate(query, retrieved[:RETRIEVAL_TOP_K])

        rows.append({
            "ID":          qid,
            "abstractive": abstractive,
            "refs":        ",".join(refs),
        })

        # Signal progress every query (optional but good practice)
        call_progress(i)

    # 5. Save submission
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(df)} rows → {RESULT_PATH}")

    # 6. Final required progress signal
    call_progress(total)
    print("✅ Done.")


if __name__ == "__main__":
    main()
