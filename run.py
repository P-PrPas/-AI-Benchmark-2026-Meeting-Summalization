#!/usr/bin/env python3
"""
CAMNET-P: Thai Parliamentary Meeting Summarization
Docker Model Submission Entry Point

Input  : READ  /model/test/test_set.json
Output : WRITE /result/submission.csv
Signal : CALL  /benchmark_lib/progress <n> (required at end)
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

# ──────────────────────────────────────────────
# PATHS  (DO NOT CHANGE — mounted by server)
# ──────────────────────────────────────────────
TEST_PATH     = "/model/test/test_set.json"
RESULT_DIR    = "/result"
RESULT_PATH   = os.path.join(RESULT_DIR, "submission.csv")
PROGRESS_LIB  = "/benchmark_lib/progress"

# ──────────────────────────────────────────────
# MODEL PATHS  (baked into Docker image)
# ──────────────────────────────────────────────
BGE_MODEL_PATH = "/model/weights/bge-m3"
LLM_MODEL_PATH = "/model/weights/Qwen3.6-27B-unsloth"

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
RETRIEVAL_TOP_K  = 10   # paragraphs retrieved for LLM context
REFERENCE_TOP_N  = 3    # paragraphs reported as refs in submission
SCORE_THRESHOLD  = 0.3  # minimum cosine score to count as ref


def call_progress(i: int):
    """Signal progress to the benchmark server."""
    try:
        subprocess.run([PROGRESS_LIB, str(i)], check=True)
    except Exception as e:
        print(f"[WARN] progress signal failed at {i}: {e}")


def load_data(path: str) -> dict:
    print(f"[1/4] Loading test data from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_doc_index(data: dict) -> Dict[str, List[Dict]]:
    """Build {doc_id: [paragraph, ...]} index."""
    return {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}


# ──────────────────────────────────────────────
# RETRIEVER  (BGE-M3 + FAISS)
# ──────────────────────────────────────────────
def load_embedder():
    print(f"[2/4] Loading BGE-M3 embedder from {BGE_MODEL_PATH} ...")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(BGE_MODEL_PATH, device=device)
    print(f"      Embedder loaded on {device}")
    return model


def encode(model, texts: List[str]) -> np.ndarray:
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
    query_emb: np.ndarray,
    top_k: int,
) -> List[Dict]:
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


def load_generator():
    print(f"[3/4] Loading LLM from {LLM_MODEL_PATH} ...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_PATH, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print("      LLM loaded successfully")
    return tokenizer, model


def generate(tokenizer, model, query: str, paragraphs: List[Dict]) -> str:
    import torch

    context = "\n".join(f"[{p['para_id']}] {p['text']}" for p in paragraphs)
    prompt  = USER_TEMPLATE.format(context=context, query=query)

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
    return response.strip()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

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
        texts = [p["text"] for p in paras]
        embs  = encode(embedder, texts)
        faiss_indices[doc["doc_id"]] = build_faiss_index(embs)

    # 4. Inference
    print(f"[4/4] Running inference on {total} queries ...")
    rows = []
    for i, q in enumerate(tqdm(queries, desc="Predicting"), start=1):
        qid    = q["ID"]
        doc_id = q["doc_id"]
        query  = q["query"]

        paragraphs = doc_index.get(doc_id, [])

        # --- Retrieve ---
        query_emb = encode(embedder, [query])[0]
        retrieved = retrieve(
            faiss_indices[doc_id], paragraphs, query_emb, RETRIEVAL_TOP_K
        )
        refs = select_refs(retrieved, REFERENCE_TOP_N, SCORE_THRESHOLD)

        # --- Generate ---
        abstractive = generate(tokenizer, llm_model, query, retrieved[:RETRIEVAL_TOP_K])

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