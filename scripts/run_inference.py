#!/usr/bin/env python3
"""
Run inference on test set and generate submission.csv
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import config
from src.data_loader import load_dataset
from src.embedder import Embedder, FAISSRetriever
from src.retrieval import hybrid_rerank, select_references_from_retrieved
from src.generator import Generator
import pandas as pd


def predict_single(retriever, query, doc_id, generator=None, retrieval_top_k=10, reference_top_n=3):
    """Predict for a single query."""
    retrieved = hybrid_rerank(query, retriever.retrieve(doc_id, query, top_k=retrieval_top_k))

    refs = select_references_from_retrieved(retrieved, n=reference_top_n)

    abstractive = generator.generate(query, retrieved) if generator else ""

    return {
        "refs": refs,
        "retrieved": retrieved,
        "abstractive": abstractive,
    }


def run_inference(
    embedder: Embedder,
    retriever: FAISSRetriever,
    data_path: str,
    generator: Generator = None,
    retrieval_top_k: int = 10,
    reference_top_n: int = 3,
    show_progress: bool = True
):
    """Run inference on dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    docs = data["docs"]

    print(f"Indexing {len(docs)} documents...")
    for doc in tqdm(docs, desc="Indexing") if show_progress else docs:
        retriever.index_document(doc["doc_id"], doc["paragraphs"])

    print(f"\nRunning inference on {len(queries)} queries...")
    predictions = []

    iterator = tqdm(queries, desc="Predicting") if show_progress else queries

    for q in iterator:
        result = predict_single(
            retriever,
            q["query"],
            q["doc_id"],
            generator=generator,
            retrieval_top_k=retrieval_top_k,
            reference_top_n=reference_top_n
        )

        predictions.append({
            "ID": q["ID"],
            "doc_id": q["doc_id"],
            "query": q["query"],
            "abstractive": result["abstractive"],
            "refs": result["refs"],
            "retrieved": result["retrieved"]
        })

    return predictions


def create_submission(predictions, output_path=None):
    """Create submission CSV from predictions."""
    rows = []
    for p in predictions:
        refs_str = ",".join(p["refs"]) if p["refs"] else ""
        rows.append({
            "ID": p["ID"],
            "abstractive": p.get("abstractive", ""),
            "refs": refs_str
        })

    df = pd.DataFrame(rows)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved submission to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--output", "-o", default=os.path.join(config.OUTPUT_DIR, "submission.csv"),
                        help="Output path for submission.csv")
    parser.add_argument("--retrieval-top-k", type=int, default=config.RETRIEVAL_CANDIDATE_K,
                        help="Number of paragraphs to retrieve")
    parser.add_argument("--reference-top-n", type=int, default=config.REFERENCE_TOP_N,
                        help="Number of references to select")
    parser.add_argument("--data", choices=["train", "test"], default="test",
                        help="Dataset to run inference on")

    args = parser.parse_args()

    print("=" * 60)
    print("CAMNET-P Inference")
    print("=" * 60)

    # Initialize components
    print("\n[1/4] Loading embedder...")
    embedder = Embedder()

    print(f"\n[2/4] Loading generator ({config.LLM_MODEL_PATH.name})...")
    generator = Generator()
    generator.load_model()

    print("\n[3/4] Initializing retriever...")
    retriever = FAISSRetriever(embedder)

    # Load and index data
    print("\n[4/4] Loading and indexing data...")
    data_path = config.TRAIN_PATH if args.data == "train" else config.TEST_PATH

    predictions = run_inference(
        embedder,
        retriever,
        data_path,
        generator=generator,
        retrieval_top_k=args.retrieval_top_k,
        reference_top_n=args.reference_top_n
    )

    # Create submission
    print("\n[4/4] Creating submission...")
    submission = create_submission(predictions, args.output)

    print("\n" + "=" * 60)
    print(f"Done! Predicted {len(predictions)} queries")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Show sample predictions
    print("\nSample predictions:")
    for p in predictions[:3]:
        print(f"  {p['ID']}: refs={p['refs']}")
        if p['retrieved']:
            print(f"    Top retrieval: {p['retrieved'][0]['para_id']} - {p['retrieved'][0]['text'][:50]}...")


if __name__ == "__main__":
    main()
