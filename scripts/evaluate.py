#!/usr/bin/env python3
"""
Evaluation script for CAMNET-P summarization.
Wraps the evaluation from eval_sample/eval.py.
"""

import os
import sys
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config
from src.data_loader import load_dataset


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV with refs parsing."""
    df = pd.read_csv(file_path)

    def parse_para(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [i.strip() for i in str(x).split(",")]

    if 'refs' in df.columns:
        df['refs'] = df['refs'].apply(parse_para)

    return df


def calculate_iou(list_pred, list_sol):
    """Calculate IoU between two lists."""
    set_pred = set(list_pred) if isinstance(list_pred, list) else set()
    set_sol = set(list_sol) if isinstance(list_sol, list) else set()
    if not set_sol:
        return 0.0
    return len(set_pred.intersection(set_sol)) / len(set_pred.union(set_sol))


def run_evaluation(sol_path: str, pred_path: str) -> dict:
    """Run full evaluation on solution vs prediction files.

    Args:
        sol_path: Path to ground truth CSV
        pred_path: Path to prediction CSV

    Returns:
        Dictionary with metrics
    """
    from pythainlp.tokenize import word_tokenize
    from rouge_score import rouge_scorer, tokenizers
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    sol = load_csv(sol_path)
    pred = load_csv(pred_path)

    if len(sol) != len(pred):
        raise ValueError(f"Solution has {len(sol)} rows, prediction has {len(pred)} rows")

    df = pd.merge(sol, pred, on='ID', suffixes=('_sol', '_pred'))

    # IoU
    df['IoU'] = df.apply(
        lambda x: calculate_iou(x['refs_pred'], x['refs_sol']),
        axis=1
    )

    # RougeL with Thai tokenizer
    class ThaiSpaceTokenizer(tokenizers.Tokenizer):
        def tokenize(self, text):
            return text.split(" ")

    def tokenize_thai(text):
        if not isinstance(text, str) or text.strip() == "":
            return ""
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        return " ".join(tokens)

    scorer = rouge_scorer.RougeScorer(
        ['rougeL'],
        use_stemmer=False,
        tokenizer=ThaiSpaceTokenizer()
    )

    sol_toks = df['abstractive_sol'].apply(tokenize_thai)
    pred_toks = df['abstractive_pred'].apply(tokenize_thai)

    results = [scorer.score(g, p) for g, p in zip(sol_toks, pred_toks)]
    df['rougeL'] = [r['rougeL'].fmeasure for r in results]

    # SS-score
    print("Computing SS-score with BGE-M3...")
    model = SentenceTransformer(str(config.BGE_MODEL_PATH), device="cuda" if torch.cuda.is_available() else "cpu")
    texts = df['abstractive_sol'].tolist() + df['abstractive_pred'].tolist()

    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    ref_emb = embeddings[0:len(texts)//2]
    pred_emb = embeddings[len(texts)//2:]

    scores = F.cosine_similarity(pred_emb, ref_emb, dim=1)
    df['SS-score'] = scores.cpu().numpy()

    # Summary
    metric_cols = ["rougeL", "SS-score", "IoU"]
    metrics = df[metric_cols].mean().to_dict()

    # Final weighted score
    wss, wrl, wj = 0.45, 0.35, 0.2
    metrics['final_score'] = wss * metrics['SS-score'] + wrl * metrics['rougeL'] + wj * metrics['IoU']

    return metrics


def evaluate_train_baseline():
    """Evaluate using train set as both solution and prediction (for format check)."""
    print("Running train baseline evaluation...")

    train_data = load_dataset("train")
    queries = train_data["queries"]

    # Create submission from ground truth
    rows = []
    for q in queries:
        rows.append({
            "ID": q["ID"],
            "abstractive": q["abstractive"],
            "refs": ",".join(q["refs"]) if isinstance(q["refs"], list) else str(q["refs"])
        })
    df = pd.DataFrame(rows)

    # Save as both solution and prediction
    sol_path = os.path.join(config.OUTPUT_DIR, "train_solution.csv")
    pred_path = os.path.join(config.OUTPUT_DIR, "train_prediction.csv")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df.to_csv(sol_path, index=False, encoding="utf-8")
    df.to_csv(pred_path, index=False, encoding="utf-8")

    # Evaluate
    metrics = run_evaluation(sol_path, pred_path)

    print("\n" + "=" * 50)
    print("Train Baseline Metrics (should be ~1.0 for all)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAMNET-P predictions")
    parser.add_argument("--solution", "-s", help="Ground truth CSV path")
    parser.add_argument("--prediction", "-p", help="Prediction CSV path")
    parser.add_argument("--train-baseline", action="store_true", help="Run train baseline")

    args = parser.parse_args()

    if args.train_baseline:
        metrics = evaluate_train_baseline()
    elif args.solution and args.prediction:
        metrics = run_evaluation(args.solution, args.prediction)
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Please specify --solution and --prediction, or --train-baseline")
        sys.exit(1)


if __name__ == "__main__":
    main()
