from __future__ import annotations

import argparse
import pickle

from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from src.answer_ranker import FEATURE_ORDER, extract_answer_features
from .common import LANTA_PROJECT_ROOT, load_json, resolve_path, save_json, tokenize_thai


class ThaiSpaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split(" ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train lightweight answer ranker from answer_candidates.json")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--answer-candidates-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    project_root = resolve_path(args.project_root)
    assert project_root is not None
    args.project_root = project_root
    args.answer_candidates_path = resolve_path(args.answer_candidates_path, project_root=project_root)
    args.output_path = resolve_path(args.output_path, project_root=project_root)
    return args


def _candidate_rouge(scorer, gold_answer: str, answer: str) -> float:
    return scorer.score(tokenize_thai(gold_answer), tokenize_thai(answer))["rougeL"].fmeasure


def main() -> None:
    args = normalize_args(build_parser().parse_args())

    from sklearn.ensemble import GradientBoostingClassifier

    payload = load_json(args.answer_candidates_path)
    rows = payload.get("rows") or []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=ThaiSpaceTokenizer())
    features = []
    labels = []
    kept_groups = 0
    for row in rows:
        candidates = row.get("candidates") or []
        gold_answer = row.get("gold_answer") or ""
        query = row.get("query") or ""
        evidence = row.get("evidence") or []
        profile = row.get("profile") or "fact"
        if len(candidates) < 2 or not gold_answer:
            continue
        scores = [_candidate_rouge(scorer, gold_answer, candidate.get("answer", "")) for candidate in candidates]
        best_score = max(scores)
        if best_score <= 0:
            continue
        kept_groups += 1
        for candidate, score in zip(candidates, scores):
            row_features = extract_answer_features(
                query,
                candidate.get("answer", ""),
                evidence,
                profile,
                candidate.get("variant", "base"),
            )
            features.append([row_features[name] for name in FEATURE_ORDER])
            labels.append(1 if score == best_score else 0)

    if not features or len(set(labels)) < 2:
        raise ValueError("Not enough labeled answer candidates to train ranker.")

    model = GradientBoostingClassifier(random_state=args.seed)
    model.fit(features, labels)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_bytes(
        pickle.dumps(
            {
                "model": model,
                "feature_order": FEATURE_ORDER,
                "train_rows": len(features),
                "candidate_groups": kept_groups,
                "positive_rows": int(sum(labels)),
            }
        )
    )
    metrics = {
        "train_rows": len(features),
        "candidate_groups": kept_groups,
        "positive_rows": int(sum(labels)),
    }
    save_json(args.output_path.with_suffix(".metrics.json"), metrics)
    print(f"Saved answer ranker to {args.output_path}")
    print(metrics)


if __name__ == "__main__":
    main()
