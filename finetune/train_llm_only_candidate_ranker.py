from __future__ import annotations

import argparse
import pickle
from typing import Any

from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from src.llm_only_ranker import BASE_FEATURES, extract_llm_only_candidate_features

from .common import (
    DEFAULT_EMBED_MODEL_PATH,
    LANTA_PROJECT_ROOT,
    cache_dir_as_str,
    calculate_iou,
    configure_cache_env,
    load_json,
    resolve_model_source,
    resolve_path,
    save_json,
    tokenize_thai,
)


class ThaiSpaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split(" ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LLM-only candidate ranker from llm_only_candidates.json")
    parser.add_argument("--project-root", default=str(LANTA_PROJECT_ROOT))
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--semantic-model-name-or-path", default=str(DEFAULT_EMBED_MODEL_PATH))
    parser.add_argument("--cache-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-semantic-label", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    project_root = resolve_path(args.project_root)
    assert project_root is not None
    args.project_root = project_root
    args.candidate_path = resolve_path(args.candidate_path, project_root=project_root)
    args.output_path = resolve_path(args.output_path, project_root=project_root)
    args.cache_dir = resolve_path(args.cache_dir, project_root=project_root) if args.cache_dir else None
    return args


def _candidate_rouge(scorer: Any, gold_answer: str, pred_answer: str) -> float:
    return scorer.score(tokenize_thai(gold_answer), tokenize_thai(pred_answer))["rougeL"].fmeasure


def _semantic_scores(row: dict[str, Any], semantic_model: Any | None) -> dict[str, float]:
    if semantic_model is None:
        return {}
    gold_answer = row.get("gold_answer") or ""
    candidates = row.get("candidates") or []
    texts = [gold_answer] + [candidate.get("abstractive", "") for candidate in candidates]
    embeddings = semantic_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    gold_embedding = embeddings[0]
    return {
        candidate.get("variant", f"idx{index}"): float((gold_embedding * embedding).sum())
        for index, (candidate, embedding) in enumerate(zip(candidates, embeddings[1:]))
    }


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    configure_cache_env(args.cache_dir, offline=True)

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    semantic_model = None
    if not args.disable_semantic_label:
        from sentence_transformers import SentenceTransformer

        semantic_source = resolve_model_source(args.semantic_model_name_or_path, project_root=args.project_root)
        semantic_model = SentenceTransformer(
            semantic_source,
            device="cuda",
            cache_folder=cache_dir_as_str(args.cache_dir),
        )

    payload = load_json(args.candidate_path)
    rows = payload.get("rows") or []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=ThaiSpaceTokenizer())
    features: list[list[float]] = []
    labels: list[int] = []
    group_count = 0
    best_variant_counts: dict[str, int] = {}

    for row in rows:
        candidates = row.get("candidates") or []
        gold_answer = row.get("gold_answer") or ""
        gold_refs = row.get("gold_refs") or []
        query = row.get("query") or ""
        if len(candidates) < 2 or not gold_answer:
            continue
        semantic_scores = _semantic_scores(row, semantic_model)
        candidate_scores = []
        for candidate in candidates:
            rouge = _candidate_rouge(scorer, gold_answer, candidate.get("abstractive", ""))
            iou = calculate_iou(",".join(candidate.get("refs") or []), gold_refs)
            if semantic_model is None:
                score = 0.65 * rouge + 0.35 * iou
            else:
                score = (
                    0.35 * rouge
                    + 0.45 * semantic_scores.get(candidate.get("variant", ""), 0.0)
                    + 0.20 * iou
                )
            candidate_scores.append(score)
        best_score = max(candidate_scores)
        best_indices = {index for index, score in enumerate(candidate_scores) if score == best_score}
        if best_score <= 0:
            continue
        group_count += 1
        best_variant = str(candidates[next(iter(best_indices))].get("variant", "unknown"))
        best_variant_counts[best_variant] = best_variant_counts.get(best_variant, 0) + 1
        for index, candidate in enumerate(candidates):
            row_features = extract_llm_only_candidate_features(
                query,
                candidate,
                candidates,
                candidate_index=index,
            )
            features.append([row_features.get(name, 0.0) for name in BASE_FEATURES])
            labels.append(1 if index in best_indices else 0)

    if not features or len(set(labels)) < 2:
        raise ValueError("Not enough labeled LLM-only candidates to train ranker.")

    models = {
        "gradient_boosting": GradientBoostingClassifier(random_state=args.seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.seed,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(features, labels)
        fitted[name] = model
    selected_name = "gradient_boosting"
    selected_model = fitted[selected_name]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_bytes(
        pickle.dumps(
            {
                "model": selected_model,
                "model_name": selected_name,
                "feature_order": BASE_FEATURES,
                "train_rows": len(features),
                "candidate_groups": group_count,
                "positive_rows": int(sum(labels)),
                "best_variant_counts": best_variant_counts,
                "semantic_label": not args.disable_semantic_label,
            }
        )
    )
    metrics = {
        "model_name": selected_name,
        "train_rows": len(features),
        "candidate_groups": group_count,
        "positive_rows": int(sum(labels)),
        "best_variant_counts": best_variant_counts,
        "semantic_label": not args.disable_semantic_label,
    }
    save_json(args.output_path.with_suffix(".metrics.json"), metrics)
    print(f"Saved LLM-only candidate ranker to {args.output_path}")
    print(metrics)


if __name__ == "__main__":
    main()
