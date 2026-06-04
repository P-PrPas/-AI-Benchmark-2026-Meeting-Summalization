from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import config
from .llm_only import consensus_candidate_score, lexical_f1, ref_iou, token_set


BASE_FEATURES = [
    "candidate_index",
    "answer_len",
    "line_count",
    "ref_count",
    "parse_error",
    "invalid_ref_count",
    "query_overlap",
    "consensus_score",
    "answer_consensus",
    "ref_consensus",
    "is_base",
    "is_sampled",
    "is_max_variant",
    "is_rp_variant",
]


@dataclass(frozen=True)
class LLMOnlyRankerPrediction:
    variant: str
    score: float
    candidate: dict[str, Any]


def _variant_flags(variant: str) -> dict[str, float]:
    name = (variant or "").lower()
    return {
        "is_base": 1.0 if name == "base" else 0.0,
        "is_sampled": 1.0 if name.startswith("temp") or "temp=" in name or ":t=" in name else 0.0,
        "is_max_variant": 1.0 if name.startswith("max") or "tokens=" in name else 0.0,
        "is_rp_variant": 1.0 if name.startswith("rp") or "rp=" in name else 0.0,
    }


def extract_llm_only_candidate_features(
    query: str,
    candidate: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    *,
    candidate_index: int,
) -> dict[str, float]:
    answer = str(candidate.get("abstractive", ""))
    refs = candidate.get("refs") or []
    other_candidates = [item for item in candidates if item is not candidate]
    answer_consensus = 0.0
    ref_consensus = 0.0
    if other_candidates:
        answer_consensus = sum(
            lexical_f1(answer, str(other.get("abstractive", "")))
            for other in other_candidates
        ) / len(other_candidates)
        ref_consensus = sum(ref_iou(refs, other.get("refs") or []) for other in other_candidates) / len(other_candidates)
    query_tokens = token_set(query)
    answer_tokens = token_set(answer)
    query_overlap = len(query_tokens & answer_tokens) / max(1, len(query_tokens))
    features = {
        "candidate_index": float(candidate_index),
        "answer_len": float(len(answer)),
        "line_count": float(max(1, answer.count("\n") + 1)),
        "ref_count": float(len(refs)),
        "parse_error": 1.0 if candidate.get("parse_error") else 0.0,
        "invalid_ref_count": float(len(candidate.get("invalid_refs") or [])),
        "query_overlap": query_overlap,
        "consensus_score": float(candidate.get("consensus_score", consensus_candidate_score(candidate, candidates))),
        "answer_consensus": answer_consensus,
        "ref_consensus": ref_consensus,
    }
    features.update(_variant_flags(str(candidate.get("variant", ""))))
    return features


class LLMOnlyCandidateRanker:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = str(model_path or config.LLM_ONLY_CANDIDATE_RANKER_PATH or "")
        self.model = None
        self.feature_order = BASE_FEATURES

    def load_model(self) -> None:
        if not self.model_path:
            raise ValueError("LLM-only candidate ranker path is required")
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"LLM-only candidate ranker not found: {model_path}")
        payload = pickle.loads(model_path.read_bytes())
        self.model = payload.get("model")
        self.feature_order = payload.get("feature_order") or BASE_FEATURES
        if self.model is None:
            raise ValueError("Ranker payload must contain model")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def select(self, query: str, candidates: Sequence[dict[str, Any]]) -> LLMOnlyRankerPrediction:
        if not candidates:
            return LLMOnlyRankerPrediction(variant="empty", score=0.0, candidate={})
        if not self.is_loaded:
            self.load_model()
        rows = []
        for index, candidate in enumerate(candidates):
            features = extract_llm_only_candidate_features(query, candidate, candidates, candidate_index=index)
            rows.append([features.get(name, 0.0) for name in self.feature_order])
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(rows)[:, 1].tolist()
        else:
            scores = [float(value) for value in self.model.predict(rows).tolist()]
        best_index = max(range(len(candidates)), key=lambda index: scores[index])
        best = candidates[best_index]
        return LLMOnlyRankerPrediction(
            variant=str(best.get("variant", "unknown")),
            score=float(scores[best_index]),
            candidate=best,
        )


def load_llm_only_candidate_ranker_if_available(model_path: str | None = None) -> LLMOnlyCandidateRanker | None:
    path = model_path or config.LLM_ONLY_CANDIDATE_RANKER_PATH
    if not path:
        return None
    return LLMOnlyCandidateRanker(path)
