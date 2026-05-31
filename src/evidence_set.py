from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import config
from .prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS
from .retrieval import lexical_overlap_score, normalized_score_entropy


FEATURE_ORDER = [
    "rank",
    "score",
    "dense_score",
    "rerank_score",
    "hybrid_score",
    "bm25_score",
    "quote_score",
    "lexical_score",
    "gap_prev",
    "gap_next",
    "entropy",
    "paragraph_len",
    "query_len",
    "neighbor_prev",
    "neighbor_next",
    "profile_fact",
    "profile_list",
    "profile_synthesis",
]


@dataclass(frozen=True)
class EvidenceSetPrediction:
    refs: list[str]
    probabilities: dict[str, float]
    predicted_count: int


def _score(item: dict[str, Any]) -> float:
    return float(item.get("selection_score", item.get("score", 0.0)))


def _paragraph_number(para_id: str) -> int | None:
    digits = "".join(ch for ch in str(para_id) if ch.isdigit())
    return int(digits) if digits else None


def extract_evidence_features(
    query: str,
    retrieved: Sequence[dict[str, Any]],
    index: int,
    profile: str,
) -> dict[str, float]:
    item = retrieved[index]
    scores = [_score(row) for row in retrieved[: min(20, len(retrieved))]]
    prev_score = _score(retrieved[index - 1]) if index > 0 else _score(item)
    next_score = _score(retrieved[index + 1]) if index + 1 < len(retrieved) else _score(item)
    para_num = _paragraph_number(item.get("para_id", ""))
    prev_num = _paragraph_number(retrieved[index - 1].get("para_id", "")) if index > 0 else None
    next_num = _paragraph_number(retrieved[index + 1].get("para_id", "")) if index + 1 < len(retrieved) else None
    return {
        "rank": float(index + 1),
        "score": _score(item),
        "dense_score": float(item.get("dense_score", 0.0)),
        "rerank_score": float(item.get("rerank_score", item.get("score", 0.0))),
        "hybrid_score": float(item.get("hybrid_score", item.get("score", 0.0))),
        "bm25_score": float(item.get("bm25_score", 0.0)),
        "quote_score": float(item.get("quote_score", 0.0)),
        "lexical_score": lexical_overlap_score(query, item.get("text", "")),
        "gap_prev": prev_score - _score(item),
        "gap_next": _score(item) - next_score,
        "entropy": normalized_score_entropy(scores),
        "paragraph_len": float(len(item.get("text", ""))),
        "query_len": float(len((query or "").split())),
        "neighbor_prev": 1.0 if para_num is not None and prev_num is not None and abs(para_num - prev_num) == 1 else 0.0,
        "neighbor_next": 1.0 if para_num is not None and next_num is not None and abs(para_num - next_num) == 1 else 0.0,
        "profile_fact": 1.0 if profile == ANSWER_PROFILE_FACT else 0.0,
        "profile_list": 1.0 if profile == ANSWER_PROFILE_LIST else 0.0,
        "profile_synthesis": 1.0 if profile == ANSWER_PROFILE_SYNTHESIS else 0.0,
    }


def _threshold_for_profile(profile: str) -> float:
    if profile == ANSWER_PROFILE_FACT:
        return config.EVIDENCE_SET_FACT_THRESHOLD
    if profile == ANSWER_PROFILE_LIST:
        return config.EVIDENCE_SET_LIST_THRESHOLD
    return config.EVIDENCE_SET_SYNTH_THRESHOLD


def _max_refs_for_profile(profile: str) -> int:
    if profile == ANSWER_PROFILE_FACT:
        return config.EVIDENCE_SET_MAX_REFS_FACT
    if profile == ANSWER_PROFILE_LIST:
        return config.EVIDENCE_SET_MAX_REFS_LIST
    return config.EVIDENCE_SET_MAX_REFS_SYNTHESIS


class EvidenceSetSelector:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = str(model_path or config.EVIDENCE_SET_MODEL_PATH or "")
        self.membership_model = None
        self.cardinality_model = None

    def load_model(self) -> None:
        if not self.model_path:
            raise ValueError("EVIDENCE_SET_MODEL_PATH is required")
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Evidence-set model path not found: {model_path}")
        payload = pickle.loads(model_path.read_bytes())
        self.membership_model = payload.get("membership_model")
        self.cardinality_model = payload.get("cardinality_model")
        if self.membership_model is None:
            raise ValueError("Evidence-set payload must contain membership_model")

    @property
    def is_loaded(self) -> bool:
        return self.membership_model is not None

    def predict(self, query: str, retrieved: Sequence[dict[str, Any]], profile: str) -> EvidenceSetPrediction:
        if not self.is_loaded:
            raise RuntimeError("EvidenceSetSelector must be loaded before predict()")
        if not retrieved:
            return EvidenceSetPrediction(refs=[], probabilities={}, predicted_count=0)

        rows = []
        for index in range(min(20, len(retrieved))):
            features = extract_evidence_features(query, retrieved, index, profile)
            rows.append([features[name] for name in FEATURE_ORDER])
        probabilities = self.membership_model.predict_proba(rows)[:, 1].tolist()
        threshold = _threshold_for_profile(profile)
        max_refs = _max_refs_for_profile(profile)
        if self.cardinality_model is not None:
            count_features = rows[0]
            predicted_count = int(self.cardinality_model.predict([count_features])[0])
            max_refs = max(1, min(max_refs, predicted_count))
        else:
            predicted_count = max_refs
        ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
        selected_indices = [idx for idx, prob in ranked if prob >= threshold][:max_refs]
        if not selected_indices:
            selected_indices = [ranked[0][0]]
        selected_indices.sort()
        refs = [retrieved[idx]["para_id"] for idx in selected_indices]
        return EvidenceSetPrediction(
            refs=refs,
            probabilities={retrieved[idx]["para_id"]: float(probabilities[idx]) for idx in range(len(probabilities))},
            predicted_count=predicted_count,
        )


def load_evidence_set_selector_if_available(model_path: str | None = None) -> EvidenceSetSelector | None:
    selector_path = model_path or config.EVIDENCE_SET_MODEL_PATH
    if not selector_path:
        return None
    return EvidenceSetSelector(selector_path)
