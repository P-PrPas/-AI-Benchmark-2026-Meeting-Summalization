from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from . import config
from .prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS
from .retrieval import lexical_overlap_score, normalized_score_entropy


@dataclass(frozen=True)
class SelectorPrediction:
    keep2: bool
    keep3: bool


def _score_at(retrieved: Sequence[Dict], index: int) -> float:
    if index >= len(retrieved):
        return 0.0
    return float(retrieved[index].get("selection_score", retrieved[index].get("score", 0.0)))


def _lexical_at(query: str, retrieved: Sequence[Dict], index: int) -> float:
    if index >= len(retrieved):
        return 0.0
    return lexical_overlap_score(query, retrieved[index].get("text", ""))


def extract_selector_features(
    query: str,
    retrieved: Sequence[Dict],
    profile: str,
) -> Dict[str, float]:
    top_scores = [_score_at(retrieved, index) for index in range(min(4, len(retrieved)))]
    entropy = normalized_score_entropy(top_scores)
    profile_list = 1.0 if profile == ANSWER_PROFILE_LIST else 0.0
    profile_synthesis = 1.0 if profile == ANSWER_PROFILE_SYNTHESIS else 0.0
    profile_fact = 1.0 if profile == ANSWER_PROFILE_FACT else 0.0
    query_len = len((query or "").split())
    return {
        "top1_score": _score_at(retrieved, 0),
        "top2_score": _score_at(retrieved, 1),
        "top3_score": _score_at(retrieved, 2),
        "gap12": _score_at(retrieved, 0) - _score_at(retrieved, 1),
        "gap23": _score_at(retrieved, 1) - _score_at(retrieved, 2),
        "entropy": entropy,
        "lex1": _lexical_at(query, retrieved, 0),
        "lex2": _lexical_at(query, retrieved, 1),
        "lex3": _lexical_at(query, retrieved, 2),
        "query_len": float(query_len),
        "profile_fact": profile_fact,
        "profile_list": profile_list,
        "profile_synthesis": profile_synthesis,
        "contains_list_cue": 1.0 if any(token in query for token in ("มีอะไรบ้าง", "ได้แก่", "รายชื่อ", "หน่วยงาน")) else 0.0,
        "contains_fact_cue": 1.0 if any(token in query for token in ("คือใคร", "คืออะไร", "เมื่อใด", "กี่", "เท่าใด")) else 0.0,
    }


FEATURE_ORDER = [
    "top1_score",
    "top2_score",
    "top3_score",
    "gap12",
    "gap23",
    "entropy",
    "lex1",
    "lex2",
    "lex3",
    "query_len",
    "profile_fact",
    "profile_list",
    "profile_synthesis",
    "contains_list_cue",
    "contains_fact_cue",
]


class LearnedRefSelector:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = str(model_path or config.REF_SELECTOR_MODEL_PATH or "")
        self.keep2_model = None
        self.keep3_model = None

    def load_model(self) -> None:
        if not self.model_path:
            raise ValueError("REF_SELECTOR_MODEL_PATH is required to load learned selector")
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Ref selector model path not found: {model_path}")
        payload = pickle.loads(model_path.read_bytes())
        self.keep2_model = payload.get("keep2_model")
        self.keep3_model = payload.get("keep3_model")
        if self.keep2_model is None or self.keep3_model is None:
            raise ValueError("Ref selector payload must contain keep2_model and keep3_model")

    @property
    def is_loaded(self) -> bool:
        return self.keep2_model is not None and self.keep3_model is not None

    def predict(self, query: str, retrieved: Sequence[Dict], profile: str) -> SelectorPrediction:
        if not self.is_loaded:
            raise RuntimeError("LearnedRefSelector must be loaded before predict()")
        features = extract_selector_features(query, retrieved, profile)
        vector = [[features[name] for name in FEATURE_ORDER]]
        keep2_proba = float(self.keep2_model.predict_proba(vector)[0][1])
        keep3_proba = float(self.keep3_model.predict_proba(vector)[0][1])
        keep2 = keep2_proba >= 0.5
        keep3 = keep3_proba >= 0.5 and profile in {ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS}
        if profile == ANSWER_PROFILE_FACT:
            keep3 = False
        return SelectorPrediction(keep2=keep2, keep3=keep3)


def load_ref_selector_if_available(model_path: str | None = None) -> LearnedRefSelector | None:
    selector_path = model_path or config.REF_SELECTOR_MODEL_PATH
    if not selector_path:
        return None
    selector = LearnedRefSelector(selector_path)
    return selector
