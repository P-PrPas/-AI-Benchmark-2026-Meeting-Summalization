from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import config
from .answer_candidates import heuristic_answer_score
from .prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS, sanitize_generated_answer
from .retrieval import lexical_overlap_score, tokenize_for_overlap


FEATURE_ORDER = [
    "answer_len",
    "line_count",
    "source_overlap",
    "query_overlap",
    "source_similarity",
    "heuristic_score",
    "has_numbered_list",
    "has_preamble",
    "profile_fact",
    "profile_list",
    "profile_synthesis",
    "variant_base",
    "variant_concise",
    "variant_gold_style",
    "variant_list_style",
    "variant_source_faithful",
    "variant_fact_entity_first",
    "variant_fact_gold_phrase",
    "variant_list_gold_format",
    "variant_synthesis_short_gold",
    "variant_no_preamble",
]


@dataclass(frozen=True)
class AnswerRankerPrediction:
    variant: str
    answer: str
    score: float


def _source_overlap(answer: str, paragraphs: Sequence[dict[str, Any]]) -> float:
    answer_tokens = set(tokenize_for_overlap(answer))
    if not answer_tokens:
        return 0.0
    source_tokens = set()
    for paragraph in paragraphs:
        source_tokens.update(tokenize_for_overlap(str(paragraph.get("text", ""))))
    return len(answer_tokens & source_tokens) / max(1, len(answer_tokens))


def extract_answer_features(
    query: str,
    answer: str,
    paragraphs: Sequence[dict[str, Any]],
    profile: str,
    variant: str,
) -> dict[str, float]:
    answer = sanitize_generated_answer(answer)
    source_text = " ".join(str(paragraph.get("text", "")) for paragraph in paragraphs)
    has_numbered = 1.0 if re.search(r"(^|\n)\s*\d+[\.\)]", answer) else 0.0
    has_preamble = 1.0 if re.match(r"^\s*(จาก|ตาม|ใน)(เอกสาร|ข้อมูล|ที่ประชุม)", answer) else 0.0
    features = {
        "answer_len": float(len(answer)),
        "line_count": float(max(1, answer.count("\n") + 1)),
        "source_overlap": _source_overlap(answer, paragraphs),
        "query_overlap": lexical_overlap_score(query, answer),
        "source_similarity": lexical_overlap_score(answer, source_text),
        "heuristic_score": heuristic_answer_score(query, answer, paragraphs, profile),
        "has_numbered_list": has_numbered,
        "has_preamble": has_preamble,
        "profile_fact": 1.0 if profile == ANSWER_PROFILE_FACT else 0.0,
        "profile_list": 1.0 if profile == ANSWER_PROFILE_LIST else 0.0,
        "profile_synthesis": 1.0 if profile == ANSWER_PROFILE_SYNTHESIS else 0.0,
    }
    for name in FEATURE_ORDER:
        if name.startswith("variant_"):
            features[name] = 1.0 if name == f"variant_{variant}" else 0.0
    return features


class AnswerRanker:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = str(model_path or config.ANSWER_RANKER_MODEL_PATH or "")
        self.model = None

    def load_model(self) -> None:
        if not self.model_path:
            raise ValueError("ANSWER_RANKER_MODEL_PATH is required")
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Answer ranker model path not found: {model_path}")
        payload = pickle.loads(model_path.read_bytes())
        self.model = payload.get("model")
        if self.model is None:
            raise ValueError("Answer ranker payload must contain model")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def select(
        self,
        query: str,
        candidates: Sequence[dict[str, str]],
        paragraphs: Sequence[dict[str, Any]],
        profile: str,
    ) -> AnswerRankerPrediction:
        if not candidates:
            return AnswerRankerPrediction(variant="empty", answer="", score=0.0)
        if not self.is_loaded:
            self.load_model()
        rows = []
        for candidate in candidates:
            features = extract_answer_features(
                query,
                candidate.get("answer", ""),
                paragraphs,
                profile,
                candidate.get("variant", "base"),
            )
            rows.append([features[name] for name in FEATURE_ORDER])
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(rows)[:, 1].tolist()
        else:
            scores = self.model.predict(rows).tolist()
        best_index = max(range(len(candidates)), key=lambda index: scores[index])
        best = candidates[best_index]
        return AnswerRankerPrediction(
            variant=best.get("variant", "unknown"),
            answer=best.get("answer", ""),
            score=float(scores[best_index]),
        )


def load_answer_ranker_if_available(model_path: str | None = None) -> AnswerRanker | None:
    ranker_path = model_path or config.ANSWER_RANKER_MODEL_PATH
    if not ranker_path:
        return None
    return AnswerRanker(ranker_path)
