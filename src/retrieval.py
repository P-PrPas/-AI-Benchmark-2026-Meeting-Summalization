from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from . import config
from .prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS

if TYPE_CHECKING:
    from .embedder import Embedder, FAISSRetriever


_QUERY_REWRITE_REPLACEMENTS = (
    ("กมธ.", "คณะกรรมาธิการ"),
    ("ครม.", "คณะรัฐมนตรี"),
    ("รมว.", "รัฐมนตรีว่าการ"),
    ("รองฯ", "รอง"),
)
_QUERY_TRAILING_PATTERNS = (
    "คืออะไร",
    "คือใคร",
    "คือที่ใด",
    "คือที่ไหน",
    "คือเมื่อใด",
    "มีอะไรบ้าง",
    "ได้แก่อะไรบ้าง",
    "ได้แก่ใครบ้าง",
    "อย่างไร",
    "เมื่อใด",
    "ที่ใด",
    "ที่ไหน",
    "หรือไม่",
)


@dataclass(frozen=True)
class ReferenceSelectionConfig:
    max_refs: int = config.REFERENCE_TOP_N_MAX
    top2_min: float = config.REF_SELECTION_TOP2_MIN
    top3_min: float = config.REF_SELECTION_TOP3_MIN
    fact_max_gap: float = config.REF_SELECTION_FACT_MAX_GAP
    aggregate_max_gap: float = config.REF_SELECTION_AGG_MAX_GAP
    fact_top2_min: float = config.REF_SELECTION_FACT_TOP2_MIN
    list_top2_min: float = config.REF_SELECTION_LIST_TOP2_MIN
    list_top3_min: float = config.REF_SELECTION_LIST_TOP3_MIN
    synthesis_top2_min: float = config.REF_SELECTION_SYNTH_TOP2_MIN
    synthesis_top3_min: float = config.REF_SELECTION_SYNTH_TOP3_MIN
    list_max_gap: float = config.REF_SELECTION_LIST_MAX_GAP
    synthesis_max_gap: float = config.REF_SELECTION_SYNTH_MAX_GAP
    low_confidence_top1: float = config.REF_SELECTION_LOW_CONFIDENCE
    max_entropy: float = config.QUERY_REFINEMENT_MAX_ENTROPY


@dataclass(frozen=True)
class ReferenceSelectionResult:
    selected_refs: List[str]
    rule_refs: List[str]
    selector_refs: List[str]
    arbiter_refs: List[str]
    selector_used: bool
    arbiter_triggered: bool
    arbiter_used: bool
    arbiter_fallback: bool
    profile: str
    evidence_refs: List[str] = field(default_factory=list)
    evidence_used: bool = False


def retrieval_candidate_count(*, use_reranker: bool | None = None) -> int:
    if use_reranker is None:
        use_reranker = config.USE_RERANKER
    top_k = config.RETRIEVAL_CANDIDATE_K
    if use_reranker:
        top_k = max(top_k, config.RERANK_TOP_K)
    return top_k


def tokenize_for_overlap(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def lexical_overlap_score(query: str, paragraph_text: str) -> float:
    query_tokens = tokenize_for_overlap(query)
    if not query_tokens:
        return 0.0
    para_tokens = set(tokenize_for_overlap(paragraph_text))
    if not para_tokens:
        return 0.0
    overlap = sum(1 for token in query_tokens if token in para_tokens)
    return overlap / max(1, len(set(query_tokens)))


def _normalize_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    if float(arr.max() - arr.min()) < 1e-8:
        return [1.0 for _ in values]
    normalized = (arr - arr.min()) / (arr.max() - arr.min())
    return normalized.tolist()


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(arr.max())
    exps = np.exp(arr)
    denom = float(exps.sum())
    if denom <= 0:
        return [1.0 / len(values) for _ in values]
    return (exps / denom).tolist()


def normalized_score_entropy(values: Sequence[float]) -> float:
    probs = _softmax(values)
    if not probs:
        return 0.0
    entropy = -sum(prob * math.log(max(prob, 1e-8)) for prob in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(entropy / max(max_entropy, 1e-8))


def hybrid_rerank(
    query: str,
    retrieved: Sequence[Dict],
    *,
    dense_weight: float = config.HYBRID_DENSE_WEIGHT,
    lexical_weight: float = config.HYBRID_LEXICAL_WEIGHT,
) -> List[Dict]:
    if not retrieved:
        return []
    dense_scores = [float(item.get("score", 0.0)) for item in retrieved]
    lexical_scores = [lexical_overlap_score(query, item.get("text", "")) for item in retrieved]
    dense_norm = _normalize_scores(dense_scores)
    lexical_norm = _normalize_scores(lexical_scores)

    reranked = []
    for item, dense_score, lexical_score, dense_scaled, lexical_scaled in zip(
        retrieved,
        dense_scores,
        lexical_scores,
        dense_norm,
        lexical_norm,
    ):
        hybrid_score = dense_weight * dense_scaled + lexical_weight * lexical_scaled
        reranked.append(
            {
                **item,
                "dense_score": dense_score,
                "lexical_score": lexical_score,
                "hybrid_score": hybrid_score,
                "score": hybrid_score,
            }
        )
    reranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return reranked


def neural_rerank(
    query: str,
    retrieved: Sequence[Dict],
    reranker,
    *,
    top_k: int = config.RERANK_TOP_K,
) -> List[Dict]:
    if not retrieved or reranker is None:
        return list(retrieved)
    limit = min(len(retrieved), top_k)
    rescored = reranker.rerank(query, list(retrieved[:limit]), top_k=limit)
    rerank_scores = [float(item.get("rerank_score", 0.0)) for item in rescored[:limit]]
    rerank_norm = _normalize_scores(rerank_scores)
    merged: List[Dict] = []
    for item, rerank_scaled in zip(rescored[:limit], rerank_norm):
        hybrid_score = float(item.get("hybrid_score", item.get("score", 0.0)))
        final_score = 0.85 * rerank_scaled + 0.15 * hybrid_score
        merged.append(
            {
                **item,
                "rerank_scaled": rerank_scaled,
                "selection_score": final_score,
                "score": final_score,
            }
        )
    merged.sort(key=lambda item: item["selection_score"], reverse=True)
    if limit < len(retrieved):
        merged.extend(retrieved[limit:])
    return merged


def should_apply_neural_rerank(
    retrieved: Sequence[Dict],
    *,
    profile: str,
    reranker,
) -> bool:
    if reranker is None or not retrieved:
        return False
    if not config.ENABLE_ADAPTIVE_RERANKING:
        return True
    if profile in {ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS}:
        return True

    top_scores = [float(item.get("score", 0.0)) for item in retrieved[:5]]
    if len(top_scores) < 2:
        return False

    top1 = top_scores[0]
    top2 = top_scores[1]
    entropy = normalized_score_entropy(top_scores)
    if (
        top1 >= config.ADAPTIVE_RERANK_FACT_TOP1_MIN
        and (top1 - top2) >= config.ADAPTIVE_RERANK_FACT_MIN_GAP
        and entropy <= config.ADAPTIVE_RERANK_FACT_MAX_ENTROPY
    ):
        return False
    return True


def rerank_retrieved(
    query: str,
    dense_retrieved: Sequence[Dict],
    *,
    profile: str = ANSWER_PROFILE_FACT,
    reranker=None,
    rerank_top_k: int = config.RERANK_TOP_K,
) -> List[Dict]:
    hybrid = hybrid_rerank(query, dense_retrieved)
    if not should_apply_neural_rerank(hybrid, profile=profile, reranker=reranker):
        return hybrid
    return neural_rerank(query, hybrid, reranker, top_k=rerank_top_k)


def rewrite_query_heuristic(query: str) -> str:
    rewritten = re.sub(r"\s+", " ", (query or "").strip())
    for src, dst in _QUERY_REWRITE_REPLACEMENTS:
        rewritten = rewritten.replace(src, dst)
    rewritten = rewritten.rstrip(" ?")
    for suffix in _QUERY_TRAILING_PATTERNS:
        if rewritten.endswith(suffix):
            rewritten = rewritten[: -len(suffix)].strip()
            break
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return rewritten or query.strip()


def needs_query_refinement(
    retrieved: Sequence[Dict],
    profile: str,
    calibration_config: ReferenceSelectionConfig | None = None,
) -> bool:
    calibration_config = calibration_config or ReferenceSelectionConfig()
    if not config.ENABLE_QUERY_REFINEMENT or not retrieved:
        return False
    top_scores = [float(item.get("score", 0.0)) for item in retrieved[:5]]
    if not top_scores:
        return False
    top1 = top_scores[0]
    top2 = top_scores[1] if len(top_scores) > 1 else 0.0
    entropy = normalized_score_entropy(top_scores)
    if top1 < calibration_config.low_confidence_top1:
        return True
    if entropy > calibration_config.max_entropy:
        return True
    if profile == ANSWER_PROFILE_FACT and abs(top1 - top2) < calibration_config.fact_max_gap / 2:
        return True
    return False


def retrieve_references(
    retriever: FAISSRetriever,
    doc_id: str,
    query: str,
    top_k: int = config.RETRIEVAL_CANDIDATE_K,
) -> List[str]:
    results = hybrid_rerank(query, retriever.retrieve(doc_id, query, top_k))
    return [r["para_id"] for r in results[:top_k]]


def get_top_references_by_score(
    retriever: FAISSRetriever,
    doc_id: str,
    query: str,
    n: int = config.REFERENCE_TOP_N,
) -> List[Tuple[str, float]]:
    results = hybrid_rerank(query, retriever.retrieve(doc_id, query, top_k=max(n * 2, n)))
    return [(r["para_id"], r["score"]) for r in results[:n]]


def cross_encode_rerank(
    query: str,
    paragraphs: List[Dict],
    embedder: Embedder,
    top_k: int = config.RETRIEVAL_CANDIDATE_K,
) -> List[Dict]:
    if not paragraphs:
        return []

    query_emb = embedder.encode_query(query)
    para_embs, para_ids = embedder.encode_paragraphs(paragraphs)

    dense_scores = np.dot(para_embs, query_emb)
    prelim = []
    for idx in np.argsort(dense_scores)[::-1][:top_k]:
        prelim.append(
            {
                "para_id": para_ids[idx],
                "text": paragraphs[idx]["text"],
                "score": float(dense_scores[idx]),
            }
        )
    return hybrid_rerank(query, prelim)


def _selection_scores(retrieved: Sequence[Dict]) -> List[float]:
    scores = [float(item.get("selection_score", item.get("score", 0.0))) for item in retrieved]
    return _softmax(scores[: max(config.REFERENCE_TOP_N_MAX, 4)])


def _top2_min_for_profile(profile: str, calibration_config: ReferenceSelectionConfig) -> float:
    if profile == ANSWER_PROFILE_FACT:
        return calibration_config.fact_top2_min
    if profile == ANSWER_PROFILE_SYNTHESIS:
        return calibration_config.synthesis_top2_min
    return calibration_config.list_top2_min


def _top3_min_for_profile(profile: str, calibration_config: ReferenceSelectionConfig) -> float:
    if profile == ANSWER_PROFILE_SYNTHESIS:
        return calibration_config.synthesis_top3_min
    return calibration_config.list_top3_min


def _gap_limit_for_profile(profile: str, calibration_config: ReferenceSelectionConfig) -> float:
    if profile == ANSWER_PROFILE_FACT:
        return calibration_config.fact_max_gap
    if profile == ANSWER_PROFILE_SYNTHESIS:
        return calibration_config.synthesis_max_gap
    return calibration_config.list_max_gap


def _dynamic_rule_select(
    retrieved: Sequence[Dict],
    *,
    profile: str,
    calibration_config: ReferenceSelectionConfig | None = None,
    n: int | None = None,
) -> List[str]:
    if not retrieved:
        return []
    calibration_config = calibration_config or ReferenceSelectionConfig()
    limit = min(len(retrieved), n or calibration_config.max_refs)
    probs = _selection_scores(retrieved[:limit])
    selected = [retrieved[0]["para_id"]]
    gap_limit = _gap_limit_for_profile(profile, calibration_config)
    top2_min = _top2_min_for_profile(profile, calibration_config)
    top3_min = _top3_min_for_profile(profile, calibration_config)

    if limit >= 2:
        gap12 = probs[0] - probs[1]
        if probs[1] >= top2_min and gap12 <= gap_limit:
            selected.append(retrieved[1]["para_id"])
    if limit >= 3 and profile in {ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS}:
        gap23 = probs[1] - probs[2] if len(probs) > 2 else 1.0
        if probs[2] >= top3_min and gap23 <= gap_limit:
            selected.append(retrieved[2]["para_id"])
    return selected


def _arbiter_candidate_count(profile: str) -> int:
    if profile == ANSWER_PROFILE_FACT:
        return min(2, max(1, config.REF_ARBITER_MAX_CANDIDATES))
    if profile == ANSWER_PROFILE_SYNTHESIS:
        return min(3, max(2, config.REF_ARBITER_MAX_CANDIDATES))
    return min(3, max(2, config.REF_ARBITER_MAX_CANDIDATES))


def _should_trigger_ref_arbiter(
    retrieved: Sequence[Dict],
    *,
    profile: str,
    calibration_config: ReferenceSelectionConfig,
) -> bool:
    if not config.ENABLE_LLM_REF_ARBITER or not retrieved:
        return False

    trigger_mode = (config.REF_ARBITER_TRIGGER_MODE or "ambiguous_only").strip().lower()
    if trigger_mode == "always":
        return True
    if trigger_mode != "ambiguous_only":
        return False

    top_k = min(len(retrieved), _arbiter_candidate_count(profile))
    top_scores = [float(item.get("selection_score", item.get("score", 0.0))) for item in retrieved[:top_k]]
    if len(top_scores) < 2:
        return False

    top1 = top_scores[0]
    top2 = top_scores[1]
    entropy = normalized_score_entropy(top_scores)
    gap_limit = calibration_config.fact_max_gap if profile == ANSWER_PROFILE_FACT else calibration_config.aggregate_max_gap

    if abs(top1 - top2) <= gap_limit:
        return True
    if entropy >= calibration_config.max_entropy:
        return True
    if top1 < calibration_config.low_confidence_top1:
        return True
    if profile in {ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS} and len(top_scores) >= 3:
        top3 = top_scores[2]
        if abs(top2 - top3) <= gap_limit:
            return True
    return False


def _apply_ref_arbiter(
    query: str,
    retrieved: Sequence[Dict],
    *,
    profile: str,
    rule_refs: Sequence[str],
    generator: Any | None = None,
) -> tuple[List[str], bool, bool]:
    if generator is None:
        return list(rule_refs), False, False
    candidate_count = min(len(retrieved), _arbiter_candidate_count(profile))
    candidate_paragraphs = [dict(item) for item in retrieved[:candidate_count]]
    arbiter_refs = generator.arbitrate_references(
        query,
        candidate_paragraphs,
        profile=profile,
        rule_refs=list(rule_refs),
    )
    if not arbiter_refs:
        return list(rule_refs), True, True

    candidate_ids = {item["para_id"] for item in candidate_paragraphs}
    valid_refs = []
    for para_id in arbiter_refs:
        if para_id in candidate_ids and para_id not in valid_refs:
            valid_refs.append(para_id)
    if not valid_refs:
        return list(rule_refs), True, True
    return valid_refs, True, False


def _apply_learned_selector(
    query: str,
    retrieved: Sequence[Dict],
    *,
    profile: str,
    rule_refs: Sequence[str],
    ref_selector: Any | None = None,
) -> tuple[List[str], bool]:
    if ref_selector is None:
        return list(rule_refs), False
    prediction = ref_selector.predict(query, retrieved, profile)
    selected = [retrieved[0]["para_id"]]
    if len(retrieved) >= 2 and prediction.keep2:
        selected.append(retrieved[1]["para_id"])
    if len(retrieved) >= 3 and prediction.keep3 and profile in {ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS}:
        selected.append(retrieved[2]["para_id"])
    return selected, True


def _apply_evidence_set_selector(
    query: str,
    retrieved: Sequence[Dict],
    *,
    profile: str,
    rule_refs: Sequence[str],
    evidence_selector: Any | None = None,
) -> tuple[List[str], bool]:
    if evidence_selector is None:
        return list(rule_refs), False
    prediction = evidence_selector.predict(query, retrieved, profile)
    candidate_ids = {item["para_id"] for item in retrieved[:20]}
    selected = [para_id for para_id in prediction.refs if para_id in candidate_ids]
    if not selected:
        return list(rule_refs), False
    return selected, True


def select_references_with_diagnostics(
    query: str,
    retrieved: Sequence[Dict],
    *,
    profile: str | None = None,
    calibration_config: ReferenceSelectionConfig | None = None,
    n: int | None = None,
    mode: str | None = None,
    ref_selector: Any | None = None,
    evidence_selector: Any | None = None,
    generator: Any | None = None,
) -> ReferenceSelectionResult:
    profile = profile or ANSWER_PROFILE_FACT
    if not retrieved:
        return ReferenceSelectionResult(
            selected_refs=[],
            rule_refs=[],
            selector_refs=[],
            arbiter_refs=[],
            selector_used=False,
            arbiter_triggered=False,
            arbiter_used=False,
            arbiter_fallback=False,
            profile=profile,
        )

    if mode:
        resolved_mode = mode.lower()
    elif config.ENABLE_DYNAMIC_REF_SELECTION:
        if config.ENABLE_EVIDENCE_SET_SELECTOR:
            resolved_mode = "dynamic_rules_then_evidence_set"
        elif config.ENABLE_LEARNED_REF_SELECTOR:
            resolved_mode = "dynamic_rules_then_selector"
        elif config.ENABLE_LLM_REF_ARBITER:
            resolved_mode = "dynamic_rules_then_llm_arbiter"
        else:
            resolved_mode = "dynamic_rules"
    else:
        resolved_mode = "fixed"
    if resolved_mode not in {
        "fixed",
        "dynamic_rules",
        "dynamic_rules_then_selector",
        "dynamic_rules_then_evidence_set",
        "dynamic_rules_then_llm_arbiter",
        "dynamic",
        "dynamic_rules_then_arbiter",
    }:
        raise ValueError(f"Unsupported reference selection mode: {resolved_mode}")
    if resolved_mode == "dynamic":
        resolved_mode = "dynamic_rules"
    if resolved_mode == "dynamic_rules_then_arbiter":
        resolved_mode = "dynamic_rules_then_llm_arbiter"

    calibration_config = calibration_config or ReferenceSelectionConfig()
    if resolved_mode == "fixed":
        selected_refs = [item["para_id"] for item in retrieved[: (n or config.REFERENCE_TOP_N)]]
        return ReferenceSelectionResult(
            selected_refs=selected_refs,
            rule_refs=selected_refs,
            selector_refs=[],
            arbiter_refs=[],
            selector_used=False,
            arbiter_triggered=False,
            arbiter_used=False,
            arbiter_fallback=False,
            profile=profile,
        )

    rule_limit = n or calibration_config.max_refs
    rule_refs = _dynamic_rule_select(
        retrieved,
        profile=profile,
        calibration_config=calibration_config,
        n=rule_limit,
    )
    if resolved_mode == "dynamic_rules":
        return ReferenceSelectionResult(
            selected_refs=rule_refs,
            rule_refs=rule_refs,
            selector_refs=[],
            arbiter_refs=[],
            selector_used=False,
            arbiter_triggered=False,
            arbiter_used=False,
            arbiter_fallback=False,
            profile=profile,
        )
    if resolved_mode == "dynamic_rules_then_selector":
        selector_refs, selector_used = _apply_learned_selector(
            query,
            retrieved,
            profile=profile,
            rule_refs=rule_refs,
            ref_selector=ref_selector,
        )
        final_refs = selector_refs if selector_used else rule_refs
        return ReferenceSelectionResult(
            selected_refs=final_refs,
            rule_refs=rule_refs,
            selector_refs=selector_refs if selector_used else [],
            arbiter_refs=[],
            selector_used=selector_used,
            arbiter_triggered=False,
            arbiter_used=False,
            arbiter_fallback=False,
            profile=profile,
        )
    if resolved_mode == "dynamic_rules_then_evidence_set":
        evidence_refs, evidence_used = _apply_evidence_set_selector(
            query,
            retrieved,
            profile=profile,
            rule_refs=rule_refs,
            evidence_selector=evidence_selector,
        )
        final_refs = evidence_refs if evidence_used else rule_refs
        return ReferenceSelectionResult(
            selected_refs=final_refs,
            rule_refs=rule_refs,
            selector_refs=[],
            arbiter_refs=[],
            selector_used=False,
            arbiter_triggered=False,
            arbiter_used=False,
            arbiter_fallback=False,
            profile=profile,
            evidence_refs=evidence_refs if evidence_used else [],
            evidence_used=evidence_used,
        )

    arbiter_triggered = _should_trigger_ref_arbiter(
        retrieved,
        profile=profile,
        calibration_config=calibration_config,
    )
    if not arbiter_triggered:
        return ReferenceSelectionResult(
            selected_refs=rule_refs,
            rule_refs=rule_refs,
            selector_refs=[],
            arbiter_refs=[],
            selector_used=False,
            arbiter_triggered=False,
            arbiter_used=False,
            arbiter_fallback=False,
            profile=profile,
        )

    selected_refs, arbiter_used, arbiter_fallback = _apply_ref_arbiter(
        query,
        retrieved,
        profile=profile,
        rule_refs=rule_refs,
        generator=generator,
    )
    arbiter_refs = selected_refs if arbiter_used and not arbiter_fallback else []
    return ReferenceSelectionResult(
        selected_refs=selected_refs,
        rule_refs=rule_refs,
        selector_refs=[],
        arbiter_refs=arbiter_refs,
        selector_used=False,
        arbiter_triggered=arbiter_triggered,
        arbiter_used=arbiter_used,
        arbiter_fallback=arbiter_fallback,
        profile=profile,
    )


def select_references_from_retrieved(
    retrieved: Sequence[Dict],
    profile: str | None = None,
    calibration_config: ReferenceSelectionConfig | None = None,
    n: int | None = None,
    mode: str | None = None,
) -> List[str]:
    query = retrieved[0].get("query", "") if retrieved else ""
    return select_references_with_diagnostics(
        query,
        retrieved,
        profile=profile,
        calibration_config=calibration_config,
        n=n,
        mode=mode,
        ref_selector=None,
        generator=None,
    ).selected_refs


def sentence_split(text: str) -> List[str]:
    chunks = re.split(r"(?<=[\.\?!…])\s+|\n+", (text or "").strip())
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    results = []
    for item in items:
        key = re.sub(r"\s+", " ", item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def compress_evidence(
    query: str,
    paragraphs: Sequence[Mapping[str, str]],
    profile: str,
) -> List[Dict]:
    paragraphs = [dict(item) for item in paragraphs if (item.get("text") or "").strip()]
    if not paragraphs:
        return []
    if profile == ANSWER_PROFILE_LIST:
        compressed = []
        for paragraph in paragraphs:
            text = paragraph.get("text", "").strip()
            numbered = re.findall(r"(?:^|\s)(\d+\.\s*[^0-9\n]+(?:\s(?!\d+\.)[^0-9\n]+)*)", text)
            if numbered:
                text = "\n".join(item.strip() for item in numbered)
            compressed.append({**paragraph, "text": text})
        return compressed
    if profile == ANSWER_PROFILE_SYNTHESIS:
        compressed = []
        for paragraph in paragraphs:
            sentences = sentence_split(paragraph.get("text", ""))
            ranked = sorted(
                ((lexical_overlap_score(query, sentence), sentence) for sentence in sentences),
                key=lambda item: item[0],
                reverse=True,
            )
            selected = [sentence for _, sentence in ranked[:3]] or sentences[:3]
            compressed.append({**paragraph, "text": " ".join(_dedupe_preserve_order(selected))})
        return compressed

    sentence_candidates: List[tuple[float, int, Dict]] = []
    for para_rank, paragraph in enumerate(paragraphs):
        for sent_rank, sentence in enumerate(sentence_split(paragraph.get("text", ""))):
            score = lexical_overlap_score(query, sentence) + max(0.0, 0.15 - (0.05 * para_rank)) - (0.01 * sent_rank)
            sentence_candidates.append(
                (
                    score,
                    para_rank,
                    {
                        "para_id": paragraph["para_id"],
                        "text": sentence,
                    },
                )
            )
    sentence_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = _dedupe_preserve_order(item[2]["text"] for item in sentence_candidates[:2])
    if not selected:
        return paragraphs[:1]
    return [{"para_id": paragraphs[0]["para_id"], "text": " ".join(selected)}]


def build_generation_context(
    query: str,
    reranked: Sequence[Dict],
    selected_refs: Sequence[str],
    profile: str,
) -> List[Dict]:
    selected_ref_set = set(selected_refs)
    primary = [dict(item) for item in reranked if item.get("para_id") in selected_ref_set]
    if not primary:
        primary = [dict(item) for item in reranked[:1]]

    if profile == ANSWER_PROFILE_FACT:
        paragraphs = primary[:2]
    elif profile == ANSWER_PROFILE_LIST:
        extra = [dict(item) for item in reranked if item.get("para_id") not in selected_ref_set][:2]
        paragraphs = primary + extra
    else:
        extra = [dict(item) for item in reranked if item.get("para_id") not in selected_ref_set][:2]
        paragraphs = primary + extra

    if not config.ENABLE_EVIDENCE_COMPRESSION:
        return paragraphs
    return compress_evidence(query, paragraphs, profile)


def compute_retrieval_metrics(
    gold_refs_list: Sequence[Sequence[str]],
    retrieved_list: Sequence[Sequence[Dict]],
) -> Dict[str, float]:
    total = max(1, len(gold_refs_list))
    hit1 = hit3 = hit10 = hit20 = 0
    recall10 = []
    recall20 = []
    iou3 = []
    for gold_refs, retrieved in zip(gold_refs_list, retrieved_list):
        gold_set = set(gold_refs or [])
        top1 = {item["para_id"] for item in retrieved[:1]}
        top3 = {item["para_id"] for item in retrieved[:3]}
        top10 = {item["para_id"] for item in retrieved[:10]}
        top20 = {item["para_id"] for item in retrieved[:20]}
        if gold_set & top1:
            hit1 += 1
        if gold_set & top3:
            hit3 += 1
        if gold_set & top10:
            hit10 += 1
        if gold_set & top20:
            hit20 += 1
        if gold_set:
            recall10.append(len(gold_set & top10) / len(gold_set))
            recall20.append(len(gold_set & top20) / len(gold_set))
            iou3.append(len(gold_set & top3) / len(gold_set | top3) if (gold_set | top3) else 0.0)
    return {
        "hit_any_gold_at_1": hit1 / total,
        "hit_any_gold_at_3": hit3 / total,
        "hit_any_gold_at_10": hit10 / total,
        "hit_any_gold_at_20": hit20 / total,
        "mean_ref_recall_at_10": float(np.mean(recall10)) if recall10 else 0.0,
        "mean_ref_recall_at_20": float(np.mean(recall20)) if recall20 else 0.0,
        "mean_iou_top_3": float(np.mean(iou3)) if iou3 else 0.0,
    }


def compute_selected_reference_metrics(
    gold_refs_list: Sequence[Sequence[str]],
    predicted_refs_list: Sequence[Sequence[str]],
) -> Dict[str, float]:
    total = max(1, len(gold_refs_list))
    iou_scores = []
    pred_counts = []
    count_1 = count_2 = count_3_plus = 0
    for gold_refs, predicted_refs in zip(gold_refs_list, predicted_refs_list):
        gold_set = set(gold_refs or [])
        pred_set = set(predicted_refs or [])
        union = gold_set | pred_set
        iou_scores.append(len(gold_set & pred_set) / len(union) if union else 0.0)
        pred_count = len(pred_set)
        pred_counts.append(pred_count)
        if pred_count <= 1:
            count_1 += 1
        elif pred_count == 2:
            count_2 += 1
        else:
            count_3_plus += 1
    return {
        "selected_ref_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "pred_ref_count_mean": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "pred_ref_count_pct_1": count_1 / total,
        "pred_ref_count_pct_2": count_2 / total,
        "pred_ref_count_pct_3_plus": count_3_plus / total,
    }


def compute_selected_reference_metrics_by_profile(
    gold_refs_list: Sequence[Sequence[str]],
    predicted_refs_list: Sequence[Sequence[str]],
    profiles: Sequence[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for profile in (ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS):
        indices = [index for index, item_profile in enumerate(profiles) if item_profile == profile]
        if not indices:
            metrics[f"selected_ref_iou_{profile}"] = 0.0
            metrics[f"pred_ref_count_mean_{profile}"] = 0.0
            continue
        subset_gold = [gold_refs_list[index] for index in indices]
        subset_pred = [predicted_refs_list[index] for index in indices]
        subset_metrics = compute_selected_reference_metrics(subset_gold, subset_pred)
        metrics[f"selected_ref_iou_{profile}"] = subset_metrics["selected_ref_iou"]
        metrics[f"pred_ref_count_mean_{profile}"] = subset_metrics["pred_ref_count_mean"]
    return metrics


def compute_arbiter_metrics(selection_results: Sequence[ReferenceSelectionResult]) -> Dict[str, float]:
    total = max(1, len(selection_results))
    triggered = sum(1 for result in selection_results if result.arbiter_triggered)
    used = sum(1 for result in selection_results if result.arbiter_used)
    fallback = sum(1 for result in selection_results if result.arbiter_fallback)
    return {
        "arbiter_trigger_rate": triggered / total,
        "arbiter_usage_rate": used / total,
        "arbiter_fallback_rate": fallback / total,
    }


def compute_selector_metrics(selection_results: Sequence[ReferenceSelectionResult]) -> Dict[str, float]:
    total = max(1, len(selection_results))
    used = sum(1 for result in selection_results if result.selector_used)
    return {
        "selector_usage_rate": used / total,
    }


def compute_evidence_set_metrics(selection_results: Sequence[ReferenceSelectionResult]) -> Dict[str, float]:
    total = max(1, len(selection_results))
    used = sum(1 for result in selection_results if result.evidence_used)
    return {
        "evidence_set_usage_rate": used / total,
    }


if __name__ == "__main__":
    print("Retrieval module loaded successfully")
