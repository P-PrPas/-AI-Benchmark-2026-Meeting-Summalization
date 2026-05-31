from __future__ import annotations

import re
from typing import Any, Sequence

from . import config
from .prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS, NO_ANSWER_TEXT, sanitize_generated_answer
from .retrieval import lexical_overlap_score, sentence_split


def extract_evidence_sentences(
    query: str,
    paragraphs: Sequence[dict[str, Any]],
    *,
    max_sentences: int | None = None,
) -> list[str]:
    limit = max_sentences or config.SEMI_EXTRACTIVE_MAX_SENTENCES
    candidates: list[tuple[float, int, int, str]] = []
    for para_rank, paragraph in enumerate(paragraphs):
        for sent_rank, sentence in enumerate(sentence_split(str(paragraph.get("text", "")))):
            sentence = sentence.strip()
            if not sentence:
                continue
            score = lexical_overlap_score(query, sentence) + max(0.0, 0.12 - 0.03 * para_rank) - 0.005 * sent_rank
            candidates.append((score, para_rank, sent_rank, sentence))
    candidates.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)
    seen = set()
    selected = []
    for _, _, _, sentence in candidates:
        key = re.sub(r"\s+", " ", sentence)
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def _extract_list_items(paragraphs: Sequence[dict[str, Any]]) -> list[str]:
    items: list[str] = []
    for paragraph in paragraphs:
        text = str(paragraph.get("text", ""))
        matches = re.findall(r"(?:^|\n|\s)(?:\d+[\.\)]|[-•])\s*([^\n]+)", text)
        for item in matches:
            clean = re.sub(r"\s+", " ", item).strip(" ;")
            if clean:
                items.append(clean)
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def build_extractive_answer(query: str, paragraphs: Sequence[dict[str, Any]], profile: str) -> str:
    if not paragraphs:
        return ""
    if profile == ANSWER_PROFILE_LIST:
        list_items = _extract_list_items(paragraphs)
        if list_items:
            return "\n".join(f"{index}. {item}" for index, item in enumerate(list_items[:6], start=1))
        sentences = extract_evidence_sentences(query, paragraphs, max_sentences=4)
        return "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1))
    if profile == ANSWER_PROFILE_SYNTHESIS:
        return " ".join(extract_evidence_sentences(query, paragraphs, max_sentences=config.SEMI_EXTRACTIVE_MAX_SENTENCES))
    sentences = extract_evidence_sentences(query, paragraphs, max_sentences=1)
    if not sentences:
        return ""
    best = sentences[0]
    if lexical_overlap_score(query, best) < config.SEMI_EXTRACTIVE_FACT_MIN_OVERLAP:
        return ""
    return best


def semi_extractive_compose(
    query: str,
    paragraphs: Sequence[dict[str, Any]],
    profile: str,
    generated_answer: str,
) -> str:
    generated = sanitize_generated_answer(generated_answer)
    extractive = sanitize_generated_answer(build_extractive_answer(query, paragraphs, profile))
    if not extractive:
        return generated or NO_ANSWER_TEXT
    if not generated or generated == NO_ANSWER_TEXT:
        return extractive
    if profile == ANSWER_PROFILE_FACT:
        if len(generated) > config.FACT_MAX_ANSWER_CHARS or len(extractive) <= len(generated) * 0.85:
            return extractive
        generated_overlap = lexical_overlap_score(generated, " ".join(str(p.get("text", "")) for p in paragraphs))
        extractive_overlap = lexical_overlap_score(extractive, " ".join(str(p.get("text", "")) for p in paragraphs))
        return extractive if extractive_overlap >= generated_overlap else generated
    if profile == ANSWER_PROFILE_LIST:
        generated_numbered = bool(re.search(r"(^|\n)\s*1[\.\)]", generated))
        extractive_numbered = bool(re.search(r"(^|\n)\s*1[\.\)]", extractive))
        if extractive_numbered and not generated_numbered:
            return extractive
    return generated
