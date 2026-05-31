from __future__ import annotations

import re
from typing import Any, Sequence

from .prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS, sanitize_generated_answer
from .retrieval import lexical_overlap_score, tokenize_for_overlap


VARIANT_INSTRUCTIONS = {
    "concise": (
        "Rewrite the answer in Thai as briefly as possible. "
        "Keep only information directly supported by the references. "
        "Do not add an introduction."
    ),
    "gold_style": (
        "Answer in Thai using wording close to the provided references and the expected gold-answer style. "
        "Preserve names, dates, numbers, order, and key phrases. "
        "Avoid paraphrasing when the source wording already answers the question."
    ),
    "list_style": (
        "If the answer contains multiple items, answer as a compact numbered list. "
        "Use short list items and do not add explanations beyond the evidence."
    ),
    "source_faithful": (
        "Answer in Thai with strict source fidelity. "
        "Use only statements grounded in the references and keep the answer compact."
    ),
}


def build_candidate_prompt(base_prompt: str, variant: str, profile: str) -> str:
    if variant == "base":
        return base_prompt
    instruction = VARIANT_INSTRUCTIONS.get(variant)
    if not instruction:
        return base_prompt
    profile_hint = {
        ANSWER_PROFILE_FACT: "Default to one sentence for factual questions.",
        ANSWER_PROFILE_LIST: "Prefer numbered-list formatting for list questions.",
        ANSWER_PROFILE_SYNTHESIS: "Prefer a short synthesis without broad background.",
    }.get(profile, "")
    return f"{base_prompt}\n\nAdditional answer-style constraint:\n{instruction} {profile_hint}".strip()


def _source_overlap(answer: str, paragraphs: Sequence[dict[str, Any]]) -> float:
    tokens = set(tokenize_for_overlap(answer))
    if not tokens:
        return 0.0
    source_tokens = set()
    for paragraph in paragraphs:
        source_tokens.update(tokenize_for_overlap(str(paragraph.get("text", ""))))
    return len(tokens & source_tokens) / max(1, len(tokens))


def heuristic_answer_score(query: str, answer: str, paragraphs: Sequence[dict[str, Any]], profile: str) -> float:
    answer = sanitize_generated_answer(answer)
    if not answer:
        return -1.0
    source_text = " ".join(str(paragraph.get("text", "")) for paragraph in paragraphs)
    source_overlap = _source_overlap(answer, paragraphs)
    query_overlap = lexical_overlap_score(query, answer)
    source_similarity = lexical_overlap_score(answer, source_text)
    length = len(answer)
    if profile == ANSWER_PROFILE_FACT:
        length_score = max(0.0, 1.0 - abs(length - 130) / 220)
    elif profile == ANSWER_PROFILE_LIST:
        length_score = max(0.0, 1.0 - abs(length - 220) / 360)
    else:
        length_score = max(0.0, 1.0 - abs(length - 240) / 420)
    preamble_penalty = 0.15 if re.match(r"^\s*(จาก|ตาม|ใน)(เอกสาร|ข้อมูล|ที่ประชุม)", answer) else 0.0
    return 0.45 * source_overlap + 0.25 * source_similarity + 0.20 * length_score + 0.10 * query_overlap - preamble_penalty


def select_heuristic_candidate(
    query: str,
    candidates: Sequence[dict[str, str]],
    paragraphs: Sequence[dict[str, Any]],
    profile: str,
) -> dict[str, str]:
    if not candidates:
        return {"variant": "empty", "answer": ""}
    ranked = sorted(
        candidates,
        key=lambda item: heuristic_answer_score(query, item.get("answer", ""), paragraphs, profile),
        reverse=True,
    )
    return ranked[0]
