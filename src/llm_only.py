from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from . import config
from .prompting import NO_ANSWER_TEXT, sanitize_generated_answer


PROMPT_RAW = "raw"
PROMPT_MINIMAL = "minimal"
PROMPT_GOLD_STYLE = "gold_style"
PROMPT_REF_STRICT = "ref_strict"
PROMPT_MODES = {PROMPT_RAW, PROMPT_MINIMAL, PROMPT_GOLD_STYLE, PROMPT_REF_STRICT}


@dataclass(frozen=True)
class LLMOnlyPrediction:
    abstractive: str
    refs: list[str]
    parse_error: bool
    invalid_refs: list[str]
    raw_response: str


def normalize_prompt_mode(mode: str | None) -> str:
    normalized = (mode or PROMPT_MINIMAL).strip().lower()
    if normalized not in PROMPT_MODES:
        raise ValueError(f"Unsupported LLM-only prompt mode: {mode}. Expected one of {sorted(PROMPT_MODES)}")
    return normalized


def paragraph_ids(paragraphs: Sequence[dict[str, Any]]) -> list[str]:
    return [str(paragraph.get("para_id", "")).strip() for paragraph in paragraphs if paragraph.get("para_id")]


def truncate_paragraphs_by_chars(
    paragraphs: Sequence[dict[str, Any]],
    *,
    max_doc_chars: int | None = None,
) -> list[dict[str, Any]]:
    limit = max_doc_chars if max_doc_chars is not None else config.LLM_ONLY_MAX_DOC_CHARS
    if limit <= 0:
        return list(paragraphs)
    selected: list[dict[str, Any]] = []
    used = 0
    for paragraph in paragraphs:
        text = str(paragraph.get("text", "")).strip()
        para_id = str(paragraph.get("para_id", "")).strip()
        if not text or not para_id:
            continue
        block_len = len(para_id) + len(text) + 8
        if selected and used + block_len > limit:
            break
        selected.append({"para_id": para_id, "text": text})
        used += block_len
    return selected


def format_full_document(paragraphs: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{paragraph['para_id']}] {str(paragraph.get('text', '')).strip()}"
        for paragraph in paragraphs
        if paragraph.get("para_id") and str(paragraph.get("text", "")).strip()
    )


def build_llm_only_prompt(
    query: str,
    paragraphs: Sequence[dict[str, Any]],
    *,
    mode: str | None = None,
    max_doc_chars: int | None = None,
) -> str:
    mode = normalize_prompt_mode(mode)
    selected_paragraphs = truncate_paragraphs_by_chars(paragraphs, max_doc_chars=max_doc_chars)
    document_text = format_full_document(selected_paragraphs)
    valid_refs = ", ".join(paragraph_ids(selected_paragraphs))

    if mode == PROMPT_RAW:
        return f"{document_text}\n\nคำถาม: {query}"

    output_contract = (
        'ตอบเป็น JSON เท่านั้นในรูปแบบ {"abstractive":"...","refs":["Pxx"]}\n'
        "refs ต้องเป็น para_id ที่มีอยู่ในเอกสารเท่านั้น และเลือกเฉพาะย่อหน้าที่ใช้ตอบจริง"
    )
    if mode == PROMPT_MINIMAL:
        instruction = output_contract
    elif mode == PROMPT_GOLD_STYLE:
        instruction = (
            f"{output_contract}\n"
            "กติกาคำตอบ: ตอบให้สั้น กระชับ ไม่ขึ้นต้นด้วยคำเกริ่น เช่น จากเอกสาร/จากข้อมูล\n"
            "ถ้าข้อความในเอกสารตอบได้ตรงอยู่แล้ว ให้คงถ้อยคำเดิม ชื่อคน หน่วยงาน วันที่ ตัวเลข และลำดับรายการให้ใกล้เอกสารที่สุด\n"
            "ห้าม paraphrase ถ้าไม่จำเป็น และห้ามใส่ citation marker ใน abstractive"
        )
    else:
        instruction = (
            f"{output_contract}\n"
            f"para_id ที่อนุญาตมีเฉพาะ: {valid_refs}\n"
            "ห้ามสร้าง para_id ใหม่ ห้ามเลือก ref ที่ไม่ได้เกี่ยวข้องโดยตรง และถ้าไม่แน่ใจให้เลือก ref เดียวที่ตอบคำถามได้ชัดที่สุด"
        )

    return (
        f"{instruction}\n\n"
        f"เอกสาร:\n{document_text}\n\n"
        f"คำถาม: {query}\n"
        "คำตอบ JSON:"
    )


def build_llm_only_target(answer: str, refs: Sequence[str], *, mode: str | None = None) -> str:
    mode = normalize_prompt_mode(mode)
    clean_answer = sanitize_generated_answer(answer)
    clean_refs = [str(ref).strip() for ref in refs if str(ref).strip()]
    if mode == PROMPT_RAW:
        return f"{clean_answer}\nrefs: {','.join(clean_refs)}"
    return json.dumps({"abstractive": clean_answer, "refs": clean_refs}, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _refs_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,،\s]+", value) if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _extract_refs_from_text(text: str, valid_ref_set: set[str]) -> list[str]:
    ordered: list[str] = []
    for match in re.finditer(r"\bP\d+\b", text or "", flags=re.IGNORECASE):
        ref = match.group(0).upper()
        if ref in valid_ref_set and ref not in ordered:
            ordered.append(ref)
    return ordered


def _extract_answer_from_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*(คำตอบ|ตอบ|answer|abstractive)\s*[:：]\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*(refs?|อ้างอิง)\s*[:：].*$", "", cleaned).strip()
    cleaned = re.sub(r"\{.*\}", "", cleaned, flags=re.DOTALL).strip() if cleaned.startswith("{") else cleaned
    return sanitize_generated_answer(cleaned)


def parse_llm_only_output(raw_response: str, valid_refs: Iterable[str]) -> LLMOnlyPrediction:
    valid_ref_set = {str(ref).strip().upper() for ref in valid_refs if str(ref).strip()}
    valid_ref_lookup = {str(ref).strip().upper(): str(ref).strip() for ref in valid_refs if str(ref).strip()}
    parse_error = False
    invalid_refs: list[str] = []
    answer = ""
    refs: list[str] = []

    parsed_json = _extract_json_object(raw_response)
    if parsed_json is not None:
        answer_value = (
            parsed_json.get("abstractive")
            or parsed_json.get("answer")
            or parsed_json.get("คำตอบ")
            or ""
        )
        answer = sanitize_generated_answer(str(answer_value))
        raw_refs = _refs_from_value(parsed_json.get("refs") or parsed_json.get("ref_id") or parsed_json.get("references"))
        for ref in raw_refs:
            normalized = ref.strip().strip("[]").upper()
            if normalized in valid_ref_set and valid_ref_lookup[normalized] not in refs:
                refs.append(valid_ref_lookup[normalized])
            elif normalized:
                invalid_refs.append(ref)
    else:
        parse_error = True
        answer = _extract_answer_from_text(raw_response)
        refs = _extract_refs_from_text(raw_response, valid_ref_set)
        if not refs:
            ref_line = re.search(r"(?im)^\s*(?:refs?|อ้างอิง)\s*[:：]\s*(.+)$", raw_response or "")
            if ref_line:
                for ref in _refs_from_value(ref_line.group(1)):
                    normalized = ref.strip().strip("[]").upper()
                    if normalized in valid_ref_set and valid_ref_lookup[normalized] not in refs:
                        refs.append(valid_ref_lookup[normalized])
                    elif normalized:
                        invalid_refs.append(ref)

    if not answer:
        parse_error = True
        answer = NO_ANSWER_TEXT
    return LLMOnlyPrediction(
        abstractive=answer,
        refs=refs,
        parse_error=parse_error,
        invalid_refs=invalid_refs,
        raw_response=raw_response or "",
    )


def fallback_refs_by_answer_overlap(answer: str, paragraphs: Sequence[dict[str, Any]], *, max_refs: int = 1) -> list[str]:
    answer_tokens = set(re.findall(r"\w+", answer or "", flags=re.UNICODE))
    if not answer_tokens:
        return []
    scored: list[tuple[float, str]] = []
    for paragraph in paragraphs:
        para_id = str(paragraph.get("para_id", "")).strip()
        para_tokens = set(re.findall(r"\w+", str(paragraph.get("text", "")), flags=re.UNICODE))
        if not para_id or not para_tokens:
            continue
        score = len(answer_tokens & para_tokens) / max(1, len(answer_tokens))
        scored.append((score, para_id))
    scored.sort(reverse=True)
    return [para_id for score, para_id in scored[:max_refs] if score > 0.0]
