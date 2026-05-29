from __future__ import annotations

import re
from typing import Mapping, Sequence

from . import config


ANSWER_PROFILE_FACT = "fact"
ANSWER_PROFILE_LIST = "list"
ANSWER_PROFILE_SYNTHESIS = "synthesis"
ANSWER_PROFILES = (
    ANSWER_PROFILE_FACT,
    ANSWER_PROFILE_LIST,
    ANSWER_PROFILE_SYNTHESIS,
)

NO_CONTEXT_TEXT = "(ไม่มีข้อมูลอ้างอิง)"
NO_ANSWER_TEXT = "ไม่พบข้อมูลในเอกสาร"

LIST_QUERY_HINTS = (
    "มีอะไรบ้าง",
    "ได้แก่อะไรบ้าง",
    "ได้แก่",
    "ประกอบด้วย",
    "ข้อเสนอแนะ",
    "แนวทาง",
    "มาตรการ",
    "ประเด็น",
    "มติ",
    "เชิญผู้ใด",
    "รายชื่อ",
)
SYNTHESIS_QUERY_HINTS = (
    "สรุป",
    "อย่างไร",
    "เพราะเหตุใด",
    "เหตุใด",
    "ทำไม",
    "สาระสำคัญ",
    "ภาพรวม",
)
LIST_CONTEXT_HINTS = ("1.", "2.", "3.", "ได้แก่", "ประกอบด้วย", "ข้อเสนอแนะ", "ข้อสังเกต", "แนวทาง")
TRAILING_FRAGMENT_MARKERS = ("...", "…", "คำตอบ:", "ตอบ:")

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากบันทึกการประชุมรัฐสภาไทย\n\n"
    "หน้าที่:\n"
    "- ใช้เฉพาะข้อมูลจากย่อหน้าที่ให้มาเท่านั้น และใช้ข้อมูลตามลำดับความเกี่ยวข้องที่จัดไว้ให้\n"
    "- ถ้าคำตอบอยู่ในข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุดอยู่แล้ว ให้ใช้ถ้อยคำจากเอกสารนั้นให้มากที่สุด และตอบจากข้อมูลอ้างอิงข้อแรกก่อน\n"
    "- หลีกเลี่ยงการใช้คำพ้อง การสรุปใหม่ หรือการเกริ่นนำเกินความจำเป็น ถ้าเอกสารมีคำตอบชัดเจนอยู่แล้ว\n"
    "- รักษาสาระ ชื่อบุคคล ตำแหน่ง หน่วยงาน วันเวลา ตัวเลข และลำดับรายการให้ตรงกับเอกสาร\n"
    '- ห้ามขึ้นต้นด้วยคำเกริ่น เช่น "จากเอกสาร" หรือ "ที่ประชุมมีความเห็นว่า" ถ้าเอกสารไม่ได้ใช้คำเหล่านั้นในคำตอบโดยตรง\n'
    "- ถ้าคำตอบเป็นข้อเท็จจริงเดียว ให้ตอบ 1 ประโยคเป็นหลัก และใช้ 2 ประโยคเฉพาะเมื่อจำเป็นเพื่อให้คำตอบครบถ้วน\n"
    "- ถ้าคำตอบมีหลายรายการ ให้ตอบเป็นรายการลำดับเลขตามข้อมูลในเอกสาร\n"
    "- ถ้าคำตอบต้องสังเคราะห์จากหลายย่อหน้า ให้ตอบเป็นย่อหน้าเปิดสั้น ๆ แล้วตามด้วยสรุป 2-5 บรรทัดเท่าที่จำเป็น\n"
    f'- ถ้าข้อมูลไม่พอสำหรับตอบคำถาม ให้ตอบเพียงว่า "{NO_ANSWER_TEXT}"\n'
    '- ห้ามขึ้นต้นด้วย "ตอบ:" หรือ "คำตอบ:"\n'
    "- ห้ามใส่หมายเลขย่อหน้า เช่น [P12] ลงในคำตอบ\n"
    "- ห้ามอธิบายวิธีคิด ห้ามอธิบายโจทย์ซ้ำ และห้ามเพิ่มข้อมูลที่เอกสารไม่ได้ระบุ"
)

FACT_FEW_SHOT_EXAMPLES = """ตัวอย่างที่ 1
เอกสาร:
ข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุด:
1. [P18] วัตถุประสงค์ของการประชุมครั้งนี้เพื่อพิจารณาแนวทางแก้ไขปัญหาการบริหารจัดการน้ำในพื้นที่ลุ่มน้ำยม

คำถาม:
วัตถุประสงค์ของการประชุมครั้งนี้คืออะไร

คำตอบ:
วัตถุประสงค์ของการประชุมครั้งนี้คือพิจารณาแนวทางแก้ไขปัญหาการบริหารจัดการน้ำในพื้นที่ลุ่มน้ำยม

ตัวอย่างที่ 2
เอกสาร:
ข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุด:
1. [P52] กรมป้องกันและบรรเทาสาธารณภัยรายงานว่าสถานการณ์อุทกภัยในพื้นที่เริ่มคลี่คลายลงตั้งแต่วันที่ 15 กันยายน 2568

คำถาม:
สถานการณ์อุทกภัยในพื้นที่เริ่มคลี่คลายลงเมื่อใด

คำตอบ:
สถานการณ์อุทกภัยในพื้นที่เริ่มคลี่คลายลงตั้งแต่วันที่ 15 กันยายน 2568"""

LIST_FEW_SHOT_EXAMPLES = """ตัวอย่างที่ 1
เอกสาร:
ข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุด:
1. [P41] บุคคลที่ได้รับเชิญมาให้ข้อมูลต่อคณะกรรมาธิการ ได้แก่
2. [P42] 1. ปลัดกระทรวงมหาดไทย 2. อธิบดีกรมป้องกันและบรรเทาสาธารณภัย 3. ผู้ว่าราชการจังหวัดเชียงราย

คำถาม:
คณะกรรมาธิการเชิญหน่วยงานใดมาให้ข้อมูลบ้าง

คำตอบ:
1. ปลัดกระทรวงมหาดไทย
2. อธิบดีกรมป้องกันและบรรเทาสาธารณภัย
3. ผู้ว่าราชการจังหวัดเชียงราย"""

SYNTHESIS_FEW_SHOT_EXAMPLES = """ตัวอย่างที่ 1
เอกสาร:
ข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุด:
1. [P73] คณะกรรมาธิการเห็นว่าการส่งเสริมยานยนต์ไฟฟ้าต้องดำเนินควบคู่กับการพัฒนาโครงสร้างพื้นฐานการชาร์จไฟฟ้า
2. [P74] นอกจากนี้ควรมีมาตรการสนับสนุนผู้ประกอบการในประเทศและการพัฒนาบุคลากรด้านเทคโนโลยีแบตเตอรี่

ข้อมูลอ้างอิงเพิ่มเติม:
3. [P75] ที่ประชุมมีข้อสังเกตว่าการกำหนดมาตรฐานความปลอดภัยควรทำพร้อมกับการดูแลการจัดการซากแบตเตอรี่

คำถาม:
คณะกรรมาธิการมีข้อสังเกตอย่างไรเกี่ยวกับการส่งเสริมยานยนต์ไฟฟ้า

คำตอบ:
คณะกรรมาธิการเห็นว่าการส่งเสริมยานยนต์ไฟฟ้าควรดำเนินควบคู่กันหลายด้าน
จึงต้องพัฒนาโครงสร้างพื้นฐานการชาร์จไฟฟ้า สนับสนุนผู้ประกอบการในประเทศ และพัฒนาบุคลากรด้านเทคโนโลยีแบตเตอรี่
พร้อมทั้งกำหนดมาตรฐานความปลอดภัยและการจัดการซากแบตเตอรี่ให้ชัดเจน"""

NO_ANSWER_FEW_SHOT_EXAMPLE = """ตัวอย่างที่ 99
เอกสาร:
ข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุด:
1. [P9] ที่ประชุมรับทราบรายงานผลการดำเนินงานประจำไตรมาส

คำถาม:
ที่ประชุมกำหนดวันประชุมครั้งถัดไปเมื่อใด

คำตอบ:
ไม่พบข้อมูลในเอกสาร"""


def few_shot_examples_for_profile(profile: str) -> str:
    if profile == ANSWER_PROFILE_FACT and not config.ENABLE_FACT_FEW_SHOT:
        return ""
    if profile == ANSWER_PROFILE_LIST and not config.ENABLE_LIST_FEW_SHOT:
        return ""
    if profile == ANSWER_PROFILE_SYNTHESIS and not config.ENABLE_SYNTHESIS_FEW_SHOT:
        return ""
    if profile == ANSWER_PROFILE_LIST:
        examples = [LIST_FEW_SHOT_EXAMPLES, NO_ANSWER_FEW_SHOT_EXAMPLE]
    elif profile == ANSWER_PROFILE_SYNTHESIS:
        examples = [SYNTHESIS_FEW_SHOT_EXAMPLES, NO_ANSWER_FEW_SHOT_EXAMPLE]
    else:
        examples = [FACT_FEW_SHOT_EXAMPLES, NO_ANSWER_FEW_SHOT_EXAMPLE]
    return "\n\n".join(examples)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_numbered_text(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*\d+\.", text or ""))


def _query_contains_any(query: str, hints: Sequence[str]) -> bool:
    query_norm = normalize_text(query)
    return any(hint in query_norm for hint in hints)


def detect_answer_profile(
    query: str,
    paragraphs: Sequence[Mapping[str, str]] | None = None,
) -> str:
    query_norm = normalize_text(query)
    paragraphs = list(paragraphs or [])
    context_text = "\n".join(normalize_text(p.get("text", "")) for p in paragraphs)
    if _query_contains_any(query_norm, LIST_QUERY_HINTS):
        return ANSWER_PROFILE_LIST
    if _query_contains_any(query_norm, SYNTHESIS_QUERY_HINTS):
        return ANSWER_PROFILE_SYNTHESIS
    if any(hint in context_text for hint in LIST_CONTEXT_HINTS) or any(
        _is_numbered_text(p.get("text", "")) for p in paragraphs
    ):
        return ANSWER_PROFILE_LIST
    return ANSWER_PROFILE_FACT


def context_limit_for_profile(profile: str) -> int:
    if profile == ANSWER_PROFILE_FACT:
        return config.GENERATOR_CONTEXT_K_FACT
    if profile == ANSWER_PROFILE_SYNTHESIS:
        return config.GENERATOR_CONTEXT_K_SYNTHESIS
    return config.GENERATOR_CONTEXT_K_AGGREGATE


def format_ranked_context(
    paragraphs: Sequence[Mapping[str, str]] | None,
    *,
    primary_count: int | None = None,
) -> str:
    paragraphs = [paragraph for paragraph in (paragraphs or []) if normalize_text(paragraph.get("text", ""))]
    if not paragraphs:
        return NO_CONTEXT_TEXT

    primary_count = max(1, primary_count or min(len(paragraphs), config.GENERATOR_CONTEXT_K_FACT))
    primary = paragraphs[:primary_count]
    additional = paragraphs[primary_count:]

    def render_block(title: str, items: Sequence[Mapping[str, str]]) -> str:
        lines = [title + ":"]
        for idx, paragraph in enumerate(items, start=1):
            lines.append(f"{idx}. [{paragraph['para_id']}] {normalize_text(paragraph['text'])}")
        return "\n".join(lines)

    blocks = [render_block("ข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุด", primary)]
    if additional:
        blocks.append(render_block("ข้อมูลอ้างอิงเพิ่มเติม", additional))
    return "\n\n".join(blocks)


def build_user_prompt(
    context: str | Sequence[Mapping[str, str]],
    query: str,
    *,
    profile: str | None = None,
    primary_count: int | None = None,
) -> str:
    profile = profile or detect_answer_profile(query)
    if isinstance(context, str):
        context_text = normalize_text(context) if context.strip() == NO_CONTEXT_TEXT else context.strip()
    else:
        profile = profile or detect_answer_profile(query, context)
        context_text = format_ranked_context(
            context,
            primary_count=primary_count or context_limit_for_profile(profile),
        )
    sections = []
    examples = few_shot_examples_for_profile(profile)
    if examples:
        sections.append(examples)
    sections.append(
        "เอกสาร:\n"
        f"{context_text}\n\n"
        "คำสั่งเพิ่มเติม:\n"
        "- ข้อมูลด้านบนเรียงจากเกี่ยวข้องมากไปน้อย ให้ใช้เฉพาะข้อมูลที่ให้มาเท่านั้น\n"
        "- ถ้าคำตอบอยู่ในข้อมูลอ้างอิงข้อแรกอยู่แล้ว ให้ตอบจากข้อมูลอ้างอิงข้อแรกก่อน และใช้ข้อมูลอ้างอิงเพิ่มเติมเฉพาะเมื่อจำเป็น\n"
        "- ถ้าข้อมูลด้านบนยังไม่พอ ให้ตอบว่า ไม่พบข้อมูลในเอกสาร\n\n"
        f"คำถาม:\n{query.strip()}\n\n"
        "คำตอบ:\n"
    )
    return "\n\n".join(sections)


def _deduplicate_lines(text: str) -> str:
    deduped_lines = []
    previous_key = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        key = normalize_text(re.sub(r"(?:[.]{3,}|…+)\s*$", "", line))
        if line and key == previous_key:
            continue
        deduped_lines.append(line)
        previous_key = key if line else None
    return "\n".join(deduped_lines)


def _trim_trailing_fragment(text: str) -> str:
    trimmed = text.rstrip()
    while True:
        matched_suffix = next((suffix for suffix in TRAILING_FRAGMENT_MARKERS if trimmed.endswith(suffix)), None)
        if matched_suffix is None:
            break
        trimmed = trimmed[: -len(matched_suffix)].rstrip()
    trimmed = re.sub(r"(?:\s*[.]{3,}|…+)\s*$", "", trimmed).rstrip()
    return trimmed


def sanitize_generated_answer(answer: str) -> str:
    text = (answer or "").strip()
    text = re.sub(r"^(?:คำตอบ|ตอบ)\s*[:：]\s*", "", text)
    text = re.sub(r"(?m)^(?:เอกสาร|คำถาม|คำสั่งเพิ่มเติม)\s*:\s*$", "", text)
    text = re.sub(r"\[P\d+\]\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _deduplicate_lines(text)
    text = _trim_trailing_fragment(text)
    text = text.strip()
    return text or NO_ANSWER_TEXT


def answer_needs_retry(raw_answer: str, sanitized_answer: str, profile: str) -> bool:
    raw_answer = (raw_answer or "").strip()
    sanitized_answer = (sanitized_answer or "").strip()
    if not sanitized_answer or sanitized_answer == NO_ANSWER_TEXT:
        return False
    if raw_answer.endswith(("...", "…")):
        return True
    if profile == ANSWER_PROFILE_FACT and len(sanitized_answer) > config.FACT_MAX_ANSWER_CHARS:
        return True
    return False
