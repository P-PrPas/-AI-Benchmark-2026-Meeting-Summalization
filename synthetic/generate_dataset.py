from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


DEFAULT_LANTA_PROJECT_ROOT = Path("/project/zz991000-zdeva/zz991011/CAMNET_P")
DEFAULT_LANTA_MODEL_ROOT = Path("/project/zz991000-zdeva/zz991011/models")
DEFAULT_LANTA_CACHE_ROOT = Path("/project/zz991000-zdeva/zz991011/.cache")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_base_root() -> Path:
    return DEFAULT_LANTA_PROJECT_ROOT if DEFAULT_LANTA_PROJECT_ROOT.exists() else LOCAL_PROJECT_ROOT


def _resolve_model_root(project_root: Path) -> Path:
    if DEFAULT_LANTA_MODEL_ROOT.exists():
        return DEFAULT_LANTA_MODEL_ROOT
    return project_root / "models"


def _resolve_cache_root(project_root: Path) -> Path:
    if DEFAULT_LANTA_CACHE_ROOT.exists():
        return DEFAULT_LANTA_CACHE_ROOT
    return project_root / ".cache"


def _env_path(env_name: str, default_path: Path, project_root: Path) -> Path:
    raw = os.environ.get(env_name)
    path = Path(raw).expanduser() if raw else default_path
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _env_int(env_name: str, default_value: int) -> int:
    return int(os.environ.get(env_name, str(default_value)))


def _env_float(env_name: str, default_value: float) -> float:
    return float(os.environ.get(env_name, str(default_value)))


def _env_bool(env_name: str, default_value: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default_value
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


PROJECT_ROOT = _resolve_base_root()
MODEL_ROOT = _resolve_model_root(PROJECT_ROOT)
CACHE_ROOT = _resolve_cache_root(PROJECT_ROOT)

SOURCE_TRAIN_JSON_PATH = _env_path(
    "CAMNET_SYNTH_SOURCE_TRAIN_JSON_PATH",
    PROJECT_ROOT / "data" / "train" / "train_set.json",
    PROJECT_ROOT,
)
OUTPUT_JSON_PATH = _env_path(
    "CAMNET_SYNTH_OUTPUT_JSON_PATH",
    PROJECT_ROOT / "data" / "train" / "train_set_synthetic.json",
    PROJECT_ROOT,
)
MODEL_NAME_OR_PATH = _env_path(
    "CAMNET_SYNTH_MODEL_NAME_OR_PATH",
    MODEL_ROOT / "gemma-4-31B-it",
    PROJECT_ROOT,
)
CACHE_DIR = _env_path(
    "CAMNET_SYNTH_CACHE_DIR",
    CACHE_ROOT / "synthetic_generation",
    PROJECT_ROOT,
)
NUM_DOCS_TO_SAMPLE = _env_int("CAMNET_SYNTH_NUM_DOCS_TO_SAMPLE", 5)
NUM_QUERIES_PER_DOC = _env_int("CAMNET_SYNTH_NUM_QUERIES_PER_DOC", 4)
MAX_PARAGRAPHS_PER_PROMPT = _env_int("CAMNET_SYNTH_MAX_PARAGRAPHS_PER_PROMPT", 4)
MAX_NEW_TOKENS = _env_int("CAMNET_SYNTH_MAX_NEW_TOKENS", 512)
MAX_ATTEMPTS_PER_QUERY = _env_int("CAMNET_SYNTH_MAX_ATTEMPTS_PER_QUERY", 3)
MIN_QUERY_CHARS = _env_int("CAMNET_SYNTH_MIN_QUERY_CHARS", 12)
MIN_ANSWER_CHARS = _env_int("CAMNET_SYNTH_MIN_ANSWER_CHARS", 16)
MAX_EXISTING_QUERY_EXAMPLES = _env_int("CAMNET_SYNTH_MAX_EXISTING_QUERY_EXAMPLES", 3)
TEMPERATURE = _env_float("CAMNET_SYNTH_TEMPERATURE", 0.35)
TOP_P = _env_float("CAMNET_SYNTH_TOP_P", 0.9)
SEED = _env_int("CAMNET_SYNTH_SEED", 42)
LOCAL_FILES_ONLY = _env_bool("CAMNET_SYNTH_LOCAL_FILES_ONLY", True)


QUESTION_STYLE_GUIDES = [
    (
        "fact",
        "สร้างคำถามเชิงข้อเท็จจริงเฉพาะ เช่น บุคคล หน่วยงาน สถานที่ เวลา จำนวน หรือมติสำคัญ และตอบเป็น 1 ประโยคสมบูรณ์",
    ),
    (
        "list",
        "สร้างคำถามเชิงรายการ เช่น มีอะไรบ้าง มีใครบ้าง หรือประเด็นสำคัญมีอะไรบ้าง และตอบเป็นประโยคเกริ่นนำสั้น ๆ ตามด้วยรายการลำดับเลข",
    ),
    (
        "reason",
        "สร้างคำถามเชิงเหตุผลหรือสาเหตุที่มีคำตอบตรงในข้อความ และตอบอย่างกระชับโดยไม่แต่งเติมเกินข้อความต้นทาง",
    ),
    (
        "policy",
        "สร้างคำถามเกี่ยวกับข้อเสนอ แนวทาง มาตรการ หรือแผนดำเนินการ และตอบโดยคงถ้อยคำสำคัญจากต้นฉบับ",
    ),
    (
        "constraint",
        "สร้างคำถามเกี่ยวกับเงื่อนไข หลักเกณฑ์ หรือข้อกำหนด และตอบให้ชัดว่าข้อกำหนดนั้นคืออะไร",
    ),
]


SYNTHETIC_SYSTEM_PROMPT = """คุณเป็นผู้สร้างข้อมูลสังเคราะห์สำหรับงาน RAG QA จากบันทึกการประชุมรัฐสภาไทย

เป้าหมาย:
- สร้างตัวอย่าง training data คุณภาพสูงที่เหมาะสำหรับฝึกโมเดลตอบคำถามจากเอกสาร
- ข้อมูลทุกส่วนต้องยึดตามย่อหน้าที่ให้มาเท่านั้น

ข้อกำหนดบังคับ:
- สร้างคำถามภาษาไทยที่มีคำตอบตรงจากย่อหน้าที่ให้มา
- คำตอบต้องเป็นภาษาไทยที่ชัดเจน กระชับ เป็นธรรมชาติ และสอดคล้องกับรูปแบบ dataset
- หากคำตอบมีหลายรายการ ให้ใช้รายการลำดับเลข
- ห้ามใส่ para_id เช่น [P12] ในฟิลด์ abstractive
- refs ต้องอ้างอิงเฉพาะ para_id ที่อยู่ในชุด paragraph_ids ที่ให้มา
- ห้ามสร้างข้อเท็จจริงที่ไม่มีในข้อความ
- ห้ามตอบกว้างเกินย่อหน้าต้นทาง
- ห้ามส่งออกข้อความอื่นนอกจาก JSON object 1 ตัว
"""


def configure_cache_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_local_model_path(model_name_or_path: str | Path) -> Path:
    model_path = Path(model_name_or_path).expanduser()
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {model_path}. "
            "Synthetic generation only supports local model weights."
        )
    return model_path


def load_source_dataset(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json(path)
    docs = payload.get("docs") or []
    queries = payload.get("queries") or []
    if not docs or not queries:
        raise ValueError("Source dataset must contain non-empty 'docs' and 'queries'.")
    return docs, queries


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def build_source_query_index(queries: list[dict[str, Any]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for row in queries:
        index[row["doc_id"]].add(normalize_text(row.get("query", "")))
    return index


def build_doc_query_examples(
    queries: list[dict[str, Any]],
    *,
    max_examples: int,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in queries:
        query_text = (row.get("query") or "").strip()
        if query_text:
            grouped[row["doc_id"]].append(query_text)
    return {doc_id: rows[:max_examples] for doc_id, rows in grouped.items()}


def sample_docs(docs: list[dict[str, Any]], num_docs: int, rng: random.Random) -> list[dict[str, Any]]:
    if num_docs <= 0:
        raise ValueError("NUM_DOCS_TO_SAMPLE must be greater than 0.")
    if num_docs >= len(docs):
        return list(docs)
    return rng.sample(docs, k=num_docs)


def build_paragraph_window(
    doc: dict[str, Any],
    rng: random.Random,
    max_paragraphs: int,
) -> list[dict[str, str]]:
    paragraphs = [row for row in doc.get("paragraphs", []) if row.get("text", "").strip()]
    if not paragraphs:
        return []
    window_size = rng.randint(1, min(max_paragraphs, len(paragraphs)))
    start = rng.randint(0, len(paragraphs) - window_size)
    return paragraphs[start : start + window_size]


def select_style_guide(doc_idx: int, query_idx: int, attempt_idx: int) -> tuple[str, str]:
    guide_index = (doc_idx + query_idx + attempt_idx) % len(QUESTION_STYLE_GUIDES)
    return QUESTION_STYLE_GUIDES[guide_index]


def build_generation_prompt(
    *,
    doc_id: str,
    paragraphs: list[dict[str, str]],
    style_name: str,
    style_instruction: str,
    existing_queries: list[str],
) -> str:
    paragraph_block = "\n".join(
        f"[{paragraph['para_id']}] {paragraph['text'].strip()}"
        for paragraph in paragraphs
    )
    para_ids = ", ".join(paragraph["para_id"] for paragraph in paragraphs)
    existing_query_block = "\n".join(f"- {query}" for query in existing_queries) if existing_queries else "- ไม่มี"
    return (
        "สร้างข้อมูลสังเคราะห์สำหรับงาน RAG QA จากย่อหน้าต่อไปนี้\n\n"
        f"doc_id: {doc_id}\n"
        f"style: {style_name}\n"
        f"paragraph_ids: {para_ids}\n\n"
        "ตัวอย่างคำถามเดิมในเอกสารเดียวกันที่ห้ามซ้ำหรือใกล้เคียงเกินไป:\n"
        f"{existing_query_block}\n\n"
        "ย่อหน้าต้นทาง:\n"
        f"{paragraph_block}\n\n"
        "หลักการสร้างตัวอย่างคุณภาพสูง:\n"
        "- คำถามต้องตอบได้จากย่อหน้าเหล่านี้เท่านั้นและไม่กว้างเกินไป\n"
        "- เลือกประเด็นที่มีสาระจริง ไม่ใช่คำถามผิวเผินหรือกำกวม\n"
        "- รักษาชื่อบุคคล หน่วยงาน ตัวเลข เวลา สถานที่ และเงื่อนไขให้ตรงข้อความ\n"
        "- ถ้าคำตอบมีหลายรายการ ให้ตอบเป็นรายการลำดับเลข\n"
        "- abstractive ต้องไม่มี para_id เช่น [P12]\n"
        "- refs ต้องเลือกจาก paragraph_ids ข้างต้นเท่านั้น และต้องตรงกับย่อหน้าที่ใช้ตอบจริง\n"
        f"- แนวคำถามรอบนี้: {style_instruction}\n\n"
        "ส่งออกเป็น JSON object 1 ตัวเท่านั้น โดยมี key ต่อไปนี้:\n"
        '- "query": string\n'
        '- "abstractive": string\n'
        '- "refs": list[string]\n\n'
        "ตัวอย่างรูปแบบที่ถูกต้อง:\n"
        '{"query":"...","abstractive":"...","refs":["P1","P2"]}'
    )


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Generated payload is not a JSON object.")
    return payload


def sanitize_answer(answer: str) -> str:
    text = (answer or "").strip()
    text = re.sub(r"^(?:คำตอบ|ตอบ)\s*[:：]\s*", "", text)
    text = re.sub(r"\[(P\d+)\]\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_refs(
    refs: Any,
    *,
    allowed_ref_ids: set[str],
    paragraph_order: dict[str, int],
) -> list[str]:
    if isinstance(refs, str):
        raw_refs = [part.strip() for part in refs.split(",") if part.strip()]
    elif isinstance(refs, list):
        raw_refs = [str(part).strip() for part in refs if str(part).strip()]
    else:
        raw_refs = []
    valid_refs = [ref for ref in raw_refs if ref in allowed_ref_ids]
    return sorted(set(valid_refs), key=lambda ref: paragraph_order[ref])


def build_synthetic_row_id(doc_id: str, accepted_count: int) -> str:
    suffix = doc_id.replace("doc_", "")
    return f"syn_doc_{suffix}_q_{accepted_count:03d}"


def validate_candidate(
    candidate: dict[str, Any],
    *,
    doc: dict[str, Any],
    paragraph_window: list[dict[str, Any]],
    normalized_source_queries: set[str],
    accepted_queries_for_doc: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    query = (candidate.get("query") or "").strip()
    answer = sanitize_answer(candidate.get("abstractive") or "")
    paragraph_order = {
        paragraph["para_id"]: idx
        for idx, paragraph in enumerate(doc.get("paragraphs", []))
    }
    allowed_ref_ids = {paragraph["para_id"] for paragraph in paragraph_window}
    refs = normalize_refs(
        candidate.get("refs"),
        allowed_ref_ids=allowed_ref_ids,
        paragraph_order=paragraph_order,
    )

    if not query:
        return None, "empty_query"
    if len(query) < MIN_QUERY_CHARS:
        return None, "query_too_short"
    if not answer:
        return None, "empty_answer"
    if len(answer) < MIN_ANSWER_CHARS:
        return None, "answer_too_short"
    if not refs:
        return None, "invalid_refs"

    normalized_query = normalize_text(query)
    normalized_answer = normalize_text(answer)
    if not normalized_query:
        return None, "empty_query"
    if normalized_query in normalized_source_queries:
        return None, "duplicate_source_query"
    if normalized_query in accepted_queries_for_doc:
        return None, "duplicate_synthetic_query"
    if normalized_query == normalized_answer:
        return None, "query_equals_answer"

    return {
        "query": query,
        "abstractive": answer,
        "refs": refs,
        "_normalized_query": normalized_query,
    }, None


def load_local_model(model_path: Path):
    print(f"Loading local synthetic-generation model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
        cache_dir=str(CACHE_DIR),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
        cache_dir=str(CACHE_DIR),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Local model loaded successfully")
    return tokenizer, model


def generate_candidate(
    tokenizer,
    model,
    *,
    doc_id: str,
    paragraphs: list[dict[str, str]],
    style_name: str,
    style_instruction: str,
    existing_queries: list[str],
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYNTHETIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_generation_prompt(
                doc_id=doc_id,
                paragraphs=paragraphs,
                style_name=style_name,
                style_instruction=style_instruction,
                existing_queries=existing_queries,
            ),
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "parameters"):
        device = next(model.parameters()).device
    if device is not None:
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = generated[0][model_inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return extract_json_object(text)


def validate_output_dataset(payload: dict[str, Any]) -> None:
    docs = payload.get("docs") or []
    queries = payload.get("queries") or []
    if not docs:
        raise ValueError("Synthetic output has no docs.")
    if not queries:
        raise ValueError("Synthetic output has no queries.")
    doc_lookup = {doc["doc_id"]: doc for doc in docs}
    for row in queries:
        missing_keys = {"ID", "doc_id", "query", "abstractive", "refs"} - set(row)
        if missing_keys:
            raise ValueError(f"Synthetic query row missing keys: {sorted(missing_keys)}")
        doc = doc_lookup.get(row["doc_id"])
        if doc is None:
            raise ValueError(f"Query references missing doc_id: {row['doc_id']}")
        paragraph_ids = {paragraph["para_id"] for paragraph in doc.get("paragraphs", [])}
        if not row["query"].strip():
            raise ValueError(f"Empty query found for row {row['ID']}")
        if not row["abstractive"].strip():
            raise ValueError(f"Empty abstractive found for row {row['ID']}")
        if not row["refs"]:
            raise ValueError(f"Empty refs found for row {row['ID']}")
        invalid_refs = [ref for ref in row["refs"] if ref not in paragraph_ids]
        if invalid_refs:
            raise ValueError(f"Invalid refs for row {row['ID']}: {invalid_refs}")


def build_report_path(output_json_path: Path) -> Path:
    return output_json_path.with_name(f"{output_json_path.stem}_report.json")


def main() -> None:
    source_train_json_path = SOURCE_TRAIN_JSON_PATH.resolve()
    output_json_path = OUTPUT_JSON_PATH.resolve()
    model_path = ensure_local_model_path(MODEL_NAME_OR_PATH)
    configure_cache_env(CACHE_DIR.resolve())
    set_global_seed(SEED)

    docs, queries = load_source_dataset(source_train_json_path)
    source_query_index = build_source_query_index(queries)
    doc_query_examples = build_doc_query_examples(
        queries,
        max_examples=MAX_EXISTING_QUERY_EXAMPLES,
    )
    rng = random.Random(SEED)
    sampled_docs = sample_docs(docs, NUM_DOCS_TO_SAMPLE, rng)
    tokenizer, model = load_local_model(model_path)

    synthetic_queries: list[dict[str, Any]] = []
    used_docs: dict[str, dict[str, Any]] = {}
    accepted_per_doc: Counter[str] = Counter()
    synthetic_query_index: dict[str, set[str]] = defaultdict(set)
    drop_counter: Counter[str] = Counter()
    accepted_style_counter: Counter[str] = Counter()

    total_requested = len(sampled_docs) * NUM_QUERIES_PER_DOC
    print(f"Loaded source docs={len(docs)} queries={len(queries)}")
    print(
        f"Generating synthetic data for sampled_docs={len(sampled_docs)} "
        f"requested_queries={total_requested}"
    )

    for doc_idx, doc in enumerate(sampled_docs, start=1):
        doc_id = doc["doc_id"]
        existing_queries = doc_query_examples.get(doc_id, [])
        print(f"[doc {doc_idx}/{len(sampled_docs)}] {doc_id}")
        for query_idx in range(NUM_QUERIES_PER_DOC):
            accepted_this_slot = False
            for attempt_idx in range(MAX_ATTEMPTS_PER_QUERY):
                paragraph_window = build_paragraph_window(doc, rng, MAX_PARAGRAPHS_PER_PROMPT)
                if not paragraph_window:
                    drop_counter["empty_paragraph_window"] += 1
                    break

                style_name, style_instruction = select_style_guide(doc_idx, query_idx, attempt_idx)
                try:
                    candidate = generate_candidate(
                        tokenizer,
                        model,
                        doc_id=doc_id,
                        paragraphs=paragraph_window,
                        style_name=style_name,
                        style_instruction=style_instruction,
                        existing_queries=existing_queries,
                    )
                except json.JSONDecodeError:
                    drop_counter["malformed_json"] += 1
                    continue
                except Exception as exc:
                    print(f"  generation failed for {doc_id} slot={query_idx + 1}: {exc}")
                    drop_counter["generation_error"] += 1
                    continue

                accepted, reject_reason = validate_candidate(
                    candidate,
                    doc=doc,
                    paragraph_window=paragraph_window,
                    normalized_source_queries=source_query_index.get(doc_id, set()),
                    accepted_queries_for_doc=synthetic_query_index[doc_id],
                )
                if accepted is None:
                    drop_counter[reject_reason or "unknown_reject_reason"] += 1
                    continue

                accepted_per_doc[doc_id] += 1
                accepted_style_counter[style_name] += 1
                synthetic_query_index[doc_id].add(accepted["_normalized_query"])
                row_id = build_synthetic_row_id(doc_id, accepted_per_doc[doc_id])
                synthetic_queries.append(
                    {
                        "ID": row_id,
                        "doc_id": doc_id,
                        "query": accepted["query"],
                        "abstractive": accepted["abstractive"],
                        "refs": accepted["refs"],
                    }
                )
                used_docs[doc_id] = doc
                accepted_this_slot = True
                break

            if not accepted_this_slot:
                drop_counter["exhausted_attempts"] += 1

    output_payload = {
        "docs": [used_docs[doc_id] for doc_id in sorted(used_docs)],
        "queries": synthetic_queries,
    }
    validate_output_dataset(output_payload)
    save_json(output_json_path, output_payload)

    report_payload = {
        "project_root": str(PROJECT_ROOT),
        "source_train_json_path": str(source_train_json_path),
        "output_json_path": str(output_json_path),
        "model_name_or_path": str(model_path),
        "cache_dir": str(CACHE_DIR.resolve()),
        "num_docs_requested": NUM_DOCS_TO_SAMPLE,
        "num_docs_sampled": len(sampled_docs),
        "num_docs_used": len(used_docs),
        "num_queries_per_doc": NUM_QUERIES_PER_DOC,
        "total_queries_requested": total_requested,
        "total_queries_accepted": len(synthetic_queries),
        "dropped_by_reason": dict(drop_counter),
        "accepted_per_doc": dict(accepted_per_doc),
        "accepted_by_style": dict(accepted_style_counter),
        "local_files_only": LOCAL_FILES_ONLY,
        "seed": SEED,
        "max_attempts_per_query": MAX_ATTEMPTS_PER_QUERY,
    }
    report_path = build_report_path(output_json_path)
    save_json(report_path, report_payload)

    print(f"Saved synthetic dataset to {output_json_path}")
    print(f"Saved generation report to {report_path}")
    print(f"Accepted synthetic queries={len(synthetic_queries)}")
    print(f"Dropped rows={dict(drop_counter)}")


if __name__ == "__main__":
    main()
