from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


DEFAULT_LANTA_PROJECT_ROOT = Path("/project/zz991000-zdeva/zz991011/CAMNET_P")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_base_root() -> Path:
    return DEFAULT_LANTA_PROJECT_ROOT if DEFAULT_LANTA_PROJECT_ROOT.exists() else LOCAL_PROJECT_ROOT


def _env_path(env_name: str, default_path: Path, project_root: Path) -> Path:
    raw = os.environ.get(env_name)
    path = Path(raw).expanduser() if raw else default_path
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _env_bool(env_name: str, default_value: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default_value
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


PROJECT_ROOT = _resolve_base_root()

ORIGINAL_TRAIN_JSON_PATH = _env_path(
    "CAMNET_MERGE_ORIGINAL_TRAIN_JSON_PATH",
    PROJECT_ROOT / "data" / "train" / "train_set.json",
    PROJECT_ROOT,
)
SYNTHETIC_JSON_PATH = _env_path(
    "CAMNET_MERGE_SYNTHETIC_JSON_PATH",
    PROJECT_ROOT / "data" / "train" / "train_set_synthetic.json",
    PROJECT_ROOT,
)
OUTPUT_JSON_PATH = _env_path(
    "CAMNET_MERGE_OUTPUT_JSON_PATH",
    PROJECT_ROOT / "data" / "train" / "train_set_merged.json",
    PROJECT_ROOT,
)
STRICT_DOC_MATCH = _env_bool("CAMNET_MERGE_STRICT_DOC_MATCH", True)
SKIP_DUPLICATE_DOC_QUERY = _env_bool("CAMNET_MERGE_SKIP_DUPLICATE_DOC_QUERY", True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def validate_dataset(payload: dict[str, Any], label: str) -> None:
    docs = payload.get("docs") or []
    queries = payload.get("queries") or []
    if not docs or not queries:
        raise ValueError(f"{label} must contain non-empty 'docs' and 'queries'.")
    doc_lookup = {doc["doc_id"]: doc for doc in docs}
    for row in queries:
        missing_keys = {"ID", "doc_id", "query", "abstractive", "refs"} - set(row)
        if missing_keys:
            raise ValueError(f"{label} query row missing keys: {sorted(missing_keys)}")
        doc = doc_lookup.get(row["doc_id"])
        if doc is None:
            raise ValueError(f"{label} query references missing doc_id: {row['doc_id']}")
        para_ids = {paragraph["para_id"] for paragraph in doc.get("paragraphs", [])}
        refs = row.get("refs") or []
        if not isinstance(refs, list):
            raise ValueError(f"{label} refs must be a list for row {row['ID']}")
        invalid_refs = [ref for ref in refs if ref not in para_ids]
        if invalid_refs:
            raise ValueError(f"{label} row {row['ID']} contains invalid refs: {invalid_refs}")


def build_report_path(output_json_path: Path) -> Path:
    return output_json_path.with_name(f"{output_json_path.stem}_report.json")


def merge_docs(
    original_docs: list[dict[str, Any]],
    synthetic_docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    doc_map: dict[str, dict[str, Any]] = {}
    report = Counter()
    for doc in original_docs:
        doc_map[doc["doc_id"]] = doc
    for doc in synthetic_docs:
        existing = doc_map.get(doc["doc_id"])
        if existing is None:
            doc_map[doc["doc_id"]] = doc
            report["new_synthetic_doc"] += 1
            continue
        if existing == doc:
            report["duplicate_doc_same_content"] += 1
            continue
        if STRICT_DOC_MATCH:
            raise ValueError(f"Synthetic doc content differs from original for doc_id={doc['doc_id']}")
        report["duplicate_doc_kept_original"] += 1
    merged_docs = [doc_map[doc_id] for doc_id in sorted(doc_map)]
    return merged_docs, report


def merge_queries(
    original_queries: list[dict[str, Any]],
    synthetic_queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    merged_queries = list(original_queries)
    report = Counter()
    seen_ids = {row["ID"] for row in original_queries}
    seen_doc_query = {
        (row["doc_id"], normalize_text(row.get("query", "")))
        for row in original_queries
    }
    for row in synthetic_queries:
        row_id = row["ID"]
        doc_query_key = (row["doc_id"], normalize_text(row.get("query", "")))
        if row_id in seen_ids:
            report["duplicate_query_id"] += 1
            continue
        if SKIP_DUPLICATE_DOC_QUERY and doc_query_key in seen_doc_query:
            report["duplicate_doc_query"] += 1
            continue
        merged_queries.append(row)
        seen_ids.add(row_id)
        seen_doc_query.add(doc_query_key)
        report["accepted_synthetic_query"] += 1
    return merged_queries, report


def main() -> None:
    original_payload = load_json(ORIGINAL_TRAIN_JSON_PATH.resolve())
    synthetic_payload = load_json(SYNTHETIC_JSON_PATH.resolve())

    validate_dataset(original_payload, "Original dataset")
    validate_dataset(synthetic_payload, "Synthetic dataset")

    merged_docs, doc_report = merge_docs(
        original_payload["docs"],
        synthetic_payload["docs"],
    )
    merged_queries, query_report = merge_queries(
        original_payload["queries"],
        synthetic_payload["queries"],
    )

    merged_payload = {
        "docs": merged_docs,
        "queries": merged_queries,
    }
    validate_dataset(merged_payload, "Merged dataset")
    save_json(OUTPUT_JSON_PATH.resolve(), merged_payload)

    report_payload = {
        "project_root": str(PROJECT_ROOT),
        "original_train_json_path": str(ORIGINAL_TRAIN_JSON_PATH.resolve()),
        "synthetic_json_path": str(SYNTHETIC_JSON_PATH.resolve()),
        "output_json_path": str(OUTPUT_JSON_PATH.resolve()),
        "strict_doc_match": STRICT_DOC_MATCH,
        "skip_duplicate_doc_query": SKIP_DUPLICATE_DOC_QUERY,
        "original_doc_count": len(original_payload["docs"]),
        "synthetic_doc_count": len(synthetic_payload["docs"]),
        "merged_doc_count": len(merged_docs),
        "original_query_count": len(original_payload["queries"]),
        "synthetic_query_count": len(synthetic_payload["queries"]),
        "merged_query_count": len(merged_queries),
        "doc_report": dict(doc_report),
        "query_report": dict(query_report),
    }
    report_path = build_report_path(OUTPUT_JSON_PATH.resolve())
    save_json(report_path, report_payload)

    print(f"Saved merged dataset to {OUTPUT_JSON_PATH.resolve()}")
    print(f"Saved merge report to {report_path}")
    print(report_payload["query_report"])


if __name__ == "__main__":
    main()
