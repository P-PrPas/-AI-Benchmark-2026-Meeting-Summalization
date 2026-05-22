import json
from typing import Dict, List, Any
from . import config


def load_dataset(split: str = "train") -> Dict[str, Any]:
    """Load train or test dataset."""
    path = config.TRAIN_PATH if split == "train" else config.TEST_PATH

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def get_document_paragraphs(data: Dict, doc_id: str) -> List[Dict]:
    """Get all paragraphs for a specific document."""
    for doc in data["docs"]:
        if doc["doc_id"] == doc_id:
            return doc["paragraphs"]
    return []


def get_queries(data: Dict, doc_id: str = None) -> List[Dict]:
    """Get queries, optionally filtered by document."""
    queries = data.get("queries", [])
    if doc_id:
        queries = [q for q in queries if q["doc_id"] == doc_id]
    return queries


def build_paragraph_index(data: Dict) -> Dict[str, List[Dict]]:
    """Build index: doc_id -> list of paragraphs."""
    index = {}
    for doc in data["docs"]:
        index[doc["doc_id"]] = doc["paragraphs"]
    return index


def get_paragraph_by_id(paragraphs: List[Dict], para_id: str) -> str:
    """Get paragraph text by para_id."""
    for p in paragraphs:
        if p["para_id"] == para_id:
            return p["text"]
    return ""


if __name__ == "__main__":
    train_data = load_dataset("train")
    test_data = load_dataset("test")

    print(f"Train: {len(train_data['docs'])} docs, {len(train_data['queries'])} queries")
    print(f"Test: {len(test_data['docs'])} docs, {len(test_data['queries'])} queries")