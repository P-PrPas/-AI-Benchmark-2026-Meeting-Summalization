from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config


@dataclass(frozen=True)
class RerankerScore:
    para_id: str
    score: float


class NeuralReranker:
    """Cross-encoder style reranker for query-paragraph relevance scoring."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        batch_size: int = 8,
        max_length: int = 2048,
    ) -> None:
        self.model_path = str(model_path or config.RERANK_MODEL_PATH)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Reranker model path not found: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            fallback_pad = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback_pad is not None:
                self.tokenizer.pad_token = fallback_pad
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        if not self.is_loaded:
            self.load_model()
        assert self.model is not None and self.tokenizer is not None

        results: List[float] = []
        effective_batch_size = self.batch_size if self.tokenizer.pad_token_id is not None else 1
        for start in range(0, len(pairs), effective_batch_size):
            batch = pairs[start:start + effective_batch_size]
            queries = [item[0] for item in batch]
            passages = [item[1] for item in batch]
            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)
            logits = outputs.logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                batch_scores = logits[:, 0]
            elif logits.ndim == 2 and logits.shape[-1] >= 2:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)
            results.extend(float(score) for score in batch_scores.detach().float().cpu())
        return results

    def rerank(self, query: str, paragraphs: Sequence[Dict], top_k: int | None = None) -> List[Dict]:
        if not paragraphs:
            return []
        limit = len(paragraphs) if top_k is None else min(len(paragraphs), top_k)
        pairs = [(query, item.get("text", "")) for item in paragraphs[:limit]]
        scores = self.score_pairs(pairs)
        rescored = []
        for item, score in zip(paragraphs[:limit], scores):
            rescored.append({**item, "rerank_score": score})
        rescored.sort(key=lambda item: item["rerank_score"], reverse=True)
        if limit < len(paragraphs):
            rescored.extend(paragraphs[limit:])
        return rescored


def load_reranker_if_available(model_path: str | None = None) -> NeuralReranker | None:
    model_path = str(model_path or config.RERANK_MODEL_PATH)
    if not config.USE_RERANKER:
        return None
    if not Path(model_path).exists():
        return None
    reranker = NeuralReranker(model_path=model_path)
    return reranker
