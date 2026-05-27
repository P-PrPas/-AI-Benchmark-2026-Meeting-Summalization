from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


RERANK_SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
RERANK_ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_reranker_pair(query: str, document: str, instruction: str | None = None) -> str:
    instruction = instruction or config.RERANK_INSTRUCTION
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}".format(
        instruction=instruction,
        query=query,
        document=document,
    )


@dataclass(frozen=True)
class RerankerScore:
    para_id: str
    score: float


class NeuralReranker:
    """Official Qwen3 reranker wrapper using causal-LM yes/no scoring."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        batch_size: int = 4,
        max_length: int = 8192,
        instruction: str | None = None,
    ) -> None:
        self.model_path = str(model_path or config.RERANK_MODEL_PATH)
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction or config.RERANK_INSTRUCTION
        self.model = None
        self.tokenizer = None
        self._prefix_tokens: List[int] = []
        self._suffix_tokens: List[int] = []
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None

    def load_model(self) -> None:
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Reranker model path not found: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        if self.tokenizer.pad_token is None:
            raise ValueError("Reranker tokenizer must define a pad token or eos token.")

        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

        self._prefix_tokens = self.tokenizer.encode(RERANK_SYSTEM_PREFIX, add_special_tokens=False)
        self._suffix_tokens = self.tokenizer.encode(RERANK_ASSISTANT_SUFFIX, add_special_tokens=False)
        self._token_true_id = self._resolve_label_token_id("yes")
        self._token_false_id = self._resolve_label_token_id("no")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _resolve_label_token_id(self, label: str) -> int:
        assert self.tokenizer is not None
        token_id = self.tokenizer.convert_tokens_to_ids(label)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"Unable to resolve reranker label token for {label!r}")
            token_id = token_ids[-1]
        return int(token_id)

    def _process_inputs(self, pairs: Sequence[tuple[str, str]]):
        assert self.tokenizer is not None

        formatted_pairs = [
            format_reranker_pair(query, passage, instruction=self.instruction)
            for query, passage in pairs
        ]
        max_pair_length = max(1, self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens))
        inputs = self.tokenizer(
            formatted_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_pair_length,
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self._prefix_tokens + input_ids + self._suffix_tokens
        padded = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        assert self.model is not None
        return {key: value.to(self.model.device) for key, value in padded.items()}

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        if not self.is_loaded:
            self.load_model()
        assert self.model is not None
        assert self._token_true_id is not None and self._token_false_id is not None

        results: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start:start + self.batch_size]
            encoded = self._process_inputs(batch_pairs)
            with torch.inference_mode():
                batch_scores = self.model(**encoded).logits[:, -1, :]
            true_vector = batch_scores[:, self._token_true_id]
            false_vector = batch_scores[:, self._token_false_id]
            logits = torch.stack([false_vector, true_vector], dim=1)
            probs = torch.nn.functional.log_softmax(logits, dim=1)[:, 1].exp()
            results.extend(float(score) for score in probs.detach().float().cpu())
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


def load_reranker_if_available(model_path: str | None = None, *, force: bool = False) -> NeuralReranker | None:
    model_path = str(model_path or config.RERANK_MODEL_PATH)
    if not force and not config.USE_RERANKER:
        return None
    if not Path(model_path).exists():
        return None
    return NeuralReranker(model_path=model_path)
