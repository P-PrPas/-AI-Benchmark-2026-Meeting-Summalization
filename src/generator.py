from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config
from .prompting import (
    ANSWER_PROFILE_FACT,
    ANSWER_PROFILE_LIST,
    ANSWER_PROFILE_SYNTHESIS,
    FACT_REWRITE_SYSTEM_PROMPT,
    NO_ANSWER_TEXT,
    REF_ARBITER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    answer_needs_retry,
    build_fact_rewrite_prompt,
    build_ref_arbiter_prompt,
    build_user_prompt,
    context_limit_for_profile,
    detect_answer_profile,
    sanitize_generated_answer,
)


@dataclass(frozen=True)
class DecodeProfile:
    name: str
    max_new_tokens: int
    repetition_penalty: float
    do_sample: bool = False


DECODE_PROFILES = {
    ANSWER_PROFILE_FACT: DecodeProfile(
        name=ANSWER_PROFILE_FACT,
        max_new_tokens=config.FACT_MAX_NEW_TOKENS,
        repetition_penalty=config.DEFAULT_REPETITION_PENALTY,
    ),
    ANSWER_PROFILE_LIST: DecodeProfile(
        name=ANSWER_PROFILE_LIST,
        max_new_tokens=config.AGGREGATE_MAX_NEW_TOKENS,
        repetition_penalty=config.DEFAULT_REPETITION_PENALTY,
    ),
    ANSWER_PROFILE_SYNTHESIS: DecodeProfile(
        name=ANSWER_PROFILE_SYNTHESIS,
        max_new_tokens=config.SYNTHESIS_MAX_NEW_TOKENS,
        repetition_penalty=config.DEFAULT_REPETITION_PENALTY,
    ),
}
STRICT_RETRY_PROFILE = DecodeProfile(
    name="strict_retry",
    max_new_tokens=config.STRICT_RETRY_MAX_NEW_TOKENS,
    repetition_penalty=config.STRICT_REPETITION_PENALTY,
)


class Generator:
    """LLM generator for deterministic Thai RAG QA answers."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        system_prompt: str | None = SYSTEM_PROMPT,
    ):
        self.system_prompt = system_prompt
        self.model = None
        self.tokenizer = None
        self.model_path = model_path if model_path is not None else config.LLM_MODEL_PATH

    def load_model(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            print("No model path specified. Using mock generation.")
            return

        print(f"Loading generator model from {model_path}...")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model_path = Path(model_path)
        if not model_path.exists():
            available = []
            if model_path.parent.exists():
                available = sorted(path.name for path in model_path.parent.iterdir() if path.is_dir())
            raise FileNotFoundError(
                f"LLM model path not found: {model_path}. "
                f"Available model directories: {available}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = True
        print("Generator model loaded successfully")

    def detect_profile(self, query: str, paragraphs: List[Dict]) -> str:
        return detect_answer_profile(query, paragraphs)

    def select_context(self, query: str, paragraphs: List[Dict], profile: str | None = None) -> List[Dict]:
        profile = profile or self.detect_profile(query, paragraphs)
        limit = context_limit_for_profile(profile)
        return list(paragraphs[:limit])

    def build_prompt(self, query: str, paragraphs: List[Dict], profile: str | None = None) -> str:
        profile = profile or self.detect_profile(query, paragraphs)
        selected = self.select_context(query, paragraphs, profile=profile)
        return build_user_prompt(
            selected,
            query,
            profile=profile,
            primary_count=context_limit_for_profile(profile),
        )

    def _resolve_max_seq_len(self, max_seq_len: Optional[int] = None) -> int:
        if max_seq_len is not None:
            return max_seq_len
        if self.tokenizer is None:
            return config.GENERATOR_MAX_SEQ_LEN
        model_max_length = getattr(self.tokenizer, "model_max_length", config.GENERATOR_MAX_SEQ_LEN)
        if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 1_000_000:
            model_max_length = config.GENERATOR_MAX_SEQ_LEN
        return min(model_max_length, config.GENERATOR_MAX_SEQ_LEN)

    def _generate_once(
        self,
        prompt: str,
        *,
        decode_profile: DecodeProfile,
        max_seq_len: Optional[int] = None,
        system_prompt: str | None = None,
    ) -> str:
        assert self.tokenizer is not None and self.model is not None
        messages = []
        active_system_prompt = self.system_prompt if system_prompt is None else system_prompt
        if active_system_prompt:
            messages.append({"role": "system", "content": active_system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._resolve_max_seq_len(max_seq_len),
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=decode_profile.max_new_tokens,
                do_sample=decode_profile.do_sample,
                repetition_penalty=decode_profile.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

    def _generate_many(
        self,
        prompts: Sequence[str],
        *,
        decode_profile: DecodeProfile,
        max_seq_len: Optional[int] = None,
    ) -> List[str]:
        assert self.tokenizer is not None and self.model is not None
        if not prompts:
            return []

        rendered_prompts = []
        for prompt in prompts:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            rendered_prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._resolve_max_seq_len(max_seq_len),
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=decode_profile.max_new_tokens,
                do_sample=decode_profile.do_sample,
                repetition_penalty=decode_profile.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_width = inputs.input_ids.shape[1]
        return [
            self.tokenizer.decode(output[prompt_width:], skip_special_tokens=True)
            for output in outputs
        ]

    def generate(
        self,
        query: str,
        paragraphs: List[Dict],
        *,
        profile: str | None = None,
        max_seq_len: Optional[int] = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return self._mock_generate(query, paragraphs)

        profile = profile or self.detect_profile(query, paragraphs)
        prompt = self.build_prompt(query, paragraphs, profile=profile)
        raw_response = self._generate_once(
            prompt,
            decode_profile=DECODE_PROFILES[profile],
            max_seq_len=max_seq_len,
        )
        sanitized = sanitize_generated_answer(raw_response)
        if answer_needs_retry(raw_response, sanitized, profile):
            raw_response = self._generate_once(
                prompt,
                decode_profile=STRICT_RETRY_PROFILE,
                max_seq_len=max_seq_len,
            )
            sanitized = sanitize_generated_answer(raw_response)
        if profile == ANSWER_PROFILE_FACT:
            sanitized = self.rewrite_fact_answer(query, paragraphs, sanitized, max_seq_len=max_seq_len)
        return sanitized or NO_ANSWER_TEXT

    def _mock_generate(self, query: str, paragraphs: List[Dict]) -> str:
        if not paragraphs:
            return NO_ANSWER_TEXT
        best = next((item.get("text", "").strip() for item in paragraphs if item.get("text", "").strip()), "")
        return sanitize_generated_answer(best) or NO_ANSWER_TEXT

    def batch_generate(
        self,
        queries: List[str],
        paragraphs_list: List[List[Dict]],
        *,
        profile: str | None = None,
        max_seq_len: Optional[int] = None,
    ) -> List[str]:
        if self.model is None or self.tokenizer is None:
            return [self._mock_generate(q, p) for q, p in zip(queries, paragraphs_list)]

        resolved_profiles = [
            profile or self.detect_profile(query, paragraphs)
            for query, paragraphs in zip(queries, paragraphs_list)
        ]
        if not resolved_profiles:
            return []
        if len(set(resolved_profiles)) != 1:
            return [
                self.generate(query, paragraphs, profile=item_profile, max_seq_len=max_seq_len)
                for query, paragraphs, item_profile in zip(queries, paragraphs_list, resolved_profiles)
            ]

        resolved_profile = resolved_profiles[0]
        prompts = [
            self.build_prompt(query, paragraphs, profile=resolved_profile)
            for query, paragraphs in zip(queries, paragraphs_list)
        ]
        raw_responses = self._generate_many(
            prompts,
            decode_profile=DECODE_PROFILES[resolved_profile],
            max_seq_len=max_seq_len,
        )
        sanitized = [sanitize_generated_answer(response) for response in raw_responses]

        retry_indices = [
            index
            for index, (raw_response, clean_response) in enumerate(zip(raw_responses, sanitized))
            if answer_needs_retry(raw_response, clean_response, resolved_profile)
        ]
        if retry_indices:
            retry_prompts = [prompts[index] for index in retry_indices]
            retry_outputs = self._generate_many(
                retry_prompts,
                decode_profile=STRICT_RETRY_PROFILE,
                max_seq_len=max_seq_len,
            )
            for index, raw_response in zip(retry_indices, retry_outputs):
                sanitized[index] = sanitize_generated_answer(raw_response)

        if resolved_profile == ANSWER_PROFILE_FACT:
            sanitized = [
                self.rewrite_fact_answer(query, paragraphs, answer, max_seq_len=max_seq_len)
                for query, paragraphs, answer in zip(queries, paragraphs_list, sanitized)
            ]

        return [answer or NO_ANSWER_TEXT for answer in sanitized]

    @staticmethod
    def _tokenize_for_overlap(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

    def _source_overlap(self, answer: str, paragraphs: Sequence[Dict]) -> float:
        answer_tokens = set(self._tokenize_for_overlap(answer))
        if not answer_tokens:
            return 0.0
        evidence_tokens = set()
        for paragraph in paragraphs:
            evidence_tokens.update(self._tokenize_for_overlap(paragraph.get("text", "")))
        if not evidence_tokens:
            return 0.0
        return len(answer_tokens & evidence_tokens) / max(1, len(answer_tokens))

    def should_rewrite_fact_answer(self, query: str, paragraphs: Sequence[Dict], answer: str) -> bool:
        if not config.ENABLE_FACT_ANSWER_REWRITE:
            return False
        answer = (answer or "").strip()
        if not answer or answer == NO_ANSWER_TEXT:
            return False
        if len(answer) >= config.FACT_REWRITE_TRIGGER_CHARS:
            return True
        overlap = self._source_overlap(answer, paragraphs)
        if overlap < config.FACT_REWRITE_MIN_SOURCE_OVERLAP:
            return True
        conjunction_count = sum(answer.count(token) for token in (" และ", " รวมทั้ง", " พร้อมทั้ง", " ซึ่ง"))
        return conjunction_count >= 2

    def rewrite_fact_answer(
        self,
        query: str,
        paragraphs: Sequence[Dict],
        answer: str,
        *,
        max_seq_len: Optional[int] = None,
    ) -> str:
        if not self.should_rewrite_fact_answer(query, paragraphs, answer):
            return answer
        prompt = build_fact_rewrite_prompt(query, list(paragraphs[:2]), answer)
        raw_response = self._generate_once(
            prompt,
            decode_profile=DecodeProfile(
                name="fact_rewrite",
                max_new_tokens=config.FACT_REWRITE_MAX_NEW_TOKENS,
                repetition_penalty=config.STRICT_REPETITION_PENALTY,
            ),
            max_seq_len=max_seq_len,
            system_prompt=FACT_REWRITE_SYSTEM_PROMPT,
        )
        rewritten = sanitize_generated_answer(raw_response)
        if not rewritten or rewritten == NO_ANSWER_TEXT:
            return answer
        return rewritten

    def arbitrate_references(
        self,
        query: str,
        candidate_paragraphs: Sequence[Dict],
        *,
        profile: str,
        rule_refs: Sequence[str],
        max_seq_len: Optional[int] = None,
    ) -> List[str] | None:
        if self.model is None or self.tokenizer is None:
            return None
        if not candidate_paragraphs:
            return None
        prompt = build_ref_arbiter_prompt(
            query,
            candidate_paragraphs,
            profile=profile,
            rule_refs=rule_refs,
        )
        raw_response = self._generate_once(
            prompt,
            decode_profile=DecodeProfile(
                name="ref_arbiter",
                max_new_tokens=config.REF_ARBITER_MAX_NEW_TOKENS,
                repetition_penalty=config.STRICT_REPETITION_PENALTY,
            ),
            max_seq_len=max_seq_len,
            system_prompt=REF_ARBITER_SYSTEM_PROMPT,
        )
        candidate_ids = {paragraph["para_id"] for paragraph in candidate_paragraphs}
        parsed = []
        for para_id in re.findall(r"\bP\d+\b", raw_response):
            if para_id in candidate_ids and para_id not in parsed:
                parsed.append(para_id)
        return parsed or None


class ThaiSummarizer:
    def __init__(self, generator: Generator):
        self.generator = generator

    def summarize(
        self,
        query: str,
        paragraphs: List[Dict],
        referenced_ids: List[str] | None = None,
    ) -> str:
        return self.generator.generate(query, paragraphs)

    def summarize_with_references(
        self,
        query: str,
        paragraphs: List[Dict],
    ) -> tuple[str, List[str]]:
        abstractive = self.generator.generate(query, paragraphs)
        refs = [p["para_id"] for p in paragraphs[: config.REFERENCE_TOP_N]]
        return abstractive, refs


if __name__ == "__main__":
    gen = Generator()
    test_paragraphs = [
        {"para_id": "P1", "text": "บันทึกการประชุม"},
        {"para_id": "P5", "text": "ห้องประชุมกรรมาธิการ N 404 ชั้น 4 อาคารรัฐสภา"},
    ]
    prompt = gen.build_prompt("การประชุมจัดที่ไหน", test_paragraphs)
    print(prompt)
