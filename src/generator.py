import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from config import LLM_MODEL_PATH
from prompting import NO_ANSWER_TEXT, NO_CONTEXT_TEXT, SYSTEM_PROMPT, build_user_prompt, sanitize_generated_answer


DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_USER_TEMPLATE = None


class Generator:
    """LLM generator for abstractive summaries."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: Optional[str] = DEFAULT_USER_TEMPLATE,
    ):
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model = None
        self.tokenizer = None
        self.model_path = model_path if model_path is not None else LLM_MODEL_PATH

    def load_model(self, model_path: Optional[str] = None):
        """Load LLM model and tokenizer."""
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            print("No model path specified. Using mock generation.")
            return

        print(f"Loading generator model from {model_path}...")
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
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("Generator model loaded successfully")

    def build_prompt(self, query: str, paragraphs: List[Dict]) -> str:
        """Build prompt from query and retrieved paragraphs."""
        if not paragraphs:
            paragraphs_text = NO_CONTEXT_TEXT
        else:
            paragraphs_text = "\n".join(f"[{p['para_id']}] {p['text']}" for p in paragraphs)

        if self.user_template:
            return self.user_template.format(paragraphs_text=paragraphs_text, query=query)
        return build_user_prompt(paragraphs_text, query)

    def generate(
        self,
        query: str,
        paragraphs: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate abstractive summary for a query."""
        if self.model is None or self.tokenizer is None:
            return self._mock_generate(query, paragraphs)

        prompt = self.build_prompt(query, paragraphs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return sanitize_generated_answer(response)

    def _mock_generate(self, query: str, paragraphs: List[Dict]) -> str:
        """Mock generation for testing without LLM."""
        return NO_ANSWER_TEXT

    def batch_generate(
        self,
        queries: List[str],
        paragraphs_list: List[List[Dict]],
        **kwargs,
    ) -> List[str]:
        """Generate summaries for multiple queries at once."""
        return [self.generate(q, p, **kwargs) for q, p in zip(queries, paragraphs_list)]


class ThaiSummarizer:
    """High-level summarizer with Thai-specific optimizations."""

    def __init__(self, generator: Generator):
        self.generator = generator

    def summarize(
        self,
        query: str,
        paragraphs: List[Dict],
        referenced_ids: List[str] = None,
    ) -> str:
        """Generate summary for query with optional reference hinting."""
        return self.generator.generate(query, paragraphs)

    def summarize_with_references(
        self,
        query: str,
        paragraphs: List[Dict],
    ) -> tuple[str, List[str]]:
        """Generate summary and select references simultaneously."""
        abstractive = self.generator.generate(query, paragraphs)
        refs = [p["para_id"] for p in paragraphs[:3]]
        return abstractive, refs


if __name__ == "__main__":
    gen = Generator()
    test_paragraphs = [
        {"para_id": "P1", "text": "บันทึกการประชุม"},
        {"para_id": "P5", "text": "ห้องประชุมกรรมาธิการ N 404 ชั้น 4 อาคารรัฐสภา"},
    ]
    prompt = gen.build_prompt("การประชุมจัดที่ไหน", test_paragraphs)
    print(prompt)
