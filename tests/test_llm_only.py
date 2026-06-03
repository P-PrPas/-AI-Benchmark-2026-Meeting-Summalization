import unittest

from src.llm_only import (
    build_llm_only_prompt,
    build_llm_only_target,
    consensus_candidate_score,
    parse_llm_only_output,
    resolve_decode_spec,
)


class LLMOnlyTests(unittest.TestCase):
    def test_parse_json_output_filters_invalid_refs(self):
        parsed = parse_llm_only_output(
            '{"abstractive":"answer text","refs":["P1","P99","P2"]}',
            ["P1", "P2"],
        )
        self.assertEqual(parsed.abstractive, "answer text")
        self.assertEqual(parsed.refs, ["P1", "P2"])
        self.assertEqual(parsed.invalid_refs, ["P99"])
        self.assertFalse(parsed.parse_error)

    def test_parse_text_output_extracts_refs_and_sanitizes_answer(self):
        parsed = parse_llm_only_output(
            "คำตอบ: [P3] answer text\nrefs: P3,P4",
            ["P3", "P4"],
        )
        self.assertEqual(parsed.abstractive, "answer text")
        self.assertEqual(parsed.refs, ["P3", "P4"])
        self.assertTrue(parsed.parse_error)

    def test_parse_missing_refs_does_not_crash(self):
        parsed = parse_llm_only_output("answer only", ["P1"])
        self.assertEqual(parsed.abstractive, "answer only")
        self.assertEqual(parsed.refs, [])
        self.assertTrue(parsed.parse_error)

    def test_build_prompt_includes_full_doc_ids_and_query(self):
        prompt = build_llm_only_prompt(
            "question?",
            [{"para_id": "P1", "text": "first"}, {"para_id": "P2", "text": "second"}],
            mode="minimal",
        )
        self.assertIn("[P1] first", prompt)
        self.assertIn("[P2] second", prompt)
        self.assertIn("question?", prompt)
        self.assertIn('"abstractive"', prompt)

    def test_build_raw_target_uses_text_refs_block(self):
        target = build_llm_only_target("answer", ["P1", "P2"], mode="raw")
        self.assertEqual(target, "answer\nrefs: P1,P2")

    def test_decode_spec_supports_shorthand_and_explicit_values(self):
        temp_spec = resolve_decode_spec("temp0.2")
        self.assertEqual(temp_spec.temperature, 0.2)
        self.assertTrue(temp_spec.do_sample)
        explicit = resolve_decode_spec("sample:t=0.4:tokens=512:rp=1.0")
        self.assertEqual(explicit.temperature, 0.4)
        self.assertEqual(explicit.max_new_tokens, 512)
        self.assertEqual(explicit.repetition_penalty, 1.0)

    def test_consensus_candidate_prefers_ref_and_answer_agreement(self):
        candidates = [
            {"variant": "base", "abstractive": "alpha beta", "refs": ["P1"], "parse_error": False},
            {"variant": "sample", "abstractive": "alpha beta gamma", "refs": ["P1"], "parse_error": False},
            {"variant": "odd", "abstractive": "unrelated", "refs": ["P9"], "parse_error": False},
        ]
        scores = [consensus_candidate_score(candidate, candidates) for candidate in candidates]
        self.assertGreater(scores[0], scores[2])
        self.assertGreater(scores[1], scores[2])


if __name__ == "__main__":
    unittest.main()
