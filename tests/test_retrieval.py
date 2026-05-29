import inspect
import unittest
from unittest.mock import patch

from src import reranker as reranker_module
from src.prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST
from src.retrieval import (
    ReferenceSelectionConfig,
    build_generation_context,
    compute_arbiter_metrics,
    compress_evidence,
    rewrite_query_heuristic,
    select_references_from_retrieved,
    select_references_with_diagnostics,
)


class RetrievalTests(unittest.TestCase):
    class StubGenerator:
        def __init__(self, refs):
            self.refs = refs

        def arbitrate_references(self, query, candidate_paragraphs, *, profile, rule_refs, max_seq_len=None):
            return self.refs

    def test_dynamic_ref_selection_keeps_single_fact_ref(self):
        retrieved = [
            {"para_id": "P1", "score": 0.95, "selection_score": 0.95, "text": "ประธานชี้แจงต่อที่ประชุม"},
            {"para_id": "P2", "score": 0.20, "selection_score": 0.20, "text": "ข้อมูลประกอบอื่น"},
            {"para_id": "P3", "score": 0.10, "selection_score": 0.10, "text": "ข้อมูลรอง"},
        ]
        refs = select_references_from_retrieved(
            retrieved,
            profile=ANSWER_PROFILE_FACT,
            calibration_config=ReferenceSelectionConfig(),
            mode="dynamic",
        )
        self.assertEqual(refs, ["P1"])

    def test_dynamic_ref_selection_allows_multiple_for_list(self):
        retrieved = [
            {"para_id": "P1", "score": 0.82, "selection_score": 0.82, "text": "1. กระทรวงมหาดไทย"},
            {"para_id": "P2", "score": 0.80, "selection_score": 0.80, "text": "2. กรมป้องกันและบรรเทาสาธารณภัย"},
            {"para_id": "P3", "score": 0.78, "selection_score": 0.78, "text": "3. ผู้ว่าราชการจังหวัดเชียงราย"},
        ]
        refs = select_references_from_retrieved(
            retrieved,
            profile=ANSWER_PROFILE_LIST,
            calibration_config=ReferenceSelectionConfig(),
            mode="dynamic",
        )
        self.assertEqual(refs, ["P1", "P2", "P3"])

    def test_fixed_ref_selection_keeps_top_three(self):
        retrieved = [
            {"para_id": "P1", "score": 0.91, "selection_score": 0.91, "text": "A"},
            {"para_id": "P2", "score": 0.89, "selection_score": 0.89, "text": "B"},
            {"para_id": "P3", "score": 0.87, "selection_score": 0.87, "text": "C"},
            {"para_id": "P4", "score": 0.10, "selection_score": 0.10, "text": "D"},
        ]
        refs = select_references_from_retrieved(retrieved, n=3, mode="fixed")
        self.assertEqual(refs, ["P1", "P2", "P3"])

    def test_llm_ref_arbiter_uses_candidate_subset(self):
        retrieved = [
            {"para_id": "P1", "score": 0.82, "selection_score": 0.82, "text": "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¸™à¸²à¸¢à¸ à¸"},
            {"para_id": "P2", "score": 0.80, "selection_score": 0.80, "text": "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¸™à¸²à¸¢à¸‚ à¸‚"},
            {"para_id": "P3", "score": 0.10, "selection_score": 0.10, "text": "à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¸£à¸°à¸à¸­à¸š"},
        ]
        with patch("src.retrieval.config.ENABLE_LLM_REF_ARBITER", True), patch(
            "src.retrieval.config.REF_ARBITER_TRIGGER_MODE", "always"
        ):
            result = select_references_with_diagnostics(
                "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¹ƒà¸„à¸£",
                retrieved,
                profile=ANSWER_PROFILE_FACT,
                mode="dynamic_rules_then_llm_arbiter",
                generator=self.StubGenerator(["P2"]),
            )
        self.assertEqual(result.selected_refs, ["P2"])
        self.assertTrue(result.arbiter_used)
        self.assertFalse(result.arbiter_fallback)

    def test_llm_ref_arbiter_falls_back_on_invalid_output(self):
        retrieved = [
            {"para_id": "P1", "score": 0.82, "selection_score": 0.82, "text": "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¸™à¸²à¸¢à¸ à¸"},
            {"para_id": "P2", "score": 0.80, "selection_score": 0.80, "text": "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¸™à¸²à¸¢à¸‚ à¸‚"},
        ]
        with patch("src.retrieval.config.ENABLE_LLM_REF_ARBITER", True), patch(
            "src.retrieval.config.REF_ARBITER_TRIGGER_MODE", "always"
        ):
            result = select_references_with_diagnostics(
                "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¹ƒà¸„à¸£",
                retrieved,
                profile=ANSWER_PROFILE_FACT,
                mode="dynamic_rules_then_llm_arbiter",
                generator=self.StubGenerator(["P99"]),
            )
        self.assertEqual(result.selected_refs, result.rule_refs)
        self.assertTrue(result.arbiter_fallback)

    def test_compute_arbiter_metrics_reports_usage_and_fallback(self):
        metrics = compute_arbiter_metrics(
            [
                select_references_with_diagnostics("", [], profile=ANSWER_PROFILE_FACT),
                type("Selection", (), {
                    "arbiter_triggered": True,
                    "arbiter_used": True,
                    "arbiter_fallback": False,
                })(),
                type("Selection", (), {
                    "arbiter_triggered": True,
                    "arbiter_used": True,
                    "arbiter_fallback": True,
                })(),
            ]
        )
        self.assertGreater(metrics["arbiter_usage_rate"], 0.0)
        self.assertGreater(metrics["arbiter_fallback_rate"], 0.0)

    def test_compress_evidence_for_fact_returns_shorter_context(self):
        paragraphs = [
            {
                "para_id": "P7",
                "text": (
                    "ประธานการประชุมคือ นายสมชาย ใจดี ซึ่งทำหน้าที่เปิดการประชุมอย่างเป็นทางการ "
                    "หลังจากนั้นที่ประชุมได้พิจารณาระเบียบวาระอื่นต่อ"
                ),
            }
        ]
        compressed = compress_evidence("ประธานการประชุมคือใคร", paragraphs, ANSWER_PROFILE_FACT)
        self.assertEqual(len(compressed), 1)
        self.assertIn("ประธานการประชุม", compressed[0]["text"])
        self.assertLessEqual(len(compressed[0]["text"]), len(paragraphs[0]["text"]))

    def test_build_generation_context_skips_compression_when_disabled(self):
        reranked = [
            {"para_id": "P1", "text": "ประธานคือ นายสมชาย ใจดี และดำเนินการประชุมต่อไป", "score": 0.9},
            {"para_id": "P2", "text": "ข้อมูลเพิ่มเติม", "score": 0.2},
        ]
        with patch("src.retrieval.config.ENABLE_EVIDENCE_COMPRESSION", False):
            context = build_generation_context(
                "ประธานการประชุมคือใคร",
                reranked,
                ["P1"],
                ANSWER_PROFILE_FACT,
            )
        self.assertEqual(context[0]["text"], reranked[0]["text"])

    def test_rewrite_query_heuristic_expands_abbreviation(self):
        rewritten = rewrite_query_heuristic("กมธ. เชิญใครบ้าง")
        self.assertIn("คณะกรรมาธิการ", rewritten)
        self.assertNotIn("กมธ.", rewritten)

    def test_reranker_wrapper_does_not_use_sequence_classification_path(self):
        source = inspect.getsource(reranker_module)
        self.assertIn("AutoModelForCausalLM", source)
        self.assertNotIn("AutoModelForSequenceClassification", source)


if __name__ == "__main__":
    unittest.main()
