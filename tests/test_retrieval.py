import unittest

from src.prompting import ANSWER_PROFILE_FACT, ANSWER_PROFILE_LIST, ANSWER_PROFILE_SYNTHESIS
from src.retrieval import (
    ReferenceSelectionConfig,
    compress_evidence,
    rewrite_query_heuristic,
    select_references_from_retrieved,
)


class RetrievalTests(unittest.TestCase):
    def test_dynamic_ref_selection_keeps_single_fact_ref(self):
        retrieved = [
            {"para_id": "P1", "score": 0.95, "selection_score": 0.95, "text": "นายกฯ ชี้แจงต่อที่ประชุม"},
            {"para_id": "P2", "score": 0.20, "selection_score": 0.20, "text": "ข้อมูลประกอบอื่น"},
            {"para_id": "P3", "score": 0.10, "selection_score": 0.10, "text": "ข้อมูลรอง"},
        ]
        refs = select_references_from_retrieved(
            retrieved,
            profile=ANSWER_PROFILE_FACT,
            calibration_config=ReferenceSelectionConfig(),
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
        )
        self.assertEqual(refs, ["P1", "P2", "P3"])

    def test_compress_evidence_for_fact_returns_shorter_context(self):
        paragraphs = [
            {
                "para_id": "P7",
                "text": "ประธานการประชุมคือ นายสมชาย ใจดี ซึ่งทำหน้าที่เปิดการประชุมอย่างเป็นทางการ "
                "หลังจากนั้นที่ประชุมได้พิจารณาระเบียบวาระอื่นต่อ",
            }
        ]
        compressed = compress_evidence("ประธานการประชุมคือใคร", paragraphs, ANSWER_PROFILE_FACT)
        self.assertEqual(len(compressed), 1)
        self.assertIn("ประธานการประชุม", compressed[0]["text"])
        self.assertLessEqual(len(compressed[0]["text"]), len(paragraphs[0]["text"]))

    def test_rewrite_query_heuristic_expands_abbreviation(self):
        rewritten = rewrite_query_heuristic("กมธ. เชิญใครบ้าง")
        self.assertIn("คณะกรรมาธิการ", rewritten)
        self.assertNotIn("กมธ.", rewritten)


if __name__ == "__main__":
    unittest.main()
