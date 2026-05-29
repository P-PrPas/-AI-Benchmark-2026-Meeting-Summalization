import unittest

from finetune.common import build_source_anchored_fact_answer


class FinetuneCommonTests(unittest.TestCase):
    def test_build_source_anchored_fact_answer_prefers_evidence_sentence(self):
        answer = build_source_anchored_fact_answer(
            "ประธานการประชุมคือใคร",
            "ประธานการประชุมคือ นายสมชาย ใจดี",
            [
                {"para_id": "P1", "text": "ประธานการประชุมคือ นายสมชาย ใจดี และได้เปิดการประชุมตามระเบียบวาระ"},
                {"para_id": "P2", "text": "ที่ประชุมดำเนินการเรื่องอื่นต่อ"},
            ],
        )
        self.assertIsNotNone(answer)
        self.assertIn("นายสมชาย ใจดี", answer)


if __name__ == "__main__":
    unittest.main()
