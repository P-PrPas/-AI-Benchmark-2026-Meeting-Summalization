import unittest

from src.prompting import (
    ANSWER_PROFILE_FACT,
    ANSWER_PROFILE_LIST,
    ANSWER_PROFILE_SYNTHESIS,
    build_fact_rewrite_prompt,
    build_ref_arbiter_prompt,
    build_user_prompt,
    detect_answer_profile,
    sanitize_generated_answer,
)


class PromptingTests(unittest.TestCase):
    def test_build_user_prompt_uses_ranked_blocks(self):
        paragraphs = [
            {"para_id": "P10", "text": "ที่ประชุมมีมติรับทราบรายงานผลการดำเนินงาน"},
            {"para_id": "P11", "text": "ที่ประชุมมอบหมายให้ฝ่ายเลขานุการจัดทำหนังสือติดตามผล"},
        ]
        prompt = build_user_prompt(paragraphs, "ที่ประชุมมีมติอย่างไร", profile=ANSWER_PROFILE_FACT)
        self.assertIn("ข้อมูลอ้างอิงที่เกี่ยวข้องมากที่สุด", prompt)
        self.assertIn("1. [P10]", prompt)
        self.assertIn("ข้อมูลด้านบนเรียงจากเกี่ยวข้องมากไปน้อย", prompt)
        self.assertIn("วัตถุประสงค์ของการประชุมครั้งนี้", prompt)
        self.assertNotIn("ยานยนต์ไฟฟ้า", prompt)

    def test_sanitize_generated_answer_removes_tags_prefix_and_ellipsis(self):
        raw = "คำตอบ: [P12] ที่ประชุมมีมติรับทราบรายงานผลการดำเนินงาน...\n[P12] ที่ประชุมมีมติรับทราบรายงานผลการดำเนินงาน"
        sanitized = sanitize_generated_answer(raw)
        self.assertNotIn("[P12]", sanitized)
        self.assertFalse(sanitized.startswith("คำตอบ:"))
        self.assertFalse(sanitized.endswith("..."))
        self.assertEqual(sanitized, "ที่ประชุมมีมติรับทราบรายงานผลการดำเนินงาน")

    def test_detect_answer_profile_routes_list(self):
        paragraphs = [{"para_id": "P1", "text": "1. กระทรวงมหาดไทย 2. กรมป้องกันและบรรเทาสาธารณภัย"}]
        profile = detect_answer_profile("คณะกรรมาธิการเชิญหน่วยงานใดมาให้ข้อมูลบ้าง", paragraphs)
        self.assertEqual(profile, ANSWER_PROFILE_LIST)

    def test_detect_answer_profile_routes_synthesis(self):
        paragraphs = [
            {"para_id": "P1", "text": "คณะกรรมาธิการเห็นว่าควรพัฒนาโครงสร้างพื้นฐานการชาร์จไฟฟ้า"},
            {"para_id": "P2", "text": "ควรสนับสนุนผู้ประกอบการในประเทศและพัฒนาบุคลากร"},
            {"para_id": "P3", "text": "ควรกำหนดมาตรฐานความปลอดภัยและการจัดการซากแบตเตอรี่"},
            {"para_id": "P4", "text": "ต้องดำเนินการอย่างต่อเนื่องและบูรณาการหลายหน่วยงาน"},
            {"para_id": "P5", "text": "ที่ประชุมย้ำว่าควรมีมาตรการกำกับดูแลควบคู่กัน"},
            {"para_id": "P6", "text": "ควรผลักดันการลงทุนอย่างเป็นระบบ"},
        ]
        profile = detect_answer_profile("คณะกรรมาธิการมีข้อสังเกตอย่างไรเกี่ยวกับการส่งเสริมยานยนต์ไฟฟ้า", paragraphs)
        self.assertEqual(profile, ANSWER_PROFILE_SYNTHESIS)

    def test_detect_answer_profile_keeps_fact_even_with_many_paragraphs(self):
        paragraphs = [
            {"para_id": f"P{i}", "text": f"ย่อหน้าที่ {i} กล่าวถึงการดำเนินงานต่อเนื่องของหน่วยงาน"} for i in range(1, 7)
        ]
        profile = detect_answer_profile("ประธานการประชุมคือใคร", paragraphs)
        self.assertEqual(profile, ANSWER_PROFILE_FACT)

    def test_build_user_prompt_for_list_keeps_numbered_example(self):
        paragraphs = [{"para_id": "P1", "text": "1. กระทรวงมหาดไทย 2. กรมป้องกันและบรรเทาสาธารณภัย"}]
        prompt = build_user_prompt(paragraphs, "มีหน่วยงานใดบ้าง", profile=ANSWER_PROFILE_LIST)
        self.assertIn("1. ปลัดกระทรวงมหาดไทย", prompt)
        self.assertNotIn("สถานการณ์อุทกภัยในพื้นที่เริ่มคลี่คลายลง", prompt)

    def test_build_ref_arbiter_prompt_lists_candidate_ids_only(self):
        prompt = build_ref_arbiter_prompt(
            "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¹ƒà¸„à¸£",
            [
                {"para_id": "P10", "text": "à¸›à¸£à¸°à¸˜à¸²à¸™à¸à¸²à¸£à¸›à¸£à¸°à¸Šà¸¸à¸¡à¸„à¸·à¸­à¸™à¸²à¸¢à¸ªà¸¡à¸Šà¸²à¸¢ à¹ƒà¸ˆà¸”à¸µ"},
                {"para_id": "P11", "text": "à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¸£à¸°à¸à¸­à¸š"},
            ],
            profile=ANSWER_PROFILE_FACT,
            rule_refs=["P10"],
        )
        self.assertIn("[P10]", prompt)
        self.assertIn("[P11]", prompt)
        self.assertIn("Pxx,Pyy", prompt)

    def test_build_fact_rewrite_prompt_includes_evidence_and_draft(self):
        prompt = build_fact_rewrite_prompt(
            "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¹ƒà¸„à¸£",
            [{"para_id": "P10", "text": "à¸›à¸£à¸°à¸˜à¸²à¸™à¸à¸²à¸£à¸›à¸£à¸°à¸Šà¸¸à¸¡à¸„à¸·à¸­à¸™à¸²à¸¢à¸ªà¸¡à¸Šà¸²à¸¢ à¹ƒà¸ˆà¸”à¸µ"}],
            "à¸›à¸£à¸°à¸˜à¸²à¸™à¸„à¸·à¸­à¸™à¸²à¸¢à¸ªà¸¡à¸Šà¸²à¸¢ à¹ƒà¸ˆà¸”à¸µ à¹à¸¥à¸°à¸”à¸³à¹€à¸™à¸´à¸™à¸à¸²à¸£à¸•à¹ˆà¸­",
        )
        self.assertIn("[P10]", prompt)
        self.assertIn("draft answer", prompt)

    def test_detect_answer_profile_keeps_fact_with_single_numbered_context(self):
        paragraphs = [
            {"para_id": "P1", "text": "1. à¸™à¸²à¸¢à¸ªà¸¡à¸Šà¸²à¸¢ à¹ƒà¸ˆà¸”à¸µ à¸—à¸³à¸«à¸™à¹‰à¸²à¸—à¸µà¹ˆà¸›à¸£à¸°à¸˜à¸²à¸™à¸à¸²à¸£à¸›à¸£à¸°à¸Šà¸¸à¸¡"},
            {"para_id": "P2", "text": "à¸—à¸µà¹ˆà¸›à¸£à¸°à¸Šà¸¸à¸¡à¸”à¸³à¹€à¸™à¸´à¸™à¸£à¸²à¸¢à¸à¸²à¸£à¸•à¹ˆà¸­à¹„à¸›"},
        ]
        profile = detect_answer_profile("à¸›à¸£à¸°à¸˜à¸²à¸™à¸à¸²à¸£à¸›à¸£à¸°à¸Šà¸¸à¸¡à¸„à¸·à¸­à¹ƒà¸„à¸£", paragraphs)
        self.assertEqual(profile, ANSWER_PROFILE_FACT)


if __name__ == "__main__":
    unittest.main()
