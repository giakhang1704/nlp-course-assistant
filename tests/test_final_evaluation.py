"""Automatic Part IV scoring rules (no LLM calls)."""
import json
import unittest

from course_assistant.config import PROJECT_ROOT
from course_assistant.evaluation.final import has_fact, judge, read_jsonl


def case(status="retrieved", must=("60",), groups=(("course_info.midterm",),), parser_expected=True):
    return {"expected_status": status, "must_contain": list(must),
            "evidence_groups": [list(g) for g in groups], "parser_expected": parser_expected}


def trace(status="answered", answer="Midterm 60 phút [C1]", cited=("course_info.midterm",),
          retrieved=("course_info.midterm",), attempts=1):
    return {"status": status, "answer": answer, "generation_attempts": attempts,
            "citations": [{"document_id": d} for d in cited],
            "retrieved_chunks": [{"document_id": d} for d in retrieved]}


class FinalEvaluationTests(unittest.TestCase):
    def test_fact_matching_respects_number_and_word_boundaries(self):
        self.assertTrue(has_fact("Thuộc Chương 1.", "chương 1|chapter 1"))
        self.assertFalse(has_fact("Thuộc Chương 10.", "chương 1|chapter 1"))
        self.assertTrue(has_fact("adds probabilities", "probabilit|xác suất"))
        self.assertFalse(has_fact("có 150 giờ", "15"))
        self.assertTrue(has_fact("CHIẾM 15%", "15"))

    def test_answer_needs_status_fact_and_gold_citation(self):
        self.assertEqual((True, None), judge(case(), trace()))
        self.assertEqual("LLM trả lời thiếu/sai ý chính", judge(case(), trace(answer="Midterm 90 phút"))[1])
        self.assertEqual("citation không trỏ tới nguồn đúng", judge(case(), trace(cited=["course_info.final"]))[1])

    def test_echoed_question_is_not_an_answer(self):
        yes_no = case(must=("hỏi đáp|procedural",), groups=())
        echo = trace(answer="Theo dữ liệu mẫu của project: Chương 11 có hỏi đáp không? [C1] [C2]")
        echo["question"] = "Chương 11 có hỏi đáp không?"
        self.assertEqual("LLM lặp lại câu hỏi, không trả lời", judge(yes_no, echo)[1])
        real = trace(answer="Chương 11 liên quan đến hỏi đáp (question answering) [C2]")
        real["question"] = "Chương 11 có liên quan đến hỏi đáp không?"
        self.assertTrue(judge(yes_no, real)[0])

    def test_empty_refusal_is_scored_as_refusal_not_technical_error(self):
        empty = trace(status="generation_error", answer="")
        empty["error"] = "llm_empty_answer"
        self.assertEqual("LLM từ chối dù context có thông tin", judge(case(), empty)[1])

    def test_failure_cause_follows_the_first_failing_stage(self):
        self.assertEqual("parser không nhận câu",
                         judge(case(), trace(status="insufficient_information"), {"is_valid": False})[1])
        self.assertEqual("truy xuất trượt tài liệu đúng",
                         judge(case(), trace(status="insufficient_information", retrieved=["faq.q01"]))[1])
        self.assertEqual("evidence policy chặn trước LLM",
                         judge(case(), trace(status="insufficient_information", attempts=0))[1])
        self.assertEqual("LLM từ chối dù context có thông tin",
                         judge(case(), trace(status="insufficient_information"))[1])

    def test_refusal_and_clarification_cases(self):
        missing = case("insufficient_information", must=(), groups=())
        self.assertTrue(judge(missing, trace(status="insufficient_information"))[0])
        self.assertEqual("trả lời câu không có thông tin (hallucination)", judge(missing, trace())[1])
        clarify = case("clarification_needed", must=(), groups=())
        self.assertTrue(judge(clarify, trace(status="clarification_needed"))[0])
        self.assertEqual("tự đoán đối tượng thay vì hỏi lại", judge(clarify, trace())[1])

    def test_final_set_is_consistent(self):
        root = PROJECT_ROOT / "data/evaluation/final"
        parser_cases = read_jsonl(root / "parser_cases.jsonl")
        qa_cases = read_jsonl(root / "qa_cases.jsonl")
        self.assertEqual(81, len(parser_cases))
        self.assertEqual(34, sum(c["source"] == "official" for c in parser_cases))
        questions = (PROJECT_ROOT / "input/questions_eval.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual([c["question"] for c in qa_cases], questions)
        for c in qa_cases:
            if c["expected_status"] == "retrieved":
                self.assertTrue(c["must_contain"] and c["evidence_groups"], c["id"])
        for d in read_jsonl(root / "dialogues.jsonl"):
            self.assertEqual(len(d["turns"]), len(d["gold_turns"]), d["session_id"])


if __name__ == "__main__":
    unittest.main()
