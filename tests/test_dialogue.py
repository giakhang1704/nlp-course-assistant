"""Part III contracts with real KB/retrieval and an explicitly fake generator."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import unicodedata
from unittest.mock import patch

from course_assistant.config import PROJECT_ROOT
from course_assistant.rag.qa import CourseAssistant, load_rag_config
from course_assistant.retrieval.index import build_index
from course_assistant.generation.llm_client import LLMError
from course_assistant.dialogue.manager import DialogueAssistant
from course_assistant.dialogue.exporter import run_dialogues, validate_dialogues


class FixtureGenerator:
    model = "offline-dialogue-fixture"

    def __init__(self):
        self.calls = 0
        self.failure = None

    def generate(self, system, question, context):
        self.calls += 1
        if self.failure:
            raise LLMError(self.failure)
        labels = [json.loads(line)["citation"] for line in context.splitlines()]
        return {"status": "answered", "answer": "Fixture " + "".join(f"[{c}]" for c in labels), "citations": labels}


class DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        for relative in ["data/raw", "data/slide_content"]:
            shutil.copytree(PROJECT_ROOT / relative, cls.root / relative)
        prompt = Path("src/course_assistant/generation/prompts/answer.txt")
        (cls.root / prompt).parent.mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT / prompt, cls.root / prompt)
        config = load_rag_config()
        build_index(cls.root, config)
        cls.client = FixtureGenerator()
        cls.qa = CourseAssistant(root=cls.root, config=config, client=cls.client)
        cls.manager = DialogueAssistant(cls.qa)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.manager.sessions.clear()
        self.client.calls, self.client.failure = 0, None

    def ask(self, question, session="one", retrieve_only=False):
        return self.manager.ask(question, session, retrieve_only)

    def test_short_algorithm_alias_establishes_grounded_focus_for_purpose_followup(self):
        first = self.ask("top-down là gì")
        self.assertEqual("answered", first["status"])
        focus = first["state_after"]["focus"]
        self.assertEqual(["TOP_DOWN_PARSER"], [e["id"] for e in focus])
        self.assertIn("slide_ch03_top_down", focus[0]["source_document_ids"])
        followup = self.ask("Thuật toán này dùng để làm gì")
        self.assertEqual("answered", followup["status"])
        self.assertEqual("top-down dùng để làm gì", followup["effective_question"])
        self.assertEqual("TOP_DOWN_PARSER", followup["context_used"][0]["id"])
        self.assertEqual("explanation", followup["effective_parse_result"]["semantic_frame"]["question_type"])
        self.assertIn("slide_ch03_top_down", [h["document_id"] for h in followup["retrieved_chunks"]])

    def test_part_reference_prefers_the_assignment_part_in_focus(self):
        self.ask("Bài tập lớn có phần parser không?")
        result = self.ask("Phần này cần xuất file gì?")
        self.assertEqual("answered", result["status"])
        self.assertEqual(["PART_I"], [e["id"] for e in result["context_used"]])

    def test_chapter_reference_after_a_concept_uses_its_single_chapter(self):
        self.ask("Viterbi là gì?")
        result = self.ask("Chương đó học vào tuần nào?")
        self.assertEqual("answered", result["status"])
        self.assertEqual(["CH06"], [e["id"] for e in result["context_used"]])

    def test_focus_survives_when_a_later_turn_cites_another_document_of_it(self):
        class Selective:
            model = "offline-selective-fixture"

            def __init__(self):
                self.wanted = ["Viterbi tìm", "Q7", "WEEK 07"]

            def generate(self, system, question, context):
                blocks = [json.loads(line) for line in context.splitlines()]
                want = self.wanted.pop(0)
                label = next(b["citation"] for b in blocks if want in json.dumps(b, ensure_ascii=False))
                return {"status": "answered", "answer": f"Fixture [{label}]", "citations": [label]}
        qa = CourseAssistant(root=self.root, config=load_rag_config(), client=Selective())
        manager = DialogueAssistant(qa)
        manager.ask("Viterbi là gì?", "s")
        second = manager.ask("Thuật toán này dùng ở chương nào?", "s")
        self.assertEqual(["faq.q07"], [c["document_id"] for c in second["citations"]])
        self.assertEqual(["VITERBI"], [e["id"] for e in second["state_after"]["focus"]])
        third = manager.ask("Chương đó học vào tuần nào?", "s")
        self.assertEqual(["CH06"], [e["id"] for e in third["context_used"]])

    def test_learning_outcome_followup_searches_beyond_the_lo_list(self):
        first = self.ask("L.O.2.5 học về gì?")
        self.assertEqual(["LO2.5"], [e["id"] for e in first["state_after"]["focus"]])
        result = self.ask("Nội dung này nằm ở chương nào?")
        self.assertEqual("answered", result["status"])
        self.assertIn("xác suất", result["effective_question"])
        docs = [h["document_id"] for h in result["retrieved_chunks"]]
        # LO numbers are not chapter numbers: search chapter summaries and FAQ.
        self.assertTrue(docs and all(d.startswith(("topics.", "faq.")) for d in docs), docs)
        self.assertIn("faq.q07", docs)

    def test_purpose_followup_without_an_antecedent_still_requests_clarification(self):
        result = self.ask("Thuật toán này dùng để làm gì")
        self.assertEqual("clarification_needed", result["status"])
        self.assertEqual([], result["context_used"])
        self.assertEqual(0, self.client.calls)

    def test_cited_topic_becomes_focus_without_a_question_entity(self):
        # Simulate successful QA when the catalog did not recognize a name.
        def cite_one(system, question, context):
            sources = [json.loads(line) for line in context.splitlines()]
            source = next(s for s in sources if s['title'] == 'Top-down Parsing')
            return {'status': 'answered', 'answer': 'Parsing ' + '[' + source['citation'] + ']',
                    'citations': [source['citation']]}
        with patch.object(self.manager.catalog, 'scan', return_value=[]), patch.object(self.client, 'generate', side_effect=cite_one):
            first = self.ask('top-down là gì')
        self.assertEqual('answered', first['status'])
        self.assertEqual('cited_documents', first['focus_origin'])
        focus = first['state_after']['focus']
        self.assertEqual(['slide_ch03_top_down'], [e['id'] for e in focus])
        second = self.ask('Thuật toán này dùng để làm gì')
        self.assertEqual('answered', second['status'])
        self.assertEqual('Top-down Parsing dùng để làm gì', second['effective_question'])

    def test_multiple_cited_topics_do_not_guess_a_singular_antecedent(self):
        with patch.object(self.manager.catalog, 'scan', return_value=[]):
            first = self.ask('top-down là gì')
        self.assertGreater(len(first['state_after']['focus']), 1)
        second = self.ask('Thuật toán này dùng để làm gì')
        self.assertEqual('clarification_needed', second['status'])

    def test_chapter_followup_preserves_original_parse_and_independent_mode(self):
        self.ask("Chương 3 nói về gì?")
        question = "Giải thích chương đó kỹ hơn."
        original = self.qa.parser.parse(question)
        result = self.ask(question)
        self.assertEqual("answered", result["status"])
        self.assertEqual(original, result["parse_result"])
        self.assertIn("Chương 3", result["effective_question"])
        self.assertEqual("CH03", result["context_used"][0]["id"])
        self.assertEqual(1, result["context_used"][0]["from_turn"])
        self.assertEqual("clarification_needed", self.qa.ask(question)["status"])

    def test_week_resource_lookup_uses_chapter_slides_not_week_gate(self):
        self.ask("Tuần 5 học những gì?")
        result = self.ask("Tài liệu của phần này ở đâu?")
        self.assertEqual("answered", result["status"])
        self.assertEqual("WEEK_05", result["context_used"][0]["id"])
        self.assertEqual({}, result["retrieval_query"]["retrieval_hints"]["scope"])
        self.assertTrue(result["source_map"])
        slides = [s for s in result["source_map"].values() if s.get("dataset_file")]
        self.assertTrue(slides)
        self.assertTrue(all(s.get("chapter") == 4 for s in slides))
        self.assertTrue(any(s.get("week") == 5 and s["is_sample"] for s in result["source_map"].values()))
        self.assertIn("CHAPTER 4.ppt", result["context"])
        self.assertIn("Chương 4", result["generation_question"])
        self.assertEqual(["schedule.week_05"], result["retrieval_query"]["scope_resolution"]["source_document_ids"])
        self.assertEqual("week", result["state_after"]["focus"][0]["type"])

    def test_relative_chapter_and_resource_association_survive_short_memory(self):
        self.ask("Chương 3 nói về gì?")
        following = self.ask("Chương tiếp theo thì sao?")
        self.assertEqual("answered", following["status"])
        self.assertIn("Chương 4", following["effective_question"])
        self.assertEqual("CH03", following["context_used"][0]["id"])
        resource = self.ask("Tài liệu của chương đó ở đâu?")
        self.assertEqual("answered", resource["status"])
        self.assertTrue(all(s.get("chapter") == 4 for s in resource["source_map"].values()))

    def test_relative_chapter_boundary_does_not_invent_chapter_thirteen(self):
        self.ask("Chương 12 nói về gì?")
        before = self.client.calls
        result = self.ask("Chương tiếp theo thì sao?")
        self.assertEqual("insufficient_information", result["status"])
        self.assertEqual(before, self.client.calls)
        self.assertEqual([], result["state_after"]["focus"])

    def test_topic_ellipsis_and_comparison_pair_keep_argument_order(self):
        self.ask("CFG là gì?")
        self.assertEqual("answered", self.ask("Giải thích kỹ hơn.")["status"])
        compared = self.ask("So sánh nó với PCFG.")
        self.assertEqual("answered", compared["status"])
        self.assertEqual(["CFG", "PCFG"], [e["id"] for e in compared["state_after"]["comparison_pair"]])
        again = self.ask("Hai cái đó khác nhau thế nào?")
        self.assertEqual("answered", again["status"])
        self.assertEqual(["CFG", "PCFG"], [e["id"] for e in again["context_used"]])

    def test_singular_reference_after_comparison_asks_instead_of_picking_one(self):
        self.ask("So sánh CFG và PCFG")
        before = self.client.calls
        result = self.ask("Giải thích nó kỹ hơn.")
        self.assertEqual("clarification_needed", result["status"])
        self.assertEqual(before, self.client.calls)
        pending = result["state_after"]["pending_clarification"]
        self.assertEqual(2, len(pending["candidates"]))
        clarified = self.ask("PCFG")
        self.assertEqual("answered", clarified["status"])
        self.assertTrue(clarified["resolved_from_pending"])
        self.assertIn("PCFG", clarified["effective_question"])

    def test_pending_question_accepts_entity_reply_and_numeric_chapter(self):
        for session, reply in [("named", "Chương 3"), ("number", "3")]:
            self.assertEqual("clarification_needed", self.ask("Giải thích chương đó kỹ hơn.", session)["status"])
            result = self.ask(reply, session)
            self.assertEqual("answered", result["status"])
            self.assertEqual(reply, result["question"])
            self.assertEqual("Giải thích chương đó kỹ hơn.", result["requested_question"])
            self.assertTrue(result["resolved_from_pending"])
            self.assertIn("Chương 3", result["effective_question"])
            self.assertIsNone(result["state_after"]["pending_clarification"])

    def test_pending_pair_accepts_two_named_entities(self):
        self.assertEqual("clarification_needed", self.ask("Hai cái đó khác nhau thế nào?")["status"])
        result = self.ask("CFG và PCFG")
        self.assertEqual("answered", result["status"])
        self.assertTrue(result["resolved_from_pending"])
        self.assertEqual(["CFG", "PCFG"], [e["id"] for e in result["context_used"]])

    def test_pending_reply_must_match_expected_type(self):
        self.ask("Giải thích chương đó kỹ hơn.")
        result = self.ask("CFG")
        self.assertFalse(result["resolved_from_pending"])
        self.assertNotIn("Chương 3", result["effective_question"])

    def test_new_independent_question_cancels_pending_request(self):
        self.ask("Deadline bài đó là khi nào?")
        result = self.ask("Môn NLP có bao nhiêu tín chỉ?")
        self.assertEqual("answered", result["status"])
        self.assertFalse(result["resolved_from_pending"])
        self.assertIsNone(result["state_after"]["pending_clarification"])

    def test_assignment_deadline_is_missing_information_after_binding(self):
        self.ask("Bài tập lớn có những yêu cầu gì?")
        before = self.client.calls
        result = self.ask("Còn deadline?")
        self.assertEqual("insufficient_information", result["status"])
        self.assertEqual("BTL01", result["context_used"][0]["id"])
        self.assertEqual(before, self.client.calls)

    def test_empty_memory_and_isolated_sessions_never_leak_antecedents(self):
        self.ask("Chương 3 nói về gì?", "a")
        self.ask("Chương 4 nói về gì?", "b")
        self.assertIn("Chương 3", self.ask("Giải thích chương đó kỹ hơn.", "a")["effective_question"])
        self.assertIn("Chương 4", self.ask("Giải thích chương đó kỹ hơn.", "b")["effective_question"])
        self.assertEqual("clarification_needed", self.ask("Giải thích chương đó kỹ hơn.", "c")["status"])

    def test_reset_clears_focus_and_pending_without_resetting_turn_numbers(self):
        self.ask("Chương 3 nói về gì?")
        reset = self.ask("/reset")
        self.assertEqual("session_reset", reset["status"])
        result = self.ask("Giải thích chương đó kỹ hơn.")
        self.assertEqual(3, result["turn_id"])
        self.assertEqual("clarification_needed", result["status"])

    def test_ttl_expires_antecedents_and_pending(self):
        self.ask("Chương 3 nói về gì?")
        for _ in range(3):
            self.ask("")
        self.assertEqual("clarification_needed", self.ask("Giải thích chương đó kỹ hơn.")["status"])
        for _ in range(2):
            self.ask("")
        result = self.ask("3")
        self.assertFalse(result["resolved_from_pending"])

    def test_failed_new_subject_and_generation_errors_do_not_reuse_old_focus(self):
        self.ask("Chương 3 nói về gì?")
        self.ask("Học phí NLP là bao nhiêu?")
        self.assertEqual("clarification_needed", self.ask("Giải thích chương đó kỹ hơn.")["status"])
        self.ask("Chương 3 nói về gì?")
        self.client.failure = "llm_http_503"
        failed = self.ask("Chương 4 nói về gì?")
        self.assertEqual("generation_error", failed["status"])
        self.assertEqual([], failed["state_after"]["focus"])

    def test_unicode_offsets_and_longest_noun_phrase(self):
        self.ask("Chương 3 nói về gì?")
        question = unicodedata.normalize("NFD", "Giải thích chương đó kỹ hơn.")
        result = self.ask(question)
        self.assertEqual("answered", result["status"])
        ref = result["resolution"]["references"][0]
        self.assertEqual(ref["surface"], question[ref["start"]:ref["end"]])
        self.assertEqual("answered", self.ask("Học phần này có bao nhiêu ECTS?")["status"])

    def test_memory_is_from_entities_and_used_sources_not_generated_words(self):
        result = self.ask("CFG là gì?")
        self.assertEqual(["CFG"], [e["id"] for e in result["state_after"]["focus"]])
        entity = result["state_after"]["focus"][0]
        self.assertTrue(set(entity["source_document_ids"]) <= {s["document_id"] for s in result["citations"]})
        self.assertFalse(any(source.startswith("C1") for source in entity["source_document_ids"]))

    def test_dialogue_runner_exports_and_does_not_overwrite_or_call_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [{"session_id": "demo", "turns": ["Chương 3 nói về gì?", "Giải thích chương đó kỹ hơn."]}]
            records = run_dialogues(self.manager, cases, directory)
            text = (Path(directory) / "dialogue.txt").read_text(encoding="utf-8")
            for marker in ["USER:", "CONTEXT_USED:", "RESOLUTION:", "EFFECTIVE_QUESTION:", "ANSWER:", "SOURCES:"]:
                self.assertIn(marker, text)
            saved = [json.loads(l) for l in (Path(directory) / "dialogue-trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(2, len(saved))
            self.assertEqual("answered", records[-1]["status"])
            before = self.client.calls
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_dialogues(self.manager, cases, directory)
            self.assertEqual(before, self.client.calls)

    def test_retrieve_only_dialogue_never_calls_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = run_dialogues(self.manager, [{"session_id": "dry", "turns": [
                "Tuần 5 học những gì?", "Tài liệu của phần này ở đâu?"]}], directory, retrieve_only=True)
            self.assertEqual(["retrieved", "retrieved"], [r["status"] for r in rows])
            self.assertEqual(0, self.client.calls)

    def test_dialogue_input_validation_precedes_generation(self):
        for value in [[{}], [{"session_id": "x", "turns": [1]}],
                      [{"session_id": "x", "turns": ["a"]}, {"session_id": "x", "turns": ["b"]}]]:
            with self.assertRaises(ValueError):
                validate_dialogues(value)
        self.assertEqual(0, self.client.calls)
