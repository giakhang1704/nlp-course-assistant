"""Part II integration tests; API responses below are fixtures, not real QA results."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from course_assistant.config import PROJECT_ROOT
from course_assistant.rag.qa import CourseAssistant, load_rag_config, export_qa
from course_assistant.knowledge.chunker import chunk_documents
from course_assistant.knowledge.loaders import load_documents
from course_assistant.retrieval.index import build_index, load_index
from course_assistant.generation.llm_client import OllamaClient, LLMError
from course_assistant.generation.context_builder import build_context


class FixtureClient:
    model = "offline-test-fixture"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, *args):
        self.calls += 1
        return self.response


class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        for relative in ["data/raw", "data/slide_content"]:
            shutil.copytree(PROJECT_ROOT / relative, cls.root / relative)
        prompt = Path("src/course_assistant/generation/prompts/answer.txt")
        (cls.root / prompt).parent.mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT / prompt, cls.root / prompt)
        cls.config = load_rag_config()
        build_index(cls.root, cls.config)
        cls.client = FixtureClient({"status": "answered", "answer": "Môn NLP có 3 tín chỉ [C1].", "citations": ["C1"]})
        cls.assistant = CourseAssistant(root=cls.root, config=cls.config, client=cls.client)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_allowlist_no_labels_or_duplicate_slide_texts(self):
        docs = load_documents(self.root)
        manifest = json.loads((self.root / "data/slide_content/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["item_count"], sum(d["type"] == "slide_topic" for d in docs))
        self.assertEqual(len(docs), len({d["id"] for d in docs}))
        self.assertFalse(any(d["metadata"]["source_file"] in {"evaluation_queries.txt", "entities.txt"} for d in docs))
        week = next(d for d in docs if d["id"] == "schedule.week_05")
        self.assertTrue(week["metadata"]["is_sample"])

    def test_learning_outcome_question_selects_the_lo_list(self):
        for question in ["L.O.2.5 là gì?", "L.O.4.2 liên quan đến nội dung nào?"]:
            with self.subTest(question=question):
                result = self.assistant.ask(question, retrieve_only=True)
                self.assertEqual("retrieved", result["status"])
                self.assertIn("course_info.detailed_learning_outcomes",
                              [h["document_id"] for h in result["retrieved_chunks"]])

    def test_refusal_is_rechecked_once_only_for_parsed_questions(self):
        class Sequence:
            model = "offline-test-fixture"

            def __init__(self, responses):
                self.responses, self.calls = list(responses), 0

            def generate(self, *args):
                self.calls += 1
                return self.responses.pop(0)
        refuse = {"status": "insufficient_information", "answer": "", "citations": []}
        answer = {"status": "answered", "answer": "Môn NLP có 3 tín chỉ [C1].", "citations": ["C1"]}
        client = Sequence([refuse, answer])
        assistant = CourseAssistant(root=self.root, config=self.config, client=client)
        result = assistant.ask("Môn NLP có bao nhiêu tín chỉ?")
        self.assertEqual(("answered", 2, 1), (result["status"], client.calls, result["generation_refusal_recheck"]))
        client = Sequence([refuse])
        assistant = CourseAssistant(root=self.root, config=self.config, client=client)
        result = assistant.ask("Sau khi chọn xong nội dung để tóm tắt thì còn phải làm hai việc gì?")
        self.assertEqual("raw_question", result["route"])
        self.assertLessEqual(client.calls, 1)
        if client.calls:
            self.assertEqual("Không tìm thấy thông tin này trong dữ liệu môn học.", result["answer"])

    def test_semantic_slot_and_course_count(self):
        for question, expected in [("Môn NLP có bao nhiêu tín chỉ?", "course_info.credits:000"),
                                   ("NLP có mấy chương?", "course_info.chapter_count:000"),
                                   ("Thi giữa kỳ bao nhiêu phút?", "course_info.midterm:000")]:
            with self.subTest(question=question):
                result = self.assistant.ask(question, retrieve_only=True)
                self.assertEqual("semantic", result["route"])
                self.assertEqual("retrieved", result["status"])
                self.assertIn(expected, result["evidence_decision"]["selected_chunk_ids"])

    def test_unavailable_local_llm_reports_status_and_keeps_context(self):
        import httpx
        def unreachable(request):
            raise httpx.ConnectError("refused", request=request)
        old = self.assistant.client
        try:
            with patch.dict(os.environ, {}, clear=True):
                self.assistant.client = OllamaClient(self.config["llm"], transport=httpx.MockTransport(unreachable))
                result = self.assistant.ask("Môn NLP có bao nhiêu tín chỉ?")
                self.assertEqual("llm_unavailable", result["status"])
                self.assertEqual("llm_local_server_unreachable", result["error"])
                self.assertTrue(result["context"])
                self.assertIsNone(result["answer"])
        finally:
            self.assistant.client = old

    def test_sample_generation_is_labeled_and_preserves_source(self):
        old = self.assistant.client
        try:
            self.assistant.client = FixtureClient({"status": "answered", "answer": "Tuần 5 học features [C1].", "citations": ["C1"]})
            result = self.assistant.ask("Tuần 5 học những gì?")
            self.assertEqual("answered", result["status"])
            self.assertIn("dữ liệu mẫu", result["answer"])
            self.assertTrue(result["citations"][0]["is_sample"])
        finally:
            self.assistant.client = old

    def test_raw_question_retrieves_slide_knowledge(self):
        question = "Inflection khác Derivation ở điểm nào vậy?"
        result = self.assistant.ask(question, retrieve_only=True)
        self.assertEqual("raw_question", result["route"])
        self.assertEqual("()", result["parse_result"]["parse_tree"])
        self.assertEqual("slide_ch02_word_formation:000", result["retrieved_chunks"][0]["id"])
        self.assertEqual("retrieved", result["status"])

    def test_week_selects_correct_week_and_preserves_sample_label(self):
        result = self.assistant.ask("Tuần 5 học những gì?", retrieve_only=True)
        self.assertEqual(["schedule.week_05:000"], result["evidence_decision"]["selected_chunk_ids"])
        self.assertTrue(result["source_map"]["C1"]["is_sample"])

    def test_missing_ood_and_unknown_chapter_do_not_call_model(self):
        before = self.client.calls
        for question in ["Ai dạy môn NLP?", "Môn NLP học ở phòng nào?",
                         "Email của giảng viên là gì?", "Deadline của bài tập lớn là khi nào?",
                         "Thời tiết Đà Lạt ngày mai ra sao?", "Chương 13 nói về gì?"]:
            with self.subTest(question=question):
                self.assertEqual("insufficient_information", self.assistant.ask(question)["status"])
        self.assertEqual(before, self.client.calls)

    def test_context_and_invalid_input_do_not_call_model(self):
        before = self.client.calls
        for question, expected in [("Tài liệu của phần này ở đâu?", "clarification_needed"),
                                   ("Giải thích cái phần vừa nói", "clarification_needed"),
                                   ("", "empty_input"), ("a " * 129, "input_too_long")]:
            self.assertEqual(expected, self.assistant.ask(question)["status"])
        self.assertEqual(before, self.client.calls)

    def test_generated_citations_resolve_to_supplied_source(self):
        result = self.assistant.ask("Môn NLP có bao nhiêu tín chỉ?")
        self.assertEqual("answered", result["status"])
        self.assertEqual("course_info.txt", result["citations"][0]["source_file"])
        self.assertEqual("offline-test-fixture", result["model"])

    def test_raw_field_hints_find_values_without_changing_parser_output(self):
        for question, fields in [
            ("Mã môn học NLP là gì?", {"course_id"}),
            ("Tên tiếng Anh của CO3085 là gì?", {"course_name_en"}),
            ("CO3085 thuộc học kỳ nào?", {"semester"}),
            ("Học phần này quy đổi được bao nhiêu ECTS?", {"ects"}),
            ("CO3085 dành bao nhiêu giờ lý thuyết và bao nhiêu giờ tự học?", {"lecture_hours", "self_study_hours"}),
            ("Thi giữa kỳ chiếm tỷ trọng điểm bao nhiêu?", {"midterm_exam"}),
            ("Bài final kéo dài bao nhiêu phút?", {"final"}),
        ]:
            with self.subTest(question=question):
                before = self.assistant.parser.parse(question)
                result = self.assistant.ask(question, retrieve_only=True)
                self.assertEqual(before, result["parse_result"])
                self.assertEqual("retrieved", result["status"])
                self.assertEqual(fields, {s["field"] for s in result["source_map"].values()})
                self.assertNotIn("retrieval_hints", before["query"])

    def test_unresolved_references_ask_for_details_before_retrieval(self):
        for question in ["Nêu ví dụ cho chủ đề ấy.", "Tóm tắt thuật toán vừa nêu.",
                         "Deadline bài đó?", "Tài liệu chương ấy ở đâu?",
                         "Hai phương pháp trước đó khác nhau thế nào?"]:
            with self.subTest(question=question):
                result = self.assistant.ask(question, retrieve_only=True)
                self.assertEqual("clarification_needed", result["status"])
                self.assertEqual([], result["retrieved_chunks"])

    def test_raw_schedule_and_multiword_titles_keep_relevant_evidence(self):
        for question, document in [
            ("Lịch mẫu tuần 8 liệt kê những nội dung nào?", "schedule.week_08"),
            ("Tuần cuối cùng có hoạt động gì?", "schedule.week_15"),
            ("Tuần đầu tiên có chủ đề gì?", "schedule.week_01"),
            ("Phần Grammar and Parser yêu cầu những file output nào?", "assignments.part_1"),
        ]:
            with self.subTest(question=question):
                result = self.assistant.ask(question, retrieve_only=True)
                self.assertEqual("retrieved", result["status"])
                self.assertIn(document, {s["document_id"] for s in result["source_map"].values()})

    def test_specific_missing_fields_do_not_use_nearby_course_facts(self):
        before = self.client.calls
        for question in ["Số điện thoại của giảng viên CO3085?",
                         "Học phí học phần này là bao nhiêu?",
                         "Thi giữa kỳ bắt đầu ngày nào và giờ nào?",
                         "Đường link nộp bài tập lớn là gì?",
                         "Giờ tiếp sinh viên của giảng viên NLP?",
                         "Phòng máy thực hành CO3085 là phòng nào?"]:
            with self.subTest(question=question):
                self.assertEqual("insufficient_information", self.assistant.ask(question)["status"])
        self.assertEqual(before, self.client.calls)

    def test_rejected_generation_has_precise_diagnostic(self):
        old = self.assistant.client
        try:
            client = FixtureClient({"status": "answered", "answer": "fixture [C999]", "citations": ["C999"]})
            self.assistant.client = client
            result = self.assistant.ask("Môn NLP có bao nhiêu tín chỉ?")
            self.assertEqual("generation_error", result["status"])
            self.assertEqual("llm_unknown_citation", result["error"])
            self.assertIn("C999", result["generation_diagnostic"]["preview"])
        finally:
            self.assistant.client = old

    def test_invented_or_unlisted_citations_rejected(self):
        old = self.assistant.client
        try:
            for response in [{"status": "answered", "answer": "Có 3 tín chỉ [C999].", "citations": ["C999"]},
                             {"status": "answered", "answer": "Có 3 tín chỉ [C1].", "citations": []},
                             {"status": "answered", "answer": "Có 3 tín chỉ.", "citations": []}]:
                self.assistant.client = FixtureClient(response)
                result = self.assistant.ask("Môn NLP có bao nhiêu tín chỉ?")
                self.assertEqual("generation_error", result["status"])
                self.assertIsNone(result["answer"])
        finally:
            self.assistant.client = old

    def test_stale_source_fails_without_rebuilding_in_place(self):
        path = self.root / "data/raw/course_info.txt"
        previous = path.read_bytes()
        try:
            path.write_bytes(previous + b"\nchanged\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_index(self.root, self.config)
        finally:
            path.write_bytes(previous)

    def test_chunking_keeps_content_and_context_respects_budget(self):
        doc = {"id": "long", "type": "topic", "title": "Test", "text": "x" * 1000,
               "metadata": {"source_file": "test.txt", "is_sample": False}}
        chunks = chunk_documents([doc], max_chars=200)
        self.assertEqual(1000, sum(len(c["text"]) for c in chunks))
        context, sources = build_context(chunks, max_chars=400)
        self.assertLessEqual(len(context), 400)
        self.assertEqual(1, len(sources))

    def test_required_exports_report_real_status_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.assistant.ask("Môn NLP có bao nhiêu tín chỉ?", retrieve_only=True)
            export_qa([result], directory)
            saved = json.loads((Path(directory) / "qa-trace.jsonl").read_text(encoding="utf-8"))
            self.assertEqual("retrieved", saved["status"])
            self.assertIsNone(saved["answer"])
            self.assertIn("CONTEXT:", (Path(directory) / "retrieval.txt").read_text(encoding="utf-8"))


class OllamaClientTests(unittest.TestCase):
    """Local model path; Ollama responses below are fixtures, no server is started."""

    def local_config(self):
        return load_rag_config()

    def test_default_config_selects_ollama_without_api_key(self):
        from course_assistant.generation.llm_client import OllamaClient, make_client
        config = self.local_config()
        self.assertEqual("ollama", config["llm"]["provider"])
        with patch.dict(os.environ, {}, clear=True):
            client = make_client(config["llm"])
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(config["llm"]["model"], client.model)

    def test_request_uses_local_chat_api_with_json_schema(self):
        import httpx
        def handler(request):
            self.assertEqual("http://localhost:11434/api/chat", str(request.url))
            self.assertNotIn("authorization", {k.lower() for k in request.headers})
            payload = json.loads(request.content)
            self.assertFalse(payload["stream"])
            self.assertEqual(0, payload["options"]["temperature"])
            self.assertEqual(["answered", "insufficient_information"],
                             payload["format"]["properties"]["status"]["enum"])
            self.assertEqual(["system", "user"], [m["role"] for m in payload["messages"]])
            return httpx.Response(200, json={"done": True, "done_reason": "stop", "message": {
                "role": "assistant",
                "content": json.dumps({"status": "answered", "answer": "test [C1]", "citations": ["C1"]})}})
        with patch.dict(os.environ, {}, clear=True):
            from course_assistant.generation.llm_client import OllamaClient
            client = OllamaClient(self.local_config()["llm"], transport=httpx.MockTransport(handler))
            self.assertEqual("answered", client.generate("system", "question", "context")["status"])
            self.assertEqual(1, client.requests_sent)

    def test_local_failures_are_errors_not_answers(self):
        import httpx
        from course_assistant.generation.llm_client import OllamaClient
        def unreachable(request):
            raise httpx.ConnectError("refused", request=request)
        cases = [(unreachable, "llm_local_server_unreachable"),
                 (lambda _: httpx.Response(404, json={"error": "model not found"}), "llm_local_model_not_found"),
                 (lambda _: httpx.Response(200, json={"done": True, "done_reason": "length",
                                                      "message": {"content": "{\"status\": \"ans"}}), "truncated"),
                 (lambda _: httpx.Response(200, json={"done": True, "done_reason": "stop",
                                                      "message": {"content": "not JSON"}}), "llm_invalid_json")]
        with patch.dict(os.environ, {}, clear=True):
            for handler, code in cases:
                client = OllamaClient(self.local_config()["llm"], transport=httpx.MockTransport(handler))
                with self.assertRaisesRegex(LLMError, code):
                    client.generate("s", "q", "c")

    def test_environment_overrides_model_and_host(self):
        from course_assistant.generation.llm_client import OllamaClient
        with patch.dict(os.environ, {"OLLAMA_MODEL": "gemma4:e4b", "OLLAMA_HOST": "http://127.0.0.1:9999/"}, clear=True):
            client = OllamaClient(self.local_config()["llm"])
        self.assertEqual("gemma4:e4b", client.model)
        self.assertEqual("http://127.0.0.1:9999", client.base_url)


class GenerationContractTests(unittest.TestCase):
    def test_equivalent_citation_format_is_normalized_without_changing_labels(self):
        from course_assistant.generation.validation import validate_generation
        response = {"status": "answered", "answer": "Nội dung [C1, C2].", "citations": ["[C1]", "C2", "C2"]}
        self.assertEqual(("answered", "Nội dung [C1][C2].", ["C1", "C2"]),
                         validate_generation(response, {"C1": {}, "C2": {}}))

    def test_echoed_question_is_rejected_and_bare_refusal_is_valid(self):
        from course_assistant.generation.validation import validate_generation, GenerationValidationError
        question = "Chương 11 có hỏi đáp không?"
        with self.assertRaisesRegex(GenerationValidationError, "llm_echoed_question"):
            validate_generation({"status": "answered", "answer": "Theo dữ liệu mẫu của project: Chương 11 có hỏi đáp không? [C1]",
                                 "citations": ["C1"]}, {"C1": {}}, question)
        self.assertEqual(("answered", "Có, Chương 11 có procedural semantics [C1].", ["C1"]),
                         validate_generation({"status": "answered", "answer": "Có, Chương 11 có procedural semantics [C1].",
                                              "citations": ["C1"]}, {"C1": {}}, question))
        self.assertEqual(("insufficient_information", "", []),
                         validate_generation({"status": "insufficient_information", "answer": "", "citations": []}, {"C1": {}}))

    def test_declared_citations_are_rendered_only_when_none_is_inline(self):
        from course_assistant.generation.validation import validate_generation, GenerationValidationError
        uncited = {"status": "answered", "answer": "Có, PART II là Semantic Interpretation and QA.", "citations": ["C1"]}
        with self.assertRaisesRegex(GenerationValidationError, "llm_citation_mismatch"):
            validate_generation(uncited, {"C1": {}})
        self.assertEqual(("answered", "Có, PART II là Semantic Interpretation and QA. [C1]", ["C1"]),
                         validate_generation(uncited, {"C1": {}}, render_declared=True))
        with self.assertRaisesRegex(GenerationValidationError, "llm_citation_mismatch"):
            validate_generation({"status": "answered", "answer": "Nội dung [C1].", "citations": ["C2"]}, {"C1": {}, "C2": {}})

    def test_schema_closes_citation_labels_to_context(self):
        from course_assistant.generation.validation import response_schema
        schema = response_schema('{"citation":"C1","text":"example [C999]"}\n{"citation":"C2"}')
        self.assertEqual(["C1", "C2"], schema["properties"]["citations"]["items"]["enum"])

    def test_no_citation_is_added_to_make_an_answer_valid(self):
        from course_assistant.generation.validation import validate_generation, GenerationValidationError
        for answer, citations, code in [("Nội dung [C1].", [], "mismatch"),
                                         ("Nội dung.", [], "missing_citations"),
                                         ("Nội dung [C1, C999].", ["C1"], "unknown_citation")]:
            with self.subTest(code=code), self.assertRaisesRegex(GenerationValidationError, code):
                validate_generation({"status": "answered", "answer": answer, "citations": citations}, {"C1": {}})

    def test_body_only_matches_can_pass_without_a_title_keyword(self):
        from course_assistant.retrieval.lexical import LexicalRetriever
        from course_assistant.rag.question_analysis import enrich_query
        from course_assistant.rag.policy import evidence_decision
        from course_assistant.semantics.query_builder import build_fallback_query
        chunk = {"id": "toy:000", "document_id": "toy", "title": "Unit one",
                 "text": "Quux algorithm caches spans for reuse.",
                 "metadata": {"source_file": "toy.txt", "is_sample": False}}
        query = enrich_query(build_fallback_query("Quux algorithm caches spans?"))
        hits = LexicalRetriever([chunk]).retrieve(query)
        self.assertEqual([], hits[0]["anchor_matches"])
        result = {"query": query}
        self.assertEqual("evidence_available", evidence_decision(result, hits, {"min_score": 0.08}, {})["status"])

    def test_one_shared_word_is_not_enough_for_an_unrelated_question(self):
        from course_assistant.retrieval.lexical import LexicalRetriever
        from course_assistant.rag.question_analysis import enrich_query
        from course_assistant.rag.policy import evidence_decision
        from course_assistant.semantics.query_builder import build_fallback_query
        chunk = {"id": "toy:000", "document_id": "toy", "title": "World Knowledge",
                 "text": "World Knowledge supports inference.",
                 "metadata": {"source_file": "toy.txt", "is_sample": False}}
        query = enrich_query(build_fallback_query("World Cup champion 2030?"))
        hits = LexicalRetriever([chunk]).retrieve(query)
        self.assertIn("world", hits[0]["anchor_matches"])
        self.assertEqual("insufficient_information", evidence_decision({"query": query}, hits, {"min_score": 0.08}, {})["status"])
