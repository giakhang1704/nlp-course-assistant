"""Browser/API contracts with a fake generator; no live API calls."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
try:
    from fastapi.testclient import TestClient
    from course_assistant.web.app import create_app
except ImportError:
    TestClient = None
from course_assistant.config import PROJECT_ROOT
from course_assistant.rag.qa import CourseAssistant, load_rag_config
from course_assistant.retrieval.index import build_index
from course_assistant.web.presentation import clean_answer
from uuid import uuid4


class FakeGenerator:
    model = "web-test-fixture"
    def __init__(self):
        self.calls = 0
    def generate(self, system, question, context):
        self.calls += 1
        labels = [json.loads(line)["citation"] for line in context.splitlines()]
        return {"status": "answered", "answer": "Nội dung môn học " + "".join(f"[{c}]" for c in labels), "citations": labels}


@unittest.skipIf(TestClient is None, "Install requirements-web.txt to test the optional web app")
class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        for folder in ["data/raw", "data/slide_content"]:
            shutil.copytree(PROJECT_ROOT / folder, cls.root / folder)
        prompt = Path("src/course_assistant/generation/prompts/answer.txt")
        (cls.root / prompt).parent.mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT / prompt, cls.root / prompt)
        config = load_rag_config()
        build_index(cls.root, config)
        cls.generator = FakeGenerator()
        cls.qa = CourseAssistant(root=cls.root, config=config, client=cls.generator)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.db = self.root / f"{uuid4()}.sqlite3"
        self.app = create_app(self.qa, self.db)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def conversation(self):
        response = self.client.post("/api/conversations")
        self.assertEqual(201, response.status_code)
        return response.json()["id"]

    def send(self, identity, content, request_id=None):
        return self.client.post(f"/api/conversations/{identity}/messages", json={"content": content, "request_id": request_id or str(uuid4())})

    def test_health_and_course_do_not_expose_credentials(self):
        response = self.client.get("/api/health")
        self.assertTrue(response.json()["ready"])
        self.assertNotIn("key", response.text.casefold())
        self.assertEqual(12, len(self.client.get("/api/course").json()["chapters"]))
        self.assertEqual(200, self.client.get("/favicon.svg").status_code)

    def test_browser_ownership_and_csrf_origin(self):
        identity = self.conversation()
        other = TestClient(self.app)
        self.assertEqual(404, other.get(f"/api/conversations/{identity}").status_code)
        self.assertEqual(404, other.delete(f"/api/conversations/{identity}").status_code)
        self.assertEqual(403, self.client.post("/api/conversations", headers={"origin": "https://unrelated.example"}).status_code)
        cookie = self.client.cookies.get("ca_browser")
        self.assertTrue(cookie)

    def test_clean_prose_and_source_provenance_are_separate(self):
        identity = self.conversation()
        response = self.send(identity, "Tuần 5 học những gì?")
        self.assertEqual(200, response.status_code)
        turn = response.json()
        self.assertEqual("Nội dung môn học", turn["content"])
        self.assertTrue(turn["has_sample_source"])
        self.assertTrue(turn["sources"][0]["is_sample"])
        self.assertNotIn("[C", turn["content"])
        self.assertEqual(1, len(self.client.get(f"/api/conversations/{identity}").json()["turns"]))

    def test_idempotent_message_and_conflicting_payload(self):
        identity = self.conversation()
        request_id = str(uuid4())
        before = self.generator.calls
        first = self.send(identity, "CFG là gì?", request_id)
        second = self.send(identity, "CFG là gì?", request_id)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(before + 1, self.generator.calls)
        self.assertEqual(409, self.send(identity, "PCFG là gì?", request_id).status_code)

    def test_parser_analysis_is_preserved_separately_from_qa_status(self):
        identity = self.conversation()
        turn = self.send(identity, "Môn NLP có bao nhiêu tín chỉ?").json()
        analysis = turn["details"]["analysis"]
        self.assertTrue(analysis["is_valid"])
        self.assertIn("QUERY_COURSE_QUANTITY", analysis["parse_tree"])
        self.assertEqual("credits", analysis["semantic_frame"]["attribute"])
        self.assertEqual("CO3085", analysis["entities"][0]["id"])
        fallback = self.send(identity, "Hai quá trình Inflection và Derivation khác nhau ở đâu?").json()
        self.assertEqual("()", fallback["details"]["analysis"]["parse_tree"])
        self.assertTrue(fallback["details"]["retrieval"])

    def test_original_and_context_resolved_parses_are_distinct(self):
        identity = self.conversation()
        self.send(identity, "Chương 3 nói về gì?")
        turn = self.send(identity, "Giải thích chương đó kỹ hơn.").json()
        self.assertNotEqual(turn["question"], turn["details"]["effective_question"])
        self.assertIsNotNone(turn["details"]["effective_analysis"])
        self.assertEqual(turn["question"], turn["details"]["analysis"]["sentence"])

    def test_restart_restores_pending_clarification_and_cookie_owner(self):
        identity = self.conversation()
        turn = self.send(identity, "Giải thích chương đó kỹ hơn.").json()
        self.assertEqual("clarification_needed", turn["status"])
        with TestClient(create_app(self.qa, self.db)) as restarted:
            restarted.cookies.update(self.client.cookies)
            response = restarted.post(f"/api/conversations/{identity}/messages", json={"content": "Chương 3", "request_id": str(uuid4())})
            self.assertEqual(200, response.status_code)
            self.assertIn("Chương 3", response.json()["details"]["effective_question"])
            self.assertEqual("answered", response.json()["status"])

    def test_context_isolated_between_conversations_and_reset(self):
        first, second = self.conversation(), self.conversation()
        self.send(first, "Chương 3 nói về gì?")
        self.assertEqual("clarification_needed", self.send(second, "Giải thích chương đó kỹ hơn.").json()["status"])
        bound = self.send(first, "Giải thích chương đó kỹ hơn.").json()
        self.assertIn("Chương 3", bound["details"]["effective_question"])
        self.assertEqual(200, self.client.post(f"/api/conversations/{first}/reset").status_code)
        self.assertEqual([], self.client.get(f"/api/conversations/{first}").json()["turns"])
        self.assertEqual("clarification_needed", self.send(first, "Giải thích chương đó kỹ hơn.").json()["status"])

    def test_rename_delete_and_validation(self):
        identity = self.conversation()
        self.assertEqual(422, self.send(identity, "   ").status_code)
        self.assertEqual(200, self.client.patch(f"/api/conversations/{identity}", json={"title": "Parsing"}).status_code)
        self.assertEqual("Parsing", self.client.get("/api/conversations").json()[0]["title"])
        self.assertEqual(200, self.client.delete(f"/api/conversations/{identity}").status_code)
        self.assertEqual(404, self.client.get(f"/api/conversations/{identity}").status_code)


class PresentationTests(unittest.TestCase):
    def test_removes_citation_markers_and_known_sample_prefix_without_changing_facts(self):
        self.assertEqual("Bài tập có 3 phần.", clean_answer("Theo dữ liệu mẫu (đây không phải thông báo chính thức) [C1], Bài tập có 3 phần [C2]."))
        self.assertEqual("Mã môn CO3085.", clean_answer("Mã môn CO3085 [C1]."))
