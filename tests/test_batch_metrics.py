"""Local LLM retry, checkpoint and grading contracts using toy data only; no holdout runs."""
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from course_assistant.rag.batch import run_batch
from course_assistant.exporters import write_jsonl
from course_assistant.generation.llm_client import OllamaClient, LLMError
from course_assistant.rag.qa import load_rag_config
from course_assistant.evaluation.qa_review import prepare_review, score_qa


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class LocalLLMRetryTests(unittest.TestCase):
    def test_server_error_is_retried_then_answer_parsed(self):
        import httpx
        clock, calls = Clock(), []
        def handler(request):
            calls.append(clock.now)
            if len(calls) == 1:
                return httpx.Response(503, json={"error": "loading model"})
            return httpx.Response(200, json={"done": True, "done_reason": "stop", "message": {
                "content": '{"status":"insufficient_information","answer":"No data","citations":[]}'}})
        with patch.dict(os.environ, {}, clear=True):
            client = OllamaClient(load_rag_config()["llm"], httpx.MockTransport(handler), clock.sleep)
            self.assertEqual("insufficient_information", client.generate("s", "q", "c")["status"])
        self.assertEqual(2, client.requests_sent)
        self.assertEqual([2], clock.waits)

    def test_missing_model_is_not_retried(self):
        import httpx
        clock = Clock()
        with patch.dict(os.environ, {}, clear=True):
            client = OllamaClient(load_rag_config()["llm"],
                                  httpx.MockTransport(lambda _: httpx.Response(404, json={"error": "model not found"})),
                                  clock.sleep)
            with self.assertRaisesRegex(LLMError, "llm_local_model_not_found"):
                client.generate("s", "q", "c")
        self.assertEqual(1, client.requests_sent)
        self.assertEqual([], clock.waits)


class ToyAssistant:
    def __init__(self, root):
        self.root = root
        self.config = {"model": "fixture"}
        self.manifest = {"checksum": "fixture"}
        self.prompt = "fixture"
        self.client = SimpleNamespace(model="fixture")
        self.parser = SimpleNamespace(grammar=SimpleNamespace(export=lambda: "grammar"), schema={}, max_tokens=128)
        self.calls = []
        self.failure = None

    def ask(self, question, retrieve_only=False):
        self.calls.append(question)
        if self.failure == question:
            raise KeyboardInterrupt()
        return {"question": question, "route": "raw_question", "retrieved_chunks": [],
                "evidence_decision": None, "context": "fixture", "source_map": {},
                "status": "retrieved" if retrieve_only else "answered", "answer": "fixture",
                "citations": [], "model": "fixture", "error": None}


class RetrievalMetricTests(unittest.TestCase):
    def test_context_dropping_gold_is_reported_even_when_top_k_contains_it(self):
        from course_assistant.evaluation.retrieval import evaluate_retrieval
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.jsonl"
            write_jsonl(labels, [{"id": "toy", "question": "Q", "expected_status": "retrieved",
                                  "relevant_documents": ["d"]}])
            assistant = SimpleNamespace(config={"top_k": 1},
                retriever=SimpleNamespace(chunks=[{"document_id": "d"}]),
                ask=lambda *args, **kwargs: {"retrieved_chunks": [{"document_id": "d"}],
                    "source_map": {}, "route": "raw_question", "status": "retrieved"})
            result = evaluate_retrieval(assistant, labels, root / "metrics")
            self.assertEqual(1.0, result["document_recall_at_k"])
            self.assertEqual(0.0, result["context_document_recall"])
            self.assertEqual(["toy"], [r["id"] for r in result["failures"]])


class BatchTests(unittest.TestCase):
    def test_interrupt_resume_preserves_success_and_exact_question_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assistant = ToyAssistant(root)
            assistant.failure = "q2"
            output = root / "output"
            with self.assertRaises(KeyboardInterrupt):
                run_batch(assistant, ["q1", "q2", "q1"], output, progress=lambda _: None)
            saved = [json.loads(line) for line in (output / "qa-trace.jsonl").read_text().splitlines()]
            self.assertEqual(["q1"], [r["question"] for r in saved])
            assistant.failure = None
            results = run_batch(assistant, ["q1", "q2", "q1"], output, resume=True, progress=lambda _: None)
            self.assertEqual([0, 1, 2], [r["batch_index"] for r in results])
            self.assertEqual(["q1", "q2", "q2", "q1"], assistant.calls)
            before = len(assistant.calls)
            run_batch(assistant, ["q1", "q2", "q1"], output, resume=True, progress=lambda _: None)
            self.assertEqual(before, len(assistant.calls))
            with self.assertRaisesRegex(ValueError, "differs"):
                run_batch(assistant, ["changed"], output, resume=True)
            assistant.client.model = "different-model"
            with self.assertRaisesRegex(ValueError, "differs"):
                run_batch(assistant, ["q1", "q2", "q1"], output, resume=True)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assistant = ToyAssistant(root)
            output = root / "out"
            run_batch(assistant, ["q"], output, progress=lambda _: None)
            original = (output / "qa-trace.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "already"):
                run_batch(assistant, ["q"], output)
            self.assertEqual(original, (output / "qa-trace.jsonl").read_bytes())

    def test_retry_errors_only_does_not_repeat_successful_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assistant = ToyAssistant(root)
            original_ask = assistant.ask
            def ask(question, retrieve_only=False):
                result = original_ask(question, retrieve_only)
                if question == "q2":
                    result.update(status="llm_unavailable", error="llm_local_model_not_found", answer=None)
                return result
            assistant.ask = ask
            output = root / "out"
            result = run_batch(assistant, ["q1", "q2", "q3"], output, progress=lambda _: None)
            self.assertEqual(2, len(result))
            assistant.ask = original_ask
            result = run_batch(assistant, ["q1", "q2", "q3"], output, resume=True, retry_errors=True, progress=lambda _: None)
            self.assertEqual(["q1", "q2", "q2", "q3"], assistant.calls)
            self.assertTrue(all(r["status"] == "answered" for r in result))


class ReviewTests(unittest.TestCase):
    def test_pending_api_and_retrieval_only_are_not_scored_as_answer_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, trace = root / "labels.jsonl", root / "trace.jsonl"
            write_jsonl(labels, [{"id": "A", "question": "q", "expected_status": "retrieved"}])
            write_jsonl(trace, [{"question": "q", "status": "llm_unavailable", "answer": None}])
            summary = score_qa(labels, trace, root / "out")
            self.assertIsNone(summary["answer_accuracy"])
            self.assertEqual(1, summary["technical_error_count"])
            self.assertEqual(0, summary["answerable_cases_scored"])
            write_jsonl(trace, [{"question": "q", "status": "retrieved", "answer": None}])
            with self.assertRaisesRegex(ValueError, "retrieval-only"):
                score_qa(labels, trace, root / "out")

    def test_ungraded_metrics_are_null_then_manual_counts_are_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, trace, review = root / "gold.jsonl", root / "trace.jsonl", root / "review.csv"
            write_jsonl(labels, [
                {"id": "A", "question": "q1", "expected_status": "retrieved", "reference_answer": "3"},
                {"id": "B", "question": "q2", "expected_status": "insufficient_information", "reference_answer": "not in KB"}])
            write_jsonl(trace, [{"question": q, "status": "answered", "answer": "fixture [C1]", "context": "fixture",
                                "citations": [{"label": "C1"}]} for q in ["q1", "q2"]])
            prepare_review(labels, trace, review)
            report = score_qa(labels, trace, root / "metrics", review)
            self.assertIsNone(report["answer_accuracy"])
            self.assertIsNone(report["groundedness_on_reviewed_claims"])
            self.assertEqual(0, report["abstention_status_recall"])
            with review.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0].update(correct="1", claims_total="1", claims_supported="1", citations_checked="1", citations_supported="1")
            rows[1].update(correct="0", claims_total="2", claims_supported="0", citations_checked="1", citations_supported="0")
            with review.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            report = score_qa(labels, trace, root / "metrics", review)
            self.assertEqual(1, report["answer_accuracy"])
            self.assertAlmostEqual(1 / 3, report["groundedness_on_reviewed_claims"])
            self.assertEqual(0.5, report["citation_precision_on_reviewed_citations"])
            with self.assertRaisesRegex(ValueError, "exists"):
                prepare_review(labels, trace, review)
            write_jsonl(trace, [{"question": "q1", "status": "answered", "answer": "CHANGED"}])
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                score_qa(labels, trace, root / "metrics", review)
