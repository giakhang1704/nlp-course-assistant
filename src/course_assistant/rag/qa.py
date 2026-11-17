"""Part II orchestration, separate from the Part I export pipeline."""
import json
import math
import os
import time
from pathlib import Path
from ..config import PROJECT_ROOT
from ..parsing.api import CourseParser
from .policy import preliminary_decision, evidence_decision
from .question_analysis import enrich_query, unresolved_references
from ..retrieval.index import load_index
from ..generation.context_builder import build_context
from ..generation.llm_client import LLMError, UNAVAILABLE_ERRORS, load_environment, make_client
from ..generation.validation import validate_generation, GenerationValidationError, diagnostic_preview
from ..exporters import write_jsonl, write_lines

# One regeneration under the same evidence for these rejected outputs.
RETRY_NOTES = {
    "llm_citation_mismatch": "Đầu ra trước không khớp định dạng citation. Tạo lại câu trả lời từ EVIDENCE; "
                             "mỗi citation khai báo phải xuất hiện trong answer dưới dạng [C1], và ngược lại.",
    "llm_answer_missing_citations": "Đầu ra trước không khớp định dạng citation. Tạo lại câu trả lời từ EVIDENCE; "
                                    "mỗi citation khai báo phải xuất hiện trong answer dưới dạng [C1], và ngược lại.",
    "llm_unknown_citation": "Đầu ra trước dùng nhãn citation không có trong EVIDENCE. Chỉ dùng các nhãn [C1], [C2]... "
                            "đúng như trong EVIDENCE; khi liệt kê, đánh số bằng 1., 2. hoặc gạch đầu dòng, không dùng [Cn] làm số thứ tự.",
    "llm_echoed_question":"Đầu ra trước chỉ lặp lại câu hỏi. Trả lời câu hỏi bằng nội dung trong EVIDENCE; "
                           "với câu hỏi có/không, mở đầu bằng \"Có\" hoặc \"Không\" rồi nêu căn cứ kèm citation.",
}
REFUSAL_RECHECK = ("Đầu ra trước là insufficient_information. Đọc lại từng đoạn EVIDENCE: nếu có đoạn nêu trực tiếp "
                   "thông tin được hỏi (kể cả khi dùng từ khác câu hỏi, ví dụ percent = phần trăm, AI_usage = quy định "
                   "dùng AI) thì trả lời bằng thông tin đó với citation; nếu thật sự không có thì giữ insufficient_information.")


def load_rag_config(path=None, root=PROJECT_ROOT):
    """configs/rag.json by default; RAG_CONFIG or --rag-config selects another file."""
    path = Path(path or os.environ.get("RAG_CONFIG") or Path(root) / "configs/rag.json")
    if not path.is_absolute() and not path.exists():
        path = Path(root) / path  # relative to the project root when not found from the cwd
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for key, minimum in [("top_k", 1), ("context_chars", 200), ("chunk_chars", 200)]:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < minimum:
            raise ValueError(f"Invalid RAG configuration: {key}")
    for key in ["min_score", "preferred_source_boost", "min_query_coverage", "min_body_coverage"]:
        value = config.get(key, {"min_query_coverage": 0.3, "min_body_coverage": 0.5}.get(key))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Invalid RAG configuration: {key}")
    if config["llm"].get("provider") != "ollama":
        raise ValueError("Only the local provider 'ollama' is supported")
    timeout = config["llm"].get("timeout_seconds", 600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid LLM configuration: timeout_seconds")
    attempts = config["llm"].get("max_attempts", 2)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 5:
        raise ValueError("max_attempts must be an integer in [1,5]")
    for key, default in [("max_output_tokens", 800), ("num_ctx", 8192)]:
        value = config["llm"].get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return config


class CourseAssistant:
    def __init__(self, root=PROJECT_ROOT, config=None, parser=None, client=None):
        self.root = Path(root)
        self.config = config or load_rag_config(root=self.root)
        self.parser = parser or CourseParser()
        self.retriever, self.manifest = load_index(self.root, self.config)
        load_environment(self.root)
        self.client = client if client is not None else make_client(self.config["llm"])
        self.prompt = (self.root / "src/course_assistant/generation/prompts/answer.txt").read_text(encoding="utf-8")

    def new_result(self, question, parsed):
        return {"question": question, "parse_result": parsed,
                  "route": parsed["query"]["mode"] if parsed["query"] else "invalid_input",
                  "retrieval_query": parsed["query"], "retrieved_chunks": [],
                  "evidence_decision": None, "context": "", "source_map": {},
                  "status": None, "answer": None, "citations": [], "model": None, "error": None,
                  "generation_attempts": 0, "generation_seconds": None,
                  "unresolved_references": unresolved_references(question),
                  "generation_diagnostic": None}

    def ask(self, question, retrieve_only=False, k=None):
        parsed = self.parser.parse(question)
        result = self.new_result(question, parsed)
        preliminary = preliminary_decision(parsed)
        if preliminary:
            result.update(status=preliminary[0], answer=preliminary[1])
            return result
        return self.answer_query(question, parsed, enrich_query(parsed["query"]), retrieve_only, k)

    def answer_query(self, question, parsed, query, retrieve_only=False, k=None, generation_question=None):
        """Internal shared QA path; callers must validate input and resolve references first."""
        result = self.new_result(question, parsed)
        result["retrieval_query"] = query
        hits = self.retriever.retrieve(query, self.config["top_k"] if k is None else k)
        result["retrieved_chunks"] = hits
        decision = evidence_decision(parsed, hits, self.config, self.manifest, query=query)
        result["evidence_decision"] = {"status": decision["status"], "reason": decision["reason"],
                                       "selected_chunk_ids": [h["id"] for h in decision["selected"]]}
        if decision["status"] != "evidence_available":
            result.update(status=decision["status"], answer=decision["reason"])
            return result
        context, sources = build_context(decision["selected"], self.config["context_chars"])
        result.update(context=context, source_map=sources)
        if not sources:
            result.update(status="insufficient_information", answer="Bằng chứng không vừa giới hạn context; cần điều chỉnh cấu hình.")
            return result
        if retrieve_only:
            result["status"] = "retrieved"
            return result
        result["model"] = self.client.model
        started = time.perf_counter()
        before = getattr(self.client, "requests_sent", 0)
        generation = None
        try:
            asked = generation_question or question
            generation = self.client.generate(self.prompt, asked, context)
            try:
                status, answer, citations = validate_generation(generation, sources, asked)
            except GenerationValidationError as exc:
                if str(exc) not in RETRY_NOTES:
                    raise
                # Regenerate once under the same evidence. Never append a
                # citation to rejected prose or silently weaken validation.
                result["generation_validation_retries"] = 1
                generation = self.client.generate(self.prompt + "\n" + RETRY_NOTES[str(exc)], asked, context)
                status, answer, citations = validate_generation(generation, sources, asked, render_declared=True)
            if status == "insufficient_information" and query.get("mode") == "semantic" and not result.get("generation_validation_retries"):
                # The grammar understood the request and the evidence policy found
                # matching fields: ask the model once to re-read the evidence.
                # Raw (unparsed) questions are not rechecked, so out-of-domain
                # questions that only share words with the KB keep their refusal.
                result["generation_refusal_recheck"] = 1
                generation = self.client.generate(self.prompt + "\n" + REFUSAL_RECHECK, asked, context)
                status, answer, citations = validate_generation(generation, sources, asked)
            if status == "insufficient_information" and not answer:
                answer = "Không tìm thấy thông tin này trong dữ liệu môn học."
            if status == "answered" and any(sources[c].get("is_sample") for c in citations):
                if "dữ liệu mẫu" not in answer.casefold():
                    answer = "Theo dữ liệu mẫu của project: " + answer
            result.update(status=status, answer=answer,
                          citations=[{"label": c, **sources[c]} for c in citations])
        except (LLMError, GenerationValidationError) as exc:
            code = str(exc)
            result.update(status="llm_unavailable" if code in UNAVAILABLE_ERRORS else "generation_error",
                          error=code, answer=None,
                          generation_diagnostic=(getattr(exc, "diagnostic", None) or
                              (diagnostic_preview(generation)
                               if isinstance(exc, GenerationValidationError) else None)))
        finally:
            result["generation_attempts"] = getattr(self.client, "requests_sent", 0) - before
            result["generation_seconds"] = round(time.perf_counter() - started, 3)
        return result


def export_qa(results, output_dir):
    output_dir = Path(output_dir)
    write_jsonl(output_dir / "qa-trace.jsonl", results)
    retrieval_lines, answer_lines = [], []
    for index, result in enumerate(results, 1):
        retrieval_lines += [f"QUESTION {index}: {result['question']}", f"ROUTE: {result['route']}"]
        for hit in result["retrieved_chunks"]:
            retrieval_lines += [f"CHUNK: {hit['id']} | SCORE: {hit['score']:.6f}",
                                f"SOURCE: {hit['metadata']['source_file']}",
                                f"SAMPLE: {hit['metadata']['is_sample']}", hit["text"], ""]
        retrieval_lines += ["EVIDENCE: " + json.dumps(result["evidence_decision"], ensure_ascii=False),
                            "CONTEXT:", result["context"], ""]
        answer_lines += [f"QUESTION {index}: {result['question']}", f"STATUS: {result['status']}",
                         f"MODEL: {result['model'] or '-'}", "ANSWER:",
                         result["answer"] or ("Chưa sinh câu trả lời; " + (result["error"] or "chỉ chạy retrieval")),
                         "SOURCES: " + json.dumps(result["citations"], ensure_ascii=False), ""]
    write_lines(output_dir / "retrieval.txt", retrieval_lines)
    write_lines(output_dir / "answer.txt", answer_lines)
