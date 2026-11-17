"""Atomic per-question checkpoints and validated resumable QA runs."""
import hashlib
import json
import sys
from pathlib import Path
from ..exporters import write_json
from ..knowledge.loaders import read_jsonl
from .qa import export_qa

from ..generation.llm_client import UNAVAILABLE_ERRORS

ERROR_STATUSES = {"generation_error", "llm_unavailable"}
# Setup problems that will fail for every question: stop instead of trying the rest.
STOP_ERRORS = UNAVAILABLE_ERRORS | {"llm_http_400"}


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def run_signature(assistant, questions, retrieve_only):
    code_root = assistant.root / "src/course_assistant"
    code = [(p.relative_to(code_root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(code_root.rglob("*.py"))]
    parser_state = {"grammar": assistant.parser.grammar.export(), "schema": assistant.parser.schema,
                    "max_tokens": assistant.parser.max_tokens}
    # Store fingerprints, not raw configuration or credentials.
    return {"questions_sha256": sha(questions), "config_sha256": sha(assistant.config),
            "knowledge_sha256": sha(assistant.manifest), "code_sha256": sha(code),
            "parser_sha256": sha(parser_state),
            "prompt_sha256": sha(assistant.prompt),
            "model": None if retrieve_only else assistant.client.model,
            "retrieve_only": retrieve_only, "question_count": len(questions)}


def run_batch(assistant, questions, output_dir, retrieve_only=False, resume=False,
              retry_errors=False, progress=None):
    if retry_errors and not resume:
        raise ValueError("--retry-errors requires --resume")
    output_dir = Path(output_dir)
    metadata_path = output_dir / "batch.json"
    trace_path = output_dir / "qa-trace.jsonl"
    signature = run_signature(assistant, questions, retrieve_only)
    completed = {}
    if resume:
        if not metadata_path.is_file():
            raise ValueError("Batch checkpoint missing; use a new output directory without --resume")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            raise ValueError("Cannot resume: input, model, parser, code, configuration or KB differs; use a new output directory")
        if trace_path.is_file():
            for record in read_jsonl(trace_path):
                index = record.get("batch_index")
                if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(questions)
                        or questions[index] != record.get("question") or index in completed):
                    raise ValueError("Invalid batch checkpoint: question/index mismatch")
                completed[index] = record
            if len(completed) < metadata.get("completed_count", 0):
                raise ValueError("Batch trace is incomplete compared with checkpoint metadata")
        elif metadata.get("completed_count", 0):
            raise ValueError("Batch trace is missing despite completed checkpoint; restore it or use a new directory")
    elif any((output_dir / name).exists() for name in ("batch.json", "qa-trace.jsonl", "answer.txt", "retrieval.txt")):
        raise ValueError("Output directory already has results; use --resume or choose a new directory")

    def save(state):
        records = [completed[i] for i in sorted(completed)]
        export_qa(records, output_dir)
        write_json(metadata_path, {"version": 1, "signature": signature, "state": state,
                                   "completed_count": len(completed),
                                   "error_count": sum(r["status"] in ERROR_STATUSES for r in records)})
        return records

    # A complete empty checkpoint is valid and protects the run configuration.
    save("running")
    progress = progress or (lambda value: print(value, file=sys.stderr, flush=True))
    consecutive_errors = 0
    try:
        for index, question in enumerate(questions):
            previous = completed.get(index)
            if previous and not (retry_errors and previous["status"] in ERROR_STATUSES):
                continue
            result = assistant.ask(question, retrieve_only=retrieve_only)
            result["batch_index"] = index
            completed[index] = result
            save("running")
            progress(f"[{index + 1}/{len(questions)}] {result['status']}")
            consecutive_errors = consecutive_errors + 1 if result["status"] in ERROR_STATUSES else 0
            if result.get("error") in STOP_ERRORS or consecutive_errors >= 3:
                progress("Batch stopped after local LLM/configuration errors; results saved. Fix the issue, then use --resume --retry-errors.")
                return save("stopped_on_error")
        return save("complete")
    except BaseException:
        # Ctrl+C and unexpected errors keep earlier successful results.
        save("interrupted")
        raise
