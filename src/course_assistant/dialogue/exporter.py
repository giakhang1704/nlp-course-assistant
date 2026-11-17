"""Part III output: input, context used, interpretation and grounded answer."""
import json
from pathlib import Path
from ..exporters import write_jsonl, write_lines


def export_dialogue(records, output_dir):
    output_dir = Path(output_dir)
    write_jsonl(output_dir / "dialogue-trace.jsonl", records)
    lines = []
    for record in records:
        lines += [f"SESSION: {record['session_id']} | TURN: {record['turn_id']}",
                  "USER: " + record["original_question"],
                  "CONTEXT_USED: " + json.dumps(record["context_used"], ensure_ascii=False),
                  "RESOLUTION: " + json.dumps(record["resolution"], ensure_ascii=False),
                  "EFFECTIVE_QUESTION: " + record["effective_question"],
                  "GENERATION_QUESTION: " + record.get("generation_question", record["effective_question"]),
                  "ORIGINAL_PARSE: " + record["parse_result"]["parse_tree"],
                  "STATUS: " + record["status"],
                  "ANSWER: " + (record["answer"] or record["error"] or "Chỉ chạy retrieval; chưa sinh câu trả lời."),
                  "SOURCES: " + json.dumps(record["citations"] or record["source_map"], ensure_ascii=False), ""]
    write_lines(output_dir / "dialogue.txt", lines)


def validate_dialogues(values):
    sessions = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Each dialogue must be a JSON object")
        session = value.get("session_id")
        turns = value.get("turns")
        if not isinstance(session, str) or not session.strip() or len(session) > 128 or session in sessions:
            raise ValueError("Dialogue session IDs must be unique, non-empty strings of at most 128 characters")
        if not isinstance(turns, list) or not turns or any(not isinstance(q, str) for q in turns):
            raise ValueError("Dialogue turns must be a non-empty list of question strings")
        sessions.add(session)
    return values


def check_destination(output_dir):
    if any((Path(output_dir) / name).exists() for name in ("dialogue.txt", "dialogue-trace.jsonl", "dialogue-run.json")):
        raise ValueError("Dialogue output already exists; choose a new --output-dir")


def run_dialogues(manager, dialogues, output_dir, retrieve_only=False, progress=None):
    from ..rag.batch import run_signature, STOP_ERRORS, ERROR_STATUSES
    from ..exporters import write_json
    dialogues = validate_dialogues(dialogues)
    check_destination(output_dir)
    records = []
    questions = [q for d in dialogues for q in d["turns"]]
    signature = run_signature(manager.assistant, questions, retrieve_only)
    signature["sessions"] = [d["session_id"] for d in dialogues]
    signature["max_context_turns"] = manager.resolver.max_context_turns
    signature["pending_turns"] = manager.pending_turns

    def save(state):
        export_dialogue(records, output_dir)
        write_json(Path(output_dir) / "dialogue-run.json", {"state": state, "signature": signature,
            "saved_turns": len(records), "total_turns": len(questions), "retrieve_only": retrieve_only})

    save("running")
    errors = 0
    try:
        for dialogue in dialogues:
            for question in dialogue["turns"]:
                result = manager.ask(question, dialogue["session_id"], retrieve_only)
                records.append(result)
                save("running")
                if progress:
                    progress(f"[{len(records)}/{len(questions)}] {dialogue['session_id']}: {result['status']}")
                errors = errors + 1 if result["status"] in ERROR_STATUSES else 0
                if result["error"] in STOP_ERRORS or errors >= 3:
                    save("stopped_on_error")
                    return records
        save("complete")
    except BaseException:
        save("interrupted")
        raise
    return records
