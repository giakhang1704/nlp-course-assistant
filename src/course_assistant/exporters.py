"""UTF-8 file contracts: one parse tree per input line, atomic writes."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import os


def write_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for line in lines:
                stream.write(line.rstrip("\n") + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, value):
    write_lines(path, [json.dumps(value, ensure_ascii=False, indent=2)])


def write_jsonl(path, values):
    write_lines(path, (json.dumps(value, ensure_ascii=False) for value in values))


def read_sentences(path):
    # Empty interior lines are actual input and must receive ().
    return Path(path).read_text(encoding="utf-8-sig").splitlines()


def export_parse_results(results, output_dir):
    destination = Path(output_dir)
    write_lines(destination / "parse-results.txt", (r["parse_tree"] for r in results))
    write_jsonl(destination / "parser-api.jsonl", results)
    write_jsonl(destination / "semantic.jsonl", (
        {"sentence": r["sentence"], "status": r["status"], "semantic_frame": r.get("semantic_frame")}
        for r in results))
    write_jsonl(destination / "query.jsonl", (
        {"sentence": r["sentence"], "status": r["status"], "query": r.get("query")}
        for r in results))
    export_text_views(results, destination)


def _frame_text(result):
    frame = result.get("semantic_frame") or {}
    return frame.get("logical_form") or f"() [{result['status']}]"


def export_text_views(results, destination):
    """Human-readable views of section 4.1: semantic, intent/entity, query.

    Each file keeps one line per input sentence, in input order.
    """
    write_lines(destination / "semantic.txt", (
        f"{r['sentence']}\t{_frame_text(r)}" for r in results))

    def intent_entity(r):
        frame = r.get("semantic_frame") or {}
        intent = frame.get("intent", r.get("intent_type", "UNKNOWN")) if r["is_valid"] else "UNKNOWN"
        entities = ", ".join(f"{e['type']}={e['id']}" + ("*" if e.get("implicit") else "")
                             for e in r.get("entities", []))
        return (f"{r['sentence']}\tsentence_type={frame.get('sentence_type')}"
                f"\tintent={intent}\tattribute={frame.get('attribute')}"
                f"\tquestion_type={frame.get('question_type')}\tentities=[{entities}]"
                f"\tneeds_context={r.get('needs_context')}")
    write_lines(destination / "intent-entity.txt", (intent_entity(r) for r in results))

    def query_line(r):
        query = r.get("query")
        if not query:
            return f"{r['sentence']}\t(no query: {r['status']})"
        entities = ",".join(e["id"] for e in query.get("entities", []))
        return (f"{r['sentence']}\tmode={query.get('mode')}\tintent={query.get('intent')}"
                f"\tattribute={query.get('attribute')}\toperation={query.get('operation')}"
                f"\tentities=[{entities}]\tsources={','.join(query.get('preferred_sources') or [])}"
                f"\ttext={' '.join(str(query.get('text') or '').split())}")
    write_lines(destination / "query.txt", (query_line(r) for r in results))
