"""Join gold labels to actual traces; human grading remains explicit and unfilled."""
import csv
import json
from pathlib import Path
from ..exporters import write_json, write_lines
from ..knowledge.loaders import read_jsonl

REVIEW_FIELDS = ["id", "question", "category", "expected_status", "actual_status", "reference_answer",
                 "answer", "citations", "context", "correct", "claims_total", "claims_supported",
                 "citations_checked", "citations_supported", "notes"]
MANUAL_FIELDS = ["correct", "claims_total", "claims_supported", "citations_checked", "citations_supported", "notes"]
TECHNICAL = {"generation_error", "llm_unavailable", "not_run"}


def join_cases(labels_path, trace_path):
    labels = read_jsonl(labels_path)
    if not labels or len({r["id"] for r in labels}) != len(labels):
        raise ValueError("Labels must be nonempty and have unique IDs")
    if len({r["question"] for r in labels}) != len(labels):
        raise ValueError("QA review requires unique labeled questions")
    gold = {r["question"]: r for r in labels}
    traces = {}
    for result in read_jsonl(trace_path):
        question = result["question"]
        if question not in gold or question in traces:
            raise ValueError("Trace has an unknown or duplicate question; use matching labels and one run")
        traces[question] = result
    return labels, traces


def review_identity(case, result):
    return {"id": case["id"], "question": case["question"], "category": case.get("category", ""),
            "expected_status": "answered" if case["expected_status"] == "retrieved" else case["expected_status"],
            "actual_status": result.get("status", "not_run"),
            "reference_answer": case.get("reference_answer", ""), "answer": result.get("answer") or "",
            "citations": json.dumps(result.get("citations", []), ensure_ascii=False),
            "context": result.get("context", "")}


def prepare_review(labels_path, trace_path, output_path):
    labels, traces = join_cases(labels_path, trace_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Review file already exists; choose a new path to preserve manual grades")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting another review.
    with output_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in labels:
            writer.writerow({**review_identity(case, traces.get(case["question"], {})),
                             **{key: "" for key in MANUAL_FIELDS}})
    return {"review_file": str(output_path), "case_count": len(labels), "trace_count": len(traces),
            "grading": "Fill correct (0/1) and claim/citation counts manually; blanks remain ungraded."}


def score_qa(labels_path, trace_path, output_dir, review_path=None):
    labels, traces = join_cases(labels_path, trace_path)
    if any(result.get("status") == "retrieved" for result in traces.values()):
        raise ValueError("Trace contains retrieval-only results; run QA without --retrieve-only before scoring answers")
    rows = []
    reviews = {}
    if review_path:
        with Path(review_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["id"] in reviews:
                    raise ValueError("Duplicate ID in review file")
                reviews[row["id"]] = row
        if set(reviews) != {r["id"] for r in labels}:
            raise ValueError("Review IDs do not match the gold labels")
    for case in labels:
        result = traces.get(case["question"], {})
        identity = review_identity(case, result)
        manual = reviews.get(case["id"], {})
        def normalized(value):
            return str(value).replace("\r\n", "\n").replace("\r", "\n")
        if manual and any(normalized(manual.get(k)) != normalized(v) for k, v in identity.items()):
            raise ValueError(f"Review no longer matches trace/reference for {case['id']}; prepare a new review")
        scores = {}
        for key in MANUAL_FIELDS[:-1]:
            value = manual.get(key, "").strip()
            if not value:
                scores[key] = None
                continue
            if not value.isdigit():
                raise ValueError(f"{case['id']}: {key} must be a nonnegative integer")
            scores[key] = int(value)
        if scores["correct"] not in {None, 0, 1}:
            raise ValueError("correct must be 0 or 1")
        for total, supported in [("claims_total", "claims_supported"), ("citations_checked", "citations_supported")]:
            if (scores[total] is None) != (scores[supported] is None):
                raise ValueError(f"{case['id']}: fill both {total} and {supported}, or leave both blank")
            if scores[total] is not None and scores[supported] > scores[total]:
                raise ValueError(f"{case['id']}: supported count exceeds total")
        rows.append({**identity, **scores, "notes": manual.get("notes", "")})

    answerable = [r for r in rows if r["expected_status"] == "answered"]
    missing = [r for r in rows if r["expected_status"] == "insufficient_information"]
    answered = [r for r in rows if r["actual_status"] == "answered"]
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    # Failures/refusals on answerable questions count as incorrect; never drop them.
    graded = [r for r in answerable if r["actual_status"] not in TECHNICAL and
              (r["actual_status"] != "answered" or r["correct"] is not None)]
    correct = sum(r["actual_status"] == "answered" and r["correct"] == 1 for r in answerable)
    claims = [r for r in answered if r["claims_total"] is not None]
    citations = [r for r in answered if r["citations_checked"] is not None]
    context_cases = [r for r in rows if r["expected_status"] == "clarification_needed"]
    reviewed_context = [r for r in context_cases if r["correct"] is not None]
    summary = {"scope": "QA status metrics and content scores from the supplied review file.",
               "labels_file": str(labels_path), "trace_file": str(trace_path),
               "case_count": len(rows), "trace_count": len(traces),
               "unrun_count": sum(r["actual_status"] == "not_run" for r in rows),
               "technical_error_count": sum(r["actual_status"] in TECHNICAL for r in rows),
               "status_accuracy": ratio(sum(r["actual_status"] == r["expected_status"] for r in rows), len(rows)),
               "overall_correctness": ratio(sum(r["correct"] == 1 for r in rows), len(rows)) if all(r["correct"] is not None for r in rows) else None,
               "context_insufficiency_cases": len(context_cases),
               "context_insufficiency_handling_accuracy": ratio(sum(r["correct"] == 1 for r in context_cases), len(context_cases)) if len(reviewed_context) == len(context_cases) else None,
               "answer_coverage": ratio(sum(r["actual_status"] == "answered" for r in answerable), len(answerable)),
               "abstention_status_recall": ratio(sum(r["actual_status"] == "insufficient_information" for r in missing), len(missing)),
               "answerable_cases": len(answerable), "answerable_cases_scored": len(graded),
               "answer_accuracy": ratio(correct, len(answerable)) if len(graded) == len(answerable) else None,
               "answer_accuracy_on_scored_cases": ratio(correct, len(graded)),
               "answered_cases": len(answered), "claim_reviewed_answers": len(claims),
               "groundedness_on_reviewed_claims": ratio(sum(r["claims_supported"] for r in claims), sum(r["claims_total"] for r in claims)),
               "citation_reviewed_answers": len(citations),
               "citation_precision_on_reviewed_citations": ratio(sum(r["citations_supported"] for r in citations), sum(r["citations_checked"] for r in citations)),
               "limitations": ["Status/citation IDs do not establish factual correctness or entailment.",
                               "Content metrics require completed review scores; null means unavailable, not zero.",
                               "Reference answers permit equivalent wording; gold document alternatives may be incomplete."]}
    output_dir = Path(output_dir)
    write_json(output_dir / "qa-evaluation.json", summary)
    write_lines(output_dir / "qa-evaluation.txt", [summary["scope"], *[
        f"{key}: {value}" for key, value in summary.items() if key != "limitations"]])
    return summary
