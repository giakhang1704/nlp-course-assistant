"""Labeled development retrieval metrics, never reported as LLM answer accuracy."""
from collections import Counter
from ..knowledge.loaders import read_jsonl
from ..exporters import write_json, write_jsonl, write_lines


def evaluate_retrieval(assistant, path, output_dir):
    from pathlib import Path
    output_dir = Path(output_dir)
    path = Path(path)
    known_documents = {c["document_id"] for c in assistant.retriever.chunks}
    records, recalls, context_recalls, reciprocal_ranks = [], [], [], []
    for case in read_jsonl(path):
        gold = set(case["relevant_documents"])
        if not gold <= known_documents:
            raise ValueError(f"Unknown gold document IDs for {case['id']}: {sorted(gold - known_documents)}")
        result = assistant.ask(case["question"], retrieve_only=True)
        retrieved = [h["document_id"] for h in result["retrieved_chunks"]]
        # Rank documents once even if multiple chunks from one document are returned.
        ranked = list(dict.fromkeys(retrieved))
        recall = len(gold & set(ranked)) / len(gold) if gold else None
        context_docs = {source["document_id"] for source in result["source_map"].values()}
        context_recall = len(gold & context_docs) / len(gold) if gold else None
        rr = next((1 / i for i, doc in enumerate(ranked, 1) if doc in gold), 0.0) if gold else None
        if gold:
            recalls.append(recall)
            context_recalls.append(context_recall)
            reciprocal_ranks.append(rr)
        records.append({"id": case["id"], "question": case["question"], "route": result["route"],
                        "expected_status": case["expected_status"], "actual_status": result["status"],
                        "status_correct": case["expected_status"] == result["status"],
                        "relevant_documents": sorted(gold), "retrieved_documents": ranked,
                        "document_recall_at_k": recall, "reciprocal_rank": rr})
        records[-1]["context_document_recall"] = context_recall
    summary = {"scope": "Retrieval/evidence policy evaluation only; no LLM calls or answer accuracy.",
               "input_file": str(path),
               "case_count": len(records), "answerable_cases": len(recalls), "top_k": assistant.config["top_k"],
               "document_recall_at_k": sum(recalls) / len(recalls) if recalls else None,
               "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
               "context_document_recall": sum(context_recalls) / len(context_recalls) if context_recalls else None,
               "status_accuracy": sum(r["status_correct"] for r in records) / len(records) if records else None,
               "routes": dict(Counter(r["route"] for r in records)),
               "failures": [r for r in records if not r["status_correct"] or
                             (r["document_recall_at_k"] is not None and r["document_recall_at_k"] < 1) or
                             (r["context_document_recall"] is not None and r["context_document_recall"] < 1)],
               "limitations": ["Use input provenance to distinguish development cases from synthetic test candidates; no independent external benchmark is implied.",
                               "Lexical scores and rule checks do not guarantee semantic support.",
                               "Generation needs live-model correctness, groundedness and hallucination review."]}
    write_json(output_dir / "retrieval-evaluation.json", summary)
    write_jsonl(output_dir / "retrieval-evaluation-cases.jsonl", records)
    write_lines(output_dir / "retrieval-evaluation.txt", [
        summary["scope"], f"Cases: {len(records)}; answerable: {len(recalls)}",
        f"Document Recall@{summary['top_k']}: {summary['document_recall_at_k']}",
        f"Context document recall: {summary['context_document_recall']}",
        f"MRR: {summary['mrr']}", f"Evidence-policy status accuracy: {summary['status_accuracy']}",
        "Failures:", *[f"{r['id']}: {r['question']} | {r['actual_status']}" for r in summary["failures"]]])
    return summary
