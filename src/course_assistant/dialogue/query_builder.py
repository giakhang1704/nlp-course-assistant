"""Translate bindings into retrieval hints, keeping the original parse intact."""
import re
from ..rag.question_analysis import enrich_query
from .catalog import token_text


def build_dialogue_query(effective_question, effective_parse, targets, catalog, resolved=False):
    query = enrich_query(effective_parse["query"])
    text = token_text(effective_question)
    resource = bool(re.search(r"\btài liệu\b|\breferences?\b|\bresources?\b", text))
    if resource and targets:
        # Week metadata belongs to schedules, not to slides or textbooks.
        query["entities"] = [e for e in query["entities"] if e["type"] not in {"week", "chapter"}]
        query["retrieval_hints"]["scope"] = {}
        resources = catalog.resource_documents(targets)
        # Keep the schedule premise as evidence for week -> chapter -> slides.
        # Its sample label must accompany this association in the model context.
        premises = sorted({d for t in targets if t.type == "week" for d in catalog.supporting_documents(t)})
        query["retrieval_hints"]["document_ids"] = sorted(set(resources + premises)) if resources else []
        query["retrieval_hints"]["document_groups"] = [group for group in (premises, resources) if group]
        query["preferred_sources"] = ["resources.txt"]
        query["dialogue_request"] = "resources"
        weeks = [t for t in targets if t.type == "week"]
        if weeks and resources:
            chapters = sorted({related.label for week in weeks for related in catalog.related(week)})
            sample = any(catalog.documents[d]["metadata"]["is_sample"] for d in premises)
            schedule = "lịch mẫu" if sample else "lịch môn học"
            query["scope_resolution"] = {"weeks": [w.id for w in weeks], "chapters": chapters,
                                         "source_document_ids": premises, "is_sample": sample}
            query["generation_question"] = (f"Theo {schedule}, {', '.join(w.label for w in weeks)} thuộc "
                f"{', '.join(chapters)}. Tài liệu nguồn của {', '.join(chapters)} có trong dữ liệu là file nào?")
    elif (targets and all(t.type == "learning_outcome" for t in targets)
          and re.search(r"\b(?:chương|tuần) (?:nào|mấy)\b", text)):
        # An LO list does not say where it is taught, and LO numbers (LO2.5)
        # are not chapter numbers: search chapter summaries and FAQ by the LO
        # description carried in the effective question.
        query["retrieval_hints"]["document_ids"] = sorted(
            d for d, doc in catalog.documents.items() if doc["type"] in {"topic_summary", "faq"})
    elif targets and not query["retrieval_hints"].get("milestone") and (resolved or all(t.type in {"assignment", "assignment_part"} for t in targets)):
        groups = [catalog.supporting_documents(target) for target in targets if target.type != "course"]
        if groups:
            query["retrieval_hints"]["document_ids"] = sorted({d for group in groups for d in group})
            query["retrieval_hints"]["document_groups"] = groups
    query["contextual_targets"] = [e.to_dict() for e in targets]
    return query
