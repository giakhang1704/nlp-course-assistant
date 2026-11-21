"""Conservative evidence admission; scores are not domain or truth probabilities."""
import re
from ..retrieval.lexical import normalize, meaningful_terms
from .question_analysis import unresolved_references


def preliminary_decision(result):
    if result["query"] is None:
        return result["status"], "Câu hỏi rỗng hoặc vượt giới hạn đầu vào."
    if result["ambiguity"]:
        return "clarification_needed", "Câu hỏi có nhiều cách hiểu; vui lòng nêu rõ yêu cầu."
    references = unresolved_references(result["sentence"])
    if result["needs_context"] or references:
        surface = references[0]["surface"] if references else "đối tượng được nhắc tới"
        return "clarification_needed", f"Chưa đủ ngữ cảnh để xác định “{surface}”. Bạn nêu rõ đối tượng hoặc gửi lại ví dụ nhé?"
    return None


def evidence_decision(result, hits, config, manifest, query=None):
    query = query if query is not None else result["query"]
    question = normalize(query["original_question"])
    attribute = query.get("attribute")
    if query.get("intent") == "GET_INSTRUCTOR" and not attribute:
        attribute = "instructor"
    if attribute == "chapter_count" or re.search(r"(mấy|bao nhiêu) chương", question):
        if manifest.get("conflicts"):
            return {"status": "conflicting_information", "reason": "Các nguồn không thống nhất: FAQ ghi 11 chương, phần đầu lịch mẫu ghi 12 chương.", "selected": []}
    admitted = [h for h in hits if h.get("field_match") or h.get("scope_match") or h.get("document_match") or
                (h.get("technical_name_weight", 0) >= 0.25 and h["lexical_score"] >= config["min_score"] / 2) or
                (h["lexical_score"] >= config["min_score"] and
                 (h["matched_terms"] or (attribute and h["metadata"].get("field") == attribute)))]
    # Content matches are valid evidence candidates too. Title/keyword matches
    # influence TF-IDF ranking; they must not exclude relevant body-only facts.
    hints = query.get("retrieval_hints", {})
    groups = hints.get("field_groups", [])
    if query.get("mode") == "raw_question":
        admitted = [h for h in admitted if h.get("field_match") or h.get("scope_match") or h.get("document_match") or h.get("anchor_phrase_matches")
                    or h.get("body_phrase_matches") or h.get("technical_name_weight", 0) >= 0.25
                    or (h["anchor_matches"] and h.get("query_term_coverage", 0) >= config.get("min_query_coverage", 0.3))
                    or h.get("query_term_coverage", 0) >= config.get("min_body_coverage", 0.5)]
    if hints.get("scope"):
        admitted = [h for h in admitted if h.get("scope_match")]
    if groups:
        fields = {field for group in groups for field in group}
        admitted = [h for h in admitted if h["metadata"].get("field", "").casefold() in fields]
        if any(not any(h["metadata"].get("field", "").casefold() in group for h in admitted) for group in groups):
            admitted = []
    # Exact requested factual fields must exist in evidence, not merely mention NLP.
    if not groups and attribute and query.get("intent") in {"GET_COURSE_INFO", "GET_INSTRUCTOR", "GET_LO"}:
        aliases = {"course_name": ["course_name", "course_name_en"],
                   "instructor": ["instructor", "instructor_name"],
                   "objective": ["course_description"], "duration_weeks": ["format"],
                   "assessment": ["class_attention_and_assignments", "major_assignment", "midterm_exam", "final_exam"],
                   "learning_outcomes": ["learning_outcomes", "detailed_learning_outcomes"]}
        fields = aliases.get(attribute, [attribute])
        exam = next((str(e["value"]).lower() for e in query.get("entities", []) if e.get("type") == "exam"), None)
        if attribute == "duration_minutes" and exam in {"midterm", "final"}:
            fields = [exam]
        if attribute == "assessment_weight" and exam in {"midterm", "final"}:
            fields = [exam + "_exam"]
        admitted = [h for h in admitted if h["metadata"].get("field") in fields
                    or any(re.search(r"(?m)^" + re.escape(field) + r":", h["text"]) for field in fields)]
    if "exam_calendar" in hints.get("requirements", []):
        # A duration in minutes is not a calendar date or a starting time.
        exams = hints.get("exams", [])
        admitted = [h for h in admitted if any(re.search(
            r"(?mi)^" + exam + r"_(?:date|datetime|start_time):\s*\S", h["text"]) for exam in exams)]
    if query.get("intent") == "GET_DEADLINE" or "deadline" in hints.get("requirements", []) or re.search(r"deadline|hạn nộp|thời hạn", question):
        admitted = [h for h in admitted if re.search(r"(?i)(deadline|due_date|hạn nộp|thời hạn)\s*[:=]\s*\S", h["text"])]
    if attribute in {"weekday", "room"}:
        admitted = [h for h in admitted if re.search(r"(?mi)^" + attribute + r":\s*\S", h["text"])]
    if query.get("operation") == "locate_slide" or "source_slides" in hints.get("requirements", []):
        admitted = [h for h in admitted if h["metadata"].get("source_slides")]
    for entity in query.get("entities", []):
        if entity.get("type") == "week":
            number = re.search(r"\d+", str(entity.get("value")))
            if number:
                admitted = [h for h in admitted if h["metadata"].get("week") == int(number[0])]
        elif entity.get("type") == "chapter":
            number = re.search(r"\d+", str(entity.get("value")))
            if number:
                chapter = int(number[0])
                admitted = [h for h in admitted if h["metadata"].get("chapter") == chapter
                            or chapter in h["metadata"].get("chapters", [])]
        elif entity.get("type") == "assignment_part" and query.get("intent") == "GET_ASSIGNMENT":
            number = re.search(r"\d+", str(entity.get("value")))
            if number:
                admitted = [h for h in admitted if h["metadata"].get("part") == int(number[0])]
    if not meaningful_terms(question) or not admitted:
        return {"status": "insufficient_information", "reason": "Không tìm thấy thông tin này trong dữ liệu môn học.", "selected": []}
    return {"status": "evidence_available", "reason": "Candidate evidence passed lexical/slot checks; semantic support still requires validation.", "selected": admitted}
