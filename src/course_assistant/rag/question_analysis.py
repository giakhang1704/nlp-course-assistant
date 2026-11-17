"""Small, explicit Part II retrieval hints; never manufacture a CFG parse."""
from copy import deepcopy
import re
import unicodedata


FIELD_LABELS = {
    "course_id": "mã môn mã học phần course code",
    "course_name": "tên môn tên học phần course name",
    "course_name_en": "tên tiếng Anh English course name",
    "semester": "học kỳ semester",
    "credits": "tín chỉ credits", "ects": "ECTS European credits",
    "chapter_count": "số chương chapter count",
    "department": "khoa department", "format": "số tuần lịch học course format weeks",
    "total_course_hours": "tổng số giờ khối lượng toàn môn total course hours",
    "lecture_hours": "giờ tiết lý thuyết lecture hours",
    "major_assignment_hours": "giờ bài tập lớn major assignment hours",
    "self_study_hours": "giờ tự học self study hours",
    "midterm_exam": "tỷ trọng tỷ lệ điểm phần trăm thi giữa kỳ midterm exam assessment weight percent",
    "final_exam": "tỷ trọng tỷ lệ điểm phần trăm thi cuối kỳ final exam assessment weight percent",
    "midterm": "hình thức thời lượng thi giữa kỳ midterm exam format duration minutes",
    "final": "hình thức thời lượng thi cuối kỳ final exam format duration minutes",
    "ai_usage": "quy định sử dụng AI khai báo báo cáo",
}


def normalized(text):
    return unicodedata.normalize("NFC", text).casefold()


def unresolved_references(question):
    """Match complete noun phrases, preferring 'học phần' over inner 'phần'."""
    text = normalized(question)
    nouns = (r"học phần|môn học|bài tập lớn|bài tập|chủ đề|nội dung|mô hình|"
             r"thuật toán|phương pháp|tài liệu|chương|phần|môn|bài|cái|bước|tác vụ|cách đo|phép đo|cuốn|quyển|mục từ|ví dụ|con số|cặp|vector|khái niệm|trợ động từ")
    pointer = r"vừa (?:đề cập|nêu|nhắc|nói|kể|xác định)|trước đó|này|đó|ấy|trên"
    pattern = rf"\b(?P<noun>{nouns})\s+(?P<pointer>{pointer})\b|\b(?<!chính )nó\b|\b(?:mục từ bên dưới|mục đầu|trường hợp cuối|cụm được gạch chân)\b"
    references = []
    for match in re.finditer(pattern, text):
        noun = match.group("noun")
        if noun in {"học phần", "môn học", "môn"} and match.group("pointer") == "này":
            continue  # The assistant serves one course, already known locally.
        prefix = text[:match.start()]
        if match[0] == "nó" and re.search(r"\bkhi .+ là .+ và khi\s*$", prefix):
            continue  # Parallel readings of an explicitly named local subject.
        if (noun in {"phần", "bài", "bài tập", "bài tập lớn"} or match[0] == "nó") and re.search(r"bài tập lớn|btl\d+|\bfinal\b|\bmidterm\b|thi (?:giữa|cuối) kỳ", prefix):
            continue  # An explicit local assignment/exam is already present.
        kind = ("chapter" if noun == "chương" else
                "assignment" if noun in {"bài", "bài tập", "bài tập lớn"} else
                "algorithm" if noun == "thuật toán" else "topic")
        references.append({"type": kind, "surface": match[0]})
    return references


def enrich_query(query):
    """Add auditable lexical/field hints to a copy of the parser's handoff.

    Field groups express alternatives within a group; every requested group must
    have evidence. Grammar validity, intent, entities and parse tree stay intact.
    """
    query = deepcopy(query)
    text = normalized(query["original_question"])
    groups, requirements = [], []

    def fields(*names):
        group = list(names)
        if group not in groups:
            groups.append(group)

    if re.search(r"\b(?:mã (?:môn|học phần)|course (?:id|code))\b", text):
        fields("course_id")
    if query.get("intent") == "GET_LO" and not re.search(r"\b(?:chương|tuần) (?:nào|mấy)\b", text):
        # "L.O.2.5" is split by the tokenizer; the LO list field is the evidence.
        fields("detailed_learning_outcomes", "learning_outcomes")
    if re.search(r"\b(?:tên.*tiếng anh|english (?:course )?name)\b", text):
        fields("course_name_en")
    if re.search(r"\b(?:học kỳ|semester)\b", text):
        fields("semester")
    if re.search(r"\bects\b", text):
        fields("ects")
    # These requests need the actual field, not a nearby description of a
    # lecturer, assessment, attendance recommendation or submission milestone.
    specific_fields = {
        "office_hours": r"office hours|tiếp sinh viên|giờ tiếp (?:sv|sinh viên)",
        "phone": r"số điện thoại|phone number",
        "room": r"phòng (?:máy|học)|địa điểm (?:học|thực hành)",
        "submission_url": r"(?:trang|link|url|đường dẫn).*nộp|nộp.*(?:url|link)",
        "tuition": r"học phí|tuition",
        "absence_limit": r"vắng.*(?:tối đa|cấm thi|bao nhiêu|mấy)|(?:tối đa|bao nhiêu|mấy).*vắng",
        "student_grades": r"(?:mssv|mã số sinh viên).*điểm|điểm.*(?:mssv|mã số sinh viên)",
        "group_members": r"nhóm (?:số )?\d+.*(?:thành viên|mssv)",
    }
    for field, pattern in specific_fields.items():
        if re.search(pattern, text):
            fields(field)
    if re.search(r"\b(?:giờ|tiết|hours?)\b", text):
        if re.search(r"\b(?:lecture|lý thuyết)\b", text):
            fields("lecture_hours")
        if re.search(r"\b(?:self[- ]study|tự học)\b", text):
            fields("self_study_hours")
        if re.search(r"\b(?:bài tập lớn|major assignment)\b", text) and not re.search(r"nộp|hạn|deadline", text):
            fields("major_assignment_hours")
        if not groups and re.search(r"\b(?:tổng|total)\b", text):
            fields("total_course_hours")
    exams = []
    for exam, pattern in [("midterm", r"\b(?:midterm|giữa kỳ)\b"),
                          ("final", r"\b(?:final|cuối kỳ)\b")]:
        if re.search(pattern, text):
            exams.append(exam)
    calendar = bool(exams and re.search(r"ngày|tháng|giờ (?:bắt đầu|thi)|lịch thi|exam date|start time", text))
    if calendar:
        requirements.append("exam_calendar")
        for exam in exams:
            if re.search(r"ngày|tháng|lịch thi|exam date", text):
                fields(exam + "_date", exam + "_datetime")
            if re.search(r"giờ (?:bắt đầu|thi)|start time", text):
                fields(exam + "_start_time", exam + "_datetime")
    else:
        weight = re.search(r"tỷ (?:lệ|trọng)|trọng số|phần trăm|percent|weight|đóng góp|chiếm.*điểm", text)
        duration = re.search(r"phút|minutes?|thời lượng|duration|hình thức|format|open.book", text)
        duration = duration or re.search(r"(?:mở|dùng|sử dụng) tài liệu", text)
        if weight and re.search(r"bài tập lớn|major assignment", text):
            fields("major_assignment")
        if weight and re.search(r"tham gia lớp|class attention", text):
            fields("class_attention_and_assignments")
        for exam in exams:
            if weight or query.get("attribute") == "assessment_weight":
                fields(exam + "_exam")
            if duration or query.get("attribute") == "duration_minutes":
                fields(exam)
    if re.search(r"deadline|hạn nộp|thời hạn|chốt nộp", text):
        requirements.append("deadline")
    if query.get("operation") == "locate_slide" or re.search(r"(?:số|trang|slide number).*slide|số slide", text):
        requirements.append("source_slides")
    if query.get("mode") == "raw_question" and re.search(r"\b(?:dùng|sử dụng|using|use) ai\b", text) and re.search(r"khai báo|báo cáo|quy định|declar", text):
        fields("ai_usage")
    scope = {}
    for name, pattern in [("week", r"\b(?:tuần|week)\s+(\d+)\b"),
                          ("chapter", r"\b(?:chương|chapter)\s+(\d+)\b")]:
        numbers = {int(n) for n in re.findall(pattern, text)}
        if len(numbers) == 1:
            scope[name] = numbers.pop()
    first_week = bool(re.search(r"\btuần đầu(?: tiên)?\b", text))
    last_week = bool(re.search(r"\btuần cuối(?: cùng)?\b", text))
    if "week" not in scope and first_week != last_week:
        scope["week"] = "first" if first_week else "last"
    query["retrieval_hints"] = {"field_groups": groups, "requirements": requirements,
                                "exams": exams, "scope": scope}
    if re.search(r"\bintent\b", text) and re.search(r"btl|bài tập lớn|assignment", text):
        query["retrieval_hints"]["sections"] = ["suggested_intents"]
    if re.search(r"\bcheckpoint\b", text):
        query["retrieval_hints"]["milestone"] = {
            "kind": "checkpoint", "ordinal": "first" if re.search(r"đầu tiên|first", text)
            else "last" if re.search(r"cuối cùng|last", text) else None}
    return query
