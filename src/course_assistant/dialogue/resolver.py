"""Deterministic, bounded reference binding. No LLM calls or guessed antecedents."""
from copy import deepcopy
from dataclasses import dataclass, field
import re
from .catalog import find_spans, token_text
from .state import MemoryEntity

KINDS = {
    "chapter": {"chapter"}, "week": {"week"}, "assignment": {"assignment", "assignment_part"},
    "assignment_part": {"assignment_part"}, "topic": {"concept", "topic", "algorithm", "model"},
    "generic": {"concept", "topic", "algorithm", "model", "week", "chapter", "assignment", "assignment_part"},
    "pair": {"concept", "topic", "algorithm", "model"},
    "resource": {"resource"},
}
KINDS["generic"].add("resource")
KINDS["generic"].add("learning_outcome")
KINDS["generic"].add("step")
KINDS["step"] = {"step"}


@dataclass
class Resolution:
    question: str
    effective_question: str
    status: str = "resolved"
    reason: str | None = None
    bindings: dict = field(default_factory=dict)
    references: list = field(default_factory=list)
    unresolved_key: str | None = None
    required_kind: str = "generic"
    candidates: list = field(default_factory=list)


def compatible(entity, kind):
    return entity.type in KINDS.get(kind, KINDS["generic"])


def references(question, parsed):
    found, occupied = [], []

    def add(start, end, kind, relative=0, template=None):
        if any(start < b and end > a for a, b in occupied):
            return
        occupied.append((start, end))
        found.append({"key": f"{start}:{end}", "surface": question[start:end],
                      "start": start, "end": end, "kind": kind,
                      "relative": relative, "template": template})

    adjacent_step = re.search(r"\b(?:ngay )?(trước|sau) bước\b", token_text(question))
    if adjacent_step:
        add(0, len(question), "step", -1 if adjacent_step[1] == "trước" else 1,
            "Bước {target} trong quy trình là gì?")
    patterns = [
        (r"\bhai (?:cái|phương pháp|thuật toán|mô hình|chủ đề|khái niệm|trợ động từ|vector)(?: (?:đó|này|ấy|vừa nêu|vừa kể))?\b", "pair", 0),
        (r"\bchương (?:tiếp theo|kế tiếp|sau)\b", "chapter", 1),
        (r"\bchương (?:trước|trước đó)\b", "chapter", -1),
        (r"\b(?:chương|tuần|bài tập lớn|bài tập|bài|thuật toán|mô hình|phương pháp|chủ đề|nội dung|phần|cái|tác vụ|cách đo|phép đo|cuốn sách|cuốn|quyển sách|quyển|mục từ|ví dụ|bước|quy trình|cặp|con số) "
         r"(?:vừa (?:đề cập|nêu|nhắc|nói|xác định)|trước đó|này|đó|ấy|trên)\b", "auto", 0),
        (r"\bnó\b", "generic", 0),
        (r"\b(?:mục từ bên dưới|mục đầu|trường hợp cuối|cụm được gạch chân)\b", "generic", 0),
    ]
    for pattern, kind, relative in patterns:
        for start, end, _ in find_spans(question, pattern):
            noun = token_text(question[start:end])
            if noun.startswith("hai vector") and not re.search(r"\b(?:đó|này|ấy|vừa nêu|vừa kể)$", noun):
                continue  # A newly introduced plural is not a prior reference.
            if noun == "nó" and token_text(question[:start]).endswith("chính"):
                continue  # Local reflexive, not an unresolved discourse subject.
            if noun == "nó" and any(r["kind"] == "pair" for r in found):
                continue  # The explicit pair supplies the comparison's context.
            if kind == "auto":
                # Do not catch the inner 'phần này' of 'học phần này'.
                if noun.startswith("phần") and token_text(question[:start]).endswith("học"):
                    continue
                kind_value = ("resource" if noun.startswith(("cuốn", "quyển")) else "chapter" if noun.startswith("chương") else "week" if noun.startswith("tuần")
                              else "assignment" if noun.startswith("bài") else "topic" if noun.startswith(
                                  ("thuật toán", "mô hình", "phương pháp", "chủ đề")) else "generic")
            else:
                kind_value = kind
            if noun.startswith("hai vector"):
                kind_value = "topic"
            add(start, end, kind_value, relative)
    for ref in (parsed.get("semantic_frame") or {}).get("references", []):
        if ref.get("start") is None or ref.get("end") is None:
            continue
        if ref["id"] == "LAST_COURSE":
            add(ref["start"], ref["end"], "course")
        else:
            kind = {"LAST_CHAPTER": "chapter", "LAST_ASSIGNMENT": "assignment",
                    "LAST_ASSIGNMENT_PART": "assignment_part"}.get(ref["id"], "generic")
            add(ref["start"], ref["end"], kind)
    return sorted(found, key=lambda r: r["start"])


class ReferenceResolver:
    def __init__(self, catalog, max_context_turns=3):
        self.catalog = catalog
        self.max_context_turns = max_context_turns

    def resolve(self, question, parsed, state, overrides=None, final_overrides=()):
        refs = references(question, parsed)
        explicit = self.catalog.scan(question)
        if not refs and not explicit:
            text = token_text(question).strip(" ?.!，")
            kind, template = None, None
            if re.fullmatch(r"(?:còn )?(?:deadline|hạn nộp|thời hạn)(?: thì sao| là khi nào)?", text):
                kind, template = "assignment", "Deadline của {target} là khi nào?"
            elif re.fullmatch(r"(?:giải thích|nói) (?:kỹ|rõ|chi tiết) hơn(?: đi)?", text):
                kind, template = "generic", "Giải thích chi tiết về {target}."
            elif re.fullmatch(r"cho (?:(?:mình|tôi) )?(?:một )?ví dụ(?: đi)?", text):
                kind, template = "generic", "Cho ví dụ về {target} từ dữ liệu môn học."
            elif re.fullmatch(r"(?:còn )?tài liệu(?: ở đâu| đâu| thì sao)?", text):
                kind, template = "generic", "Tài liệu của {target} ở đâu?"
            if kind:
                refs = [{"key": "ellipsis", "surface": question, "start": 0, "end": len(question),
                         "kind": kind, "relative": 0, "template": template}]
        result = Resolution(question, question, references=refs)
        overrides = overrides or {}
        fresh = state.turn_id - state.last_grounded_turn <= self.max_context_turns
        for ref in refs:
            kind = ref["kind"]
            if ref["key"] in overrides:
                selected = deepcopy(overrides[ref["key"]])
            elif kind == "course":
                selected = [MemoryEntity("course", "CO3085", "môn NLP", ["course_info.course_id"])]
            elif kind == "pair":
                selected = deepcopy(state.comparison_pair) if fresh else []
            else:
                selected = [deepcopy(e) for e in state.focus if compatible(e, kind)] if fresh else []
                if not selected and fresh:
                    selected = [deepcopy(e) for e in state.related_entities if compatible(e, kind)]
                # Explicit names within this question can resolve a local
                # pronoun (e.g. an exam or assignment followed by 'bài này').
                local = [m["entity"] for m in explicit if (m["end"] <= ref["start"] or kind == "step") and compatible(m["entity"], kind)]
                if len({e.key for e in local}) == 1:
                    selected = [deepcopy(local[-1])]
                # 'phần này' names an assignment part when one is in focus,
                # rather than the whole assignment mentioned with it.
                parts = [e for e in selected if e.type == "assignment_part"]
                if len(selected) > 1 and len(parts) == 1 and token_text(ref["surface"]).startswith("phần"):
                    selected = parts
                # 'chương đó' after a concept: the chapter whose documents
                # support that concept, only when exactly one chapter does.
                if kind == "chapter" and not selected and fresh:
                    chapters = {key for e in state.focus for key in self.catalog.chapters_of(e)}
                    if len(chapters) == 1:
                        entity = deepcopy(self.catalog.entities[chapters.pop()])
                        entity.from_turn = state.turn_id
                        selected = [entity]
            need = 2 if kind == "pair" else 1
            if len(selected) != need:
                result.status = "clarification_needed"
                result.unresolved_key, result.required_kind = ref["key"], kind
                result.candidates = [e.to_dict() for e in selected]
                names = ", ".join(e.label for e in selected)
                subject = {"chapter": "chương", "week": "tuần", "assignment": "bài tập",
                           "assignment_part": "phần bài tập", "topic": "chủ đề/thuật toán",
                           "resource": "cuốn sách/tài liệu",
                           "pair": "hai đối tượng"}.get(kind, "đối tượng")
                result.reason = f"Bạn đang nói đến {subject} nào?" + (f" Các lựa chọn: {names}." if names else "")
                return result
            if ref["relative"] and ref["key"] not in final_overrides:
                ref["antecedents"] = [e.to_dict() for e in selected]
                if kind == "step":
                    sequence = next((keys for keys in self.catalog.process_steps.values() if selected[0].key in keys), [])
                    index = sequence.index(selected[0].key) + ref["relative"] if sequence else -1
                    if not 0 <= index < len(sequence):
                        result.status, result.reason = "insufficient_information", "Không có bước liền kề đó trong quy trình được cung cấp."
                        return result
                    result.bindings[ref["key"]] = [deepcopy(self.catalog.entities[sequence[index]])]
                    continue
                number = int(re.search(r"\d+", selected[0].id)[0]) + ref["relative"]
                entity = self.catalog.entities.get(("chapter", f"CH{number:02d}"))
                if entity is None or not entity.source_document_ids:
                    result.status = "insufficient_information"
                    result.reason = f"Không có dữ liệu cho Chương {number} trong knowledge base."
                    return result
                selected = [deepcopy(entity)]
            result.bindings[ref["key"]] = selected
        if parsed["needs_context"] and not refs:
            result.status, result.reason = "clarification_needed", "Bạn đang đề cập đến đối tượng nào?"
            result.unresolved_key = "ellipsis"
            return result
        effective = question
        for ref in reversed(refs):
            label = " và ".join(e.label for e in result.bindings[ref["key"]])
            replacement = ref["template"].format(target=label) if ref["template"] else label
            effective = effective[:ref["start"]] + replacement + effective[ref["end"]:]
        result.effective_question = effective
        return result
