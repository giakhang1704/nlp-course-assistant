"""Ground entity names and associations in the current lexicon and KB only."""
from copy import deepcopy
from bisect import bisect_right
import re
from ..semantics.entities import token_spans
from ..parsing.tokenizer import tokenize
from .state import MemoryEntity

FOCUS_TYPES = {"chapter", "week", "assignment", "assignment_part", "concept", "topic", "model", "algorithm", "resource",
               "learning_outcome"}
LO_DOCUMENTS = ("course_info.detailed_learning_outcomes", "course_info.learning_outcomes")


def find_spans(question, pattern):
    """Regex over normalized tokens, with offsets mapped back to original Unicode."""
    spans = token_spans(question)
    starts, offset = [], 0
    for token, _, _ in spans:
        starts.append(offset)
        offset += len(token) + 1
    text = " ".join(t for t, _, _ in spans)
    found = []
    for match in re.finditer(pattern, text):
        first = bisect_right(starts, match.start()) - 1
        last = bisect_right(starts, match.end() - 1) - 1
        begin, end = spans[first][1], spans[last][2]
        if begin is not None and end is not None:
            found.append((begin, end, match))
    return found


def token_text(text):
    return " ".join(tokenize(text))


def contains(text, alias):
    return bool(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text))


class EntityCatalog:
    def __init__(self, assistant):
        self.documents = {c["document_id"]: c for c in assistant.retriever.chunks}
        self.entities, self.aliases, self.process_steps = {}, [], {}
        for mapping in assistant.parser.grammar.lexicon.values():
            for tokens, value in mapping.items():
                if value["type"] not in FOCUS_TYPES:
                    continue
                key = value["type"], value["id"]
                if key not in self.entities:
                    self.entities[key] = MemoryEntity(value["type"], value["id"], " ".join(tokens))
                self.aliases.append((tuple(tokens), key))
        # Slide titles also cover knowledge beyond the grammar's vocabulary.
        for doc_id, doc in self.documents.items():
            if doc["type"] == "slide_topic":
                key = "topic", doc_id
                self.entities[key] = MemoryEntity("topic", doc_id, doc["title"], [doc_id])
                self.aliases.append((tokenize(doc["title"]), key))
            elif doc["type"] == "resource":
                key = "resource", doc_id
                self.entities[key] = MemoryEntity("resource", doc_id, doc["title"], [doc_id])
                self.aliases.append((tokenize(doc["title"]), key))
        # Named concepts remain distinct even when they share one slide card.
        # Derive vocabulary from KB metadata, never from test questions/answers.
        known = {a for a, _ in self.aliases}
        for doc_id, doc in self.documents.items():
            for keyword in doc["metadata"].get("keywords", []):
                tokens = tokenize(keyword)
                if not tokens or tokens in known or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9 -]*", keyword):
                    continue
                if len(tokens) == 1 and not contains(token_text(doc["title"]), token_text(keyword)) and not re.search(
                    r"\b" + re.escape(token_text(keyword)) + r"\s+(?:là|biểu diễn|có|đi với|chọn)\b", token_text(doc["text"])):
                    continue
                key = "concept", "KB:" + token_text(keyword)
                if key not in self.entities:
                    self.entities[key] = MemoryEntity("concept", key[1], keyword, [])
                    self.aliases.append((tokens, key))
                self.entities[key].source_document_ids.append(doc_id)
        for doc_id, doc in self.documents.items():
            sequence = re.search(r"gồm [^.:\n]*bước:\s*([^\n]+)", doc["text"], re.I)
            if not sequence:
                continue
            names = re.findall(r"\b([a-z]+(?: [a-z]+){1,2})\s+(?=chọn|xác định|chỉnh)", sequence[1])
            if len(names) < 2:
                continue
            self.process_steps[doc_id] = []
            for name in names:
                key = "step", doc_id + ":" + name
                self.entities[key] = MemoryEntity("step", key[1], name, [doc_id])
                self.aliases.append((tokenize(name), key))
                self.process_steps[doc_id].append(key)
                if name == "information ordering":
                    for alias in ["bước sắp thứ tự", "bước sắp xếp thứ tự"]:
                        self.aliases.append((tokenize(alias), key))
        self.aliases.sort(key=lambda item: (-len(item[0]), item[1]))
        for key, entity in self.entities.items():
            if entity.type in {"week", "chapter", "assignment_part"}:
                number = int(re.search(r"\d+", entity.id)[0]) if re.search(r"\d+", entity.id) else None
                if entity.type == "assignment_part":
                    number = {"PART_I": 1, "PART_II": 2, "PART_III": 3, "PART_IV": 4}.get(entity.id)
                entity.label = {"week": "Tuần", "chapter": "Chương", "assignment_part": "Phần"}[entity.type] + f" {number}"
                if entity.type == "assignment_part":
                    entity.label += " của bài tập lớn Course Assistant"
                entity.source_document_ids = [doc_id for doc_id, d in self.documents.items() if
                    d["metadata"].get({"week": "week", "chapter": "chapter", "assignment_part": "part"}[entity.type]) == number
                    or (entity.type == "chapter" and number in d["metadata"].get("chapters", []))]
            elif entity.type == "assignment":
                entity.label = "bài tập lớn Course Assistant"
                entity.source_document_ids = [d for d in self.documents if d.startswith("assignments.")]
            elif entity.type == "learning_outcome":
                # Label an LO by its KB description, so a follow-up such as
                # 'nội dung này nằm ở chương nào' searches for that content.
                for doc_id in LO_DOCUMENTS:
                    line = re.search(r"(?m)^\s*" + re.escape(entity.id) + r"\s*-\s*(.+)$", self.documents.get(doc_id, {}).get("text", ""))
                    if line:
                        entity.label = f"{entity.id} ({line[1].strip().rstrip('.')})"
                        entity.source_document_ids = [doc_id]
                        break
            elif not entity.source_document_ids:
                names = [" ".join(t) for t, k in self.aliases if k == key]
                primary = [doc_id for doc_id, d in self.documents.items() if any(contains(
                    token_text(d["title"] + " " + " ".join(d["metadata"].get("keywords", []))), alias) for alias in names)]
                entity.source_document_ids = primary or [doc_id for doc_id, d in self.documents.items()
                    if any(contains(token_text(d["text"]), alias) for alias in names)]

    def scan(self, question, grounded_only=False):
        spans = token_spans(question)
        tokens = tuple(t for t, _, _ in spans)
        occupied, mentions = set(), []
        for alias, key in self.aliases:
            if grounded_only and not self.entities[key].source_document_ids:
                continue
            for start in range(len(tokens) - len(alias) + 1):
                indices = set(range(start, start + len(alias)))
                if not occupied & indices and tokens[start:start + len(alias)] == alias:
                    begin, end = spans[start][1], spans[start + len(alias) - 1][2]
                    if begin is not None and end is not None:
                        entity = deepcopy(self.entities[key])
                        # Use the user's actual name, rather than an arbitrary alias.
                        if entity.type in {"concept", "topic", "model", "algorithm"}:
                            entity.label = question[begin:end]
                        mentions.append({"start": begin, "end": end, "entity": entity})
                        occupied |= indices
        return sorted(mentions, key=lambda m: m["start"])

    def related(self, entity):
        if entity.type != "week":
            return []
        canonical = self.entities.get(entity.key, entity)
        chapters = {ch for doc_id in canonical.source_document_ids for ch in
                    self.documents[doc_id]["metadata"].get("chapters", [])}
        return [deepcopy(e) for e in self.entities.values() if e.type == "chapter"
                and int(re.search(r"\d+", e.id)[0]) in chapters]

    def chapters_of(self, entity):
        """Chapter entity keys whose KB documents support a concept/topic entity."""
        if entity.type not in {"concept", "topic", "algorithm", "model"}:
            return set()
        canonical = self.entities.get(entity.key, entity)
        numbers = set()
        for doc_id in canonical.source_document_ids:
            metadata = self.documents.get(doc_id, {}).get("metadata", {})
            numbers.update(metadata.get("chapters", []))
            if isinstance(metadata.get("chapter"), int):
                numbers.add(metadata["chapter"])
        keys = {("chapter", f"CH{n:02d}") for n in numbers}
        return {key for key in keys if key in self.entities}

    def resource_documents(self, entities):
        ids = set()
        for entity in entities:
            targets = self.related(entity) if entity.type == "week" else [entity]
            for target in targets:
                canonical = self.entities.get(target.key, target)
                ids.update(doc_id for doc_id in canonical.source_document_ids
                           if self.documents[doc_id]["type"] in {"slide_topic", "resource"})
        return sorted(ids)

    def supporting_documents(self, entity):
        return list(self.entities.get(entity.key, entity).source_document_ids)

    def reply_entities(self, question, kind):
        mentions = self.scan(question, grounded_only=True)
        if not mentions and re.fullmatch(r"\s*\d+[.!]?\s*", question) and kind in {"chapter", "week", "assignment_part"}:
            number = int(re.search(r"\d+", question)[0])
            label = {"chapter": "Chương", "week": "Tuần", "assignment_part": "Phần"}[kind] + f" {number}"
            mentions = self.scan(label, grounded_only=True)
            return [m["entity"] for m in mentions if m["entity"].type == kind]
        rest = question
        for mention in reversed(mentions):
            rest = rest[:mention["start"]] + " " + rest[mention["end"]:]
        remaining = set(tokenize(rest)) - {"ý", "mình", "tôi", "em", "đang", "nói", "là", "hỏi", "về", "nhé", "ạ", "và", "với", "của", "lịch", "mẫu", ".", "!", ","}
        if remaining:
            return []  # A new independent question is not a slot-filling reply.
        return [m["entity"] for m in mentions]
