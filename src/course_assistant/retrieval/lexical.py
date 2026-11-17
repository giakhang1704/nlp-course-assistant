"""TF-IDF/cosine baseline with explicit aliases and soft metadata boosts."""
import re
import unicodedata
from ..rag.question_analysis import FIELD_LABELS

ALIASES = {
    "tín chỉ": "credits", "bao nhiêu chương": "chapter", "mấy chương": "chapter",
    "khoa": "department", "giảng viên": "instructor", "email": "email",
    "phòng học": "room", "hạn nộp": "deadline", "thời hạn": "deadline",
    "tính điểm": "assessment", "trọng số": "assessment", "thi giữa kỳ": "midterm", "thi cuối kỳ": "final",
    "giữa kỳ": "midterm", "cuối kỳ": "final", "mục tiêu": "course_description objective",
    "chuẩn đầu ra": "learning_outcomes", "bao nhiêu tuần": "format duration_weeks",
    "tài liệu": "resources reference textbook", "quy định": "regulations",
    "hình thái": "morphology", "biến hình": "inflection", "phái sinh": "derivation",
    "văn phạm phi ngữ cảnh": "context free grammar cfg", "cfg": "context free grammar",
    "ner": "named entity recognition", "nhận diện thực thể": "named entity recognition",
    "pos": "part of speech", "gán nhãn từ loại": "part of speech tagging",
    "mô hình ngôn ngữ": "language model", "truy xuất thông tin": "information retrieval",
    "trích xuất thông tin": "information extraction", "cú pháp": "syntax parsing",
    "ngữ nghĩa": "semantics", "khử nhập nhằng": "ambiguity resolution",
    "học máy": "machine learning", "tóm tắt": "summarization",
    "realizer": "realization", "realization": "realizer",
    "bài tập lớn": "major assignment course assistant", "btl01": "major assignment course assistant",
    "trợ động từ": "auxiliary", "hỏi đáp": "question answering qa",
}
STOP_WORDS = set("là gì có không cho tôi biết xin hãy vui lòng giúp giải thích thêm về của và với trong môn học môn này đó phần nào bao nhiêu mấy thế như sao thì được các những một cái vậy ở để the a an is what how why please tell me".split())
STOP_WORDS.update("ra vì tại bạn mình hãy cần muốn đến theo nói nữa cách khi hay hơn giữa đó lại".split())


def vectorizer_options():
    return {"ngram_range": (1, 2), "token_pattern": r"(?u)\b\w+\b",
            "sublinear_tf": True, "norm": "l2", "stop_words": sorted(STOP_WORDS)}


def normalize(text):
    return unicodedata.normalize("NFC", text).casefold().replace("_", " ").replace("-", " ")


def contains_phrase(text, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def expand(text):
    text = normalize(text)
    additions = [value for key, value in ALIASES.items()
                 if re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", text)]
    return text + " " + " ".join(additions)


def meaningful_terms(text):
    return set(re.findall(r"(?u)\b\w+\b", expand(text))) - STOP_WORDS


def meaningful_bigrams(text):
    tokens = [t for t in re.findall(r"(?u)\b\w+\b", normalize(text))
              if t not in STOP_WORDS and t not in {"and", "or"}]
    return {" ".join(pair) for pair in zip(tokens, tokens[1:])}


class LexicalRetriever:
    def __init__(self, chunks, boost=0.08):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.boost = boost
        self.vectorizer = TfidfVectorizer(**vectorizer_options())
        texts = [expand(c["title"] + "\n" + FIELD_LABELS.get(c["metadata"].get("field", "").casefold(), "")
                        + "\n" + " ".join(c["metadata"].get("keywords", []))
                        + "\n" + c["text"]) for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)
        self.prepare_subjects()

    def prepare_subjects(self):
        self.subject_terms = [set(re.findall(r"\b[a-z]{3,}\b", normalize(
            " ".join(c["metadata"].get("keywords", []))))) - STOP_WORDS for c in self.chunks]
        self.subject_frequency = {t: sum(t in ts for ts in self.subject_terms)
                                  for t in set().union(*self.subject_terms)}

    def retrieve(self, query, k=5):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("top_k must be a positive integer")
        from sklearn.metrics.pairwise import cosine_similarity
        # Keep internal operation labels out of the lexical search text.
        question = query["original_question"]
        terms = meaningful_terms(question)
        direct_terms = set(re.findall(r"(?u)\b\w+\b", normalize(question))) - STOP_WORDS
        query_phrases = meaningful_bigrams(question)
        search_text = expand(question)
        requested = {field for group in query.get("retrieval_hints", {}).get("field_groups", []) for field in group}
        scope = dict(query.get("retrieval_hints", {}).get("scope", {}))
        requested_docs = query.get("retrieval_hints", {}).get("document_ids")
        requested_docs = None if requested_docs is None else set(requested_docs)
        if scope.get("week") in {"first", "last"}:
            weeks = [c["metadata"]["week"] for c in self.chunks if "week" in c["metadata"]]
            scope["week"] = (min(weeks) if scope["week"] == "first" else max(weeks)) if weeks else -1
        search_text += " " + " ".join(normalize(field) for field in sorted(requested))
        if query.get("attribute"):
            search_text += " " + normalize(query["attribute"])
            mappings = {"duration_weeks": "format", "objective": "course_description",
                        "assessment_weight": "assessment", "duration_minutes": "exam_format"}
            search_text += " " + mappings.get(query["attribute"], "")
        for entity in query.get("entities", []):
            if entity.get("type") in {"exam", "assessment"}:
                search_text += " " + normalize(str(entity.get("value", "")))
        vector = self.vectorizer.transform([search_text])
        scores = cosine_similarity(vector, self.matrix).ravel()
        max_idf = float(self.vectorizer.idf_.max())
        weights = {t: float(self.vectorizer.idf_[self.vectorizer.vocabulary_[t]])
                   if t in self.vectorizer.vocabulary_ else max_idf for t in direct_terms}
        total_weight = sum(weights.values())
        results = []
        milestone = query.get("retrieval_hints", {}).get("milestone")
        milestone_numbers = [int(m[1]) for c in self.chunks if (m := re.search(
            r"(?mi)^milestone:.*checkpoint\s+(\d+)", c["text"]))]
        milestone_number = (min(milestone_numbers) if milestone["ordinal"] == "first" else max(milestone_numbers)) if milestone and milestone.get("ordinal") and milestone_numbers else None
        parts = {int(e["value"]) for e in query.get("entities", []) if e.get("type") == "assignment_part" and str(e.get("value", "")).isdigit()}
        for chunk, score, subject_terms in zip(self.chunks, scores, self.subject_terms):
            document_match = requested_docs is not None and chunk["document_id"] in requested_docs
            if requested_docs is not None and not document_match:
                continue
            if score <= 0 and not document_match:
                continue
            lexical_score = float(score)
            metadata = chunk["metadata"]
            checkpoint = re.search(r"(?mi)^milestone:.*checkpoint\s+(\d+)", chunk["text"])
            if milestone and (not checkpoint or milestone_number is not None and int(checkpoint[1]) != milestone_number):
                continue
            field_match = metadata.get("field", "").casefold() in requested
            scope_match = bool(scope) and all(metadata.get(key) == value or
                (key == "chapter" and value in metadata.get("chapters", [])) for key, value in scope.items())
            if query.get("intent") == "GET_ASSIGNMENT" and parts:
                scope_match = metadata.get("part") in parts
            if milestone and checkpoint:
                scope_match = True
            section_match = any(re.search(r"(?mi)^" + re.escape(s) + r":", chunk["text"])
                                for s in query.get("retrieval_hints", {}).get("sections", []))
            preferred = metadata["source_file"] in query.get("preferred_sources", [])
            entity_match = False
            for entity in query.get("entities", []):
                value = str(entity.get("value", ""))
                if entity.get("type") == "week":
                    entity_match |= metadata.get("week") == int(re.search(r"\d+", value)[0]) if re.search(r"\d+", value) else False
                elif entity.get("type") == "chapter":
                    number = re.search(r"\d+", value)
                    entity_match |= bool(number and metadata.get("chapter") == int(number[0]))
                elif entity.get("type") in {"topic", "model", "algorithm"}:
                    entity_match |= normalize(value) in normalize(chunk["title"])
            body_terms = meaningful_terms(chunk["title"] + " " + chunk["text"])
            overlap = sorted(terms & body_terms)
            anchors = sorted(terms & meaningful_terms(chunk["title"] + " " +
                                                     " ".join(metadata.get("keywords", []))))
            # English technical names are retained verbatim in this dataset.
            # Generic Vietnamese pairs such as 'dữ liệu' are not title evidence
            # for an otherwise unrelated or unsupported technical question.
            anchor_phrases = sorted(phrase for phrase in
                query_phrases & meaningful_bigrams(chunk["title"] + " " + " ".join(metadata.get("keywords", [])))
                if all(re.fullmatch(r"[a-z]{3,}", token) for token in phrase.split()))
            content_terms = body_terms | meaningful_terms(" ".join(metadata.get("keywords", [])))
            named_terms = sorted(direct_terms & subject_terms)
            name_weight = sum(1 / self.subject_frequency[t] for t in named_terms)
            body_phrases = sorted(phrase for phrase in query_phrases if
                all(re.fullmatch(r"[a-z]{3,}", t) for t in phrase.split())
                and contains_phrase(normalize(chunk["text"]), phrase))
            # Count OOV terms as maximally informative so an unrelated query is
            # not admitted just because it shares one generic word with the KB.
            coverage = sum(weights[t] for t in direct_terms & content_terms) / total_weight if total_weight else 0.0
            results.append({**chunk, "score": lexical_score * (1 + self.boost * (preferred + entity_match + bool(anchors)))
                            + 0.5 * min(name_weight, 1.0) + 0.12 * bool(body_phrases),
                            "lexical_score": lexical_score, "matched_terms": overlap,
                            "anchor_matches": anchors,
                            "anchor_phrase_matches": anchor_phrases,
                            "technical_name_matches": named_terms,
                            "technical_name_weight": name_weight,
                            "body_phrase_matches": body_phrases,
                            "query_term_coverage": coverage,
                            "field_match": field_match or section_match,
                            "document_match": document_match,
                            "scope_match": scope_match,
                            "parser_boost": bool(preferred or entity_match)})
        results.sort(key=lambda r: (not r["field_match"], not r["scope_match"], -r["score"], r["id"]))
        # A grounded comparison needs evidence for both arguments before filling
        # the remaining positions with other high-scoring chunks.
        seeded = []
        if query.get("intent") == "GET_ASSIGNMENT" and not parts and query.get("question_type") == "requirements":
            # The overview and each part together define assignment scope.
            assignment_hits = [h for h in results if h["type"] == "assignment"]
            seeded = sorted(assignment_hits, key=lambda h: (h["metadata"].get("part") or 0, -h["score"]))[:k]
            for hit in seeded:
                hit["document_match"] = True
        for group in query.get("retrieval_hints", {}).get("document_groups", []):
            hit = next((h for h in results if h["document_id"] in group), None)
            if hit is not None and hit["id"] not in {h["id"] for h in seeded}:
                seeded.append(hit)
        if seeded:
            results = seeded + [h for h in results if h["id"] not in {s["id"] for s in seeded}]
        return results[:k]
