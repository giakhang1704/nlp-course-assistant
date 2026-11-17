"""Allowlisted source loaders; evaluation labels and duplicate TXT copies stay out."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

RAW_SOURCES = ("course_info.txt", "schedule.txt", "topics.txt", "assignments.txt",
               "regulations.txt", "resources.txt", "fag.txt")


def normalize(text):
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


def source_paths(root):
    root = Path(root)
    return [root / "data/raw" / name for name in RAW_SOURCES] + [
        root / "data/slide_content" / f"chapter_{ch:02d}.jsonl" for ch in range(1, 12)
    ] + [root / "data/slide_content/provenance.jsonl"]


def source_signature(root):
    paths = source_paths(root)
    return [{"path": p.relative_to(root).as_posix(),
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def sections(text, pattern):
    matches = list(re.finditer(pattern, text, re.M))
    for i, match in enumerate(matches):
        yield match, text[match.start():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()


def load_documents(root):
    root = Path(root)
    documents = []

    def add(doc_id, title, text, source, kind, **metadata):
        text = normalize(text)
        if not text:
            raise ValueError(f"Empty document: {doc_id}")
        documents.append({"id": doc_id, "type": kind, "title": title, "text": text,
                          "metadata": {"source_file": source, "is_sample": False, **metadata}})

    for name in RAW_SOURCES:
        text = normalize((root / "data/raw" / name).read_text(encoding="utf-8-sig"))
        stem = Path(name).stem
        sample = name in {"schedule.txt", "assignments.txt", "regulations.txt"}
        if stem == "schedule":
            for match, body in sections(text, r"^WEEK\s+(\d+)\s*$"):
                week = int(match[1])
                chapters = [int(c) for c in re.findall(r"Chapter\s+(\d+)", body)]
                add(f"schedule.week_{week:02d}", f"Tuần {week} / Week {week}", body,
                    name, "schedule", week=week, chapters=chapters, is_sample=sample)
        elif stem == "topics":
            for match, body in sections(text, r"^CHAPTER\s+(\d+):\s*(.+)$"):
                add(f"topics.chapter_{int(match[1]):02d}", match[2], body, name,
                    "topic_summary", chapter=int(match[1]))
        elif stem == "fag":
            for match, body in sections(text, r"^Q(\d+):\s*$"):
                question = re.search(r"^question:\s*(.+)$", body, re.M)
                add(f"faq.q{int(match[1]):02d}", question[1] if question else match[0],
                    body, name, "faq")
        elif stem == "resources":
            for match, body in sections(text, r"^(TEXTBOOK|REFERENCE)\s+(\d+)\s*$"):
                title = re.search(r"^title:\s*(.+)$", body, re.M)
                authors = re.search(r"^authors?:\s*(.+)$", body, re.M)
                add(f"resources.{match[1].lower()}_{match[2]}", title[1] if title else match[0],
                    body, name, "resource", keywords=[authors[1]] if authors else [])
        elif stem == "assignments":
            # Preserve shared assignment information and each complete assignment part.
            matches = list(sections(text, r"^(PART [IVX]+\s+-\s+.+|LLM/RAG EXTENSION)\s*$"))
            preamble = text[:matches[0][0].start()] if matches else text
            assignment = re.search(r"(?m)^assignment_id:\s*(\S+)", preamble)
            parent = {"assignment_id": assignment[1]} if assignment else {}
            add("assignments.overview", "Course Assistant assignment", preamble, name,
                "assignment", is_sample=sample, **parent)
            for i, (match, body) in enumerate(matches, 1):
                add(f"assignments.part_{i}", match[1], body, name, "assignment",
                    is_sample=sample, part=i if i < 5 else None, **parent)
        else:
            # A key and all continuation lines form one record, including nested lists.
            for match, body in sections(text, r"^([A-Za-z][A-Za-z_]*):[^\n]*$"):
                key = match[1]
                add(f"{stem}.{key.lower()}", key, body, name,
                    "rule" if stem == "regulations" else "course_info",
                    field=key.casefold(), is_sample=sample)

    provenance_path = root / "data/slide_content/provenance.jsonl"
    provenance = {r["id"]: r for r in read_jsonl(provenance_path)}
    for ch in range(1, 12):
        name = f"chapter_{ch:02d}.jsonl"
        for item in read_jsonl(root / "data/slide_content" / name):
            evidence = provenance.get(item["id"])
            if evidence is None:
                raise ValueError(f"Missing source metadata: {item['id']}")
            add(item["id"], item["title"], item["content_vi"], evidence["source_file"],
                "slide_topic", chapter=ch, keywords=item["keywords"],
                dataset_file=f"data/slide_content/{name}",
                source_slides=evidence["source_slides"],
                legacy_text_blocks=evidence["legacy_text_blocks"],
                source_checksum=evidence["source_sha256"])
    ids = [d["id"] for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs")
    return documents


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
