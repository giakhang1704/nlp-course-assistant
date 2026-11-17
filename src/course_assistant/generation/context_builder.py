"""Bound context size and keep a closed, explicit citation map."""
import json


def build_context(hits, max_chars=12000):
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 200:
        raise ValueError("context_chars must be an integer >= 200")
    blocks, sources = [], {}
    for hit in hits:
        label = f"C{len(blocks) + 1}"
        block = {"citation": label, "title": hit["title"], "text": hit["text"],
                 "is_sample": hit["metadata"].get("is_sample", False),
                 "source_file": hit["metadata"]["source_file"],
                 "source_slides": hit["metadata"].get("source_slides", [])}
        for key in ("chapter", "chapters", "week", "part", "assignment_id"):
            if key in hit["metadata"]:
                block[key] = hit["metadata"][key]
        encoded = json.dumps(block, ensure_ascii=False)
        if sum(len(b) for b in blocks) + len(encoded) + len(blocks) > max_chars:
            continue
        blocks.append(encoded)
        sources[label] = {"chunk_id": hit["id"], "document_id": hit["document_id"], **hit["metadata"]}
    return "\n".join(blocks), sources
