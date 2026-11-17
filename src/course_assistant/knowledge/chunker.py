"""Preserve records; split long records at paragraphs/sentences with stable IDs."""
import re


def chunk_documents(documents, max_chars=1800):
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 200:
        raise ValueError("chunk_chars must be an integer >= 200")
    chunks = []
    for document in documents:
        units = re.split(r"(?<=[.!?])\s+|\n\s*\n", document["text"])
        parts, current = [], ""
        for unit in units:
            # Hard split only exceptionally long units; never discard text.
            for offset in range(0, len(unit), max_chars):
                piece = unit[offset:offset + max_chars]
                if current and len(current) + len(piece) + 1 > max_chars:
                    parts.append(current)
                    current = ""
                current = (current + "\n" + piece).strip()
        if current:
            parts.append(current)
        for index, text in enumerate(parts):
            chunks.append({"id": f"{document['id']}:{index:03d}",
                           "document_id": document["id"], "title": document["title"],
                           "text": text, "type": document["type"],
                           "metadata": dict(document["metadata"])})
    if not chunks:
        raise ValueError("No knowledge chunks")
    return chunks
