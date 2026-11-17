"""Build and verify an on-disk, non-pickle TF-IDF index."""
import hashlib
import json
from pathlib import Path
from ..exporters import write_json, write_jsonl
from ..knowledge.loaders import load_documents, source_signature, read_jsonl
from ..knowledge.chunker import chunk_documents
from .lexical import LexicalRetriever, vectorizer_options, ALIASES
from ..rag.question_analysis import FIELD_LABELS


def feature_signature():
    return hashlib.sha256(json.dumps({"aliases": ALIASES, "vectorizer": vectorizer_options(),
                                     "field_labels": FIELD_LABELS, "normalization_version": 2},
                                    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_index(root, config):
    import sklearn
    from scipy.sparse import save_npz
    root = Path(root)
    documents = load_documents(root)
    chunks = chunk_documents(documents, config["chunk_chars"])
    retriever = LexicalRetriever(chunks, config["preferred_source_boost"])
    write_jsonl(root / config["documents"], documents)
    write_jsonl(root / config["chunks"], chunks)
    path = root / config["index"]
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = path.with_suffix(".npz")
    save_npz(matrix_path, retriever.matrix)
    state = {"version": 1, "sklearn_version": sklearn.__version__,
             "features_sha256": feature_signature(),
             "vocabulary": retriever.vectorizer.vocabulary_,
             "idf": retriever.vectorizer.idf_.tolist(), "matrix_sha256": digest(matrix_path),
             "chunks_sha256": digest(root / config["chunks"]),
             "chunk_ids": [c["id"] for c in chunks]}
    # numpy integer values in vocabulary need conversion for JSON.
    state["vocabulary"] = {key: int(value) for key, value in state["vocabulary"].items()}
    write_json(path, state)
    manifest = {"version": 1, "document_count": len(documents), "chunk_count": len(chunks),
                "document_schema_version": 2,
                "sources": source_signature(root), "chunk_chars": config["chunk_chars"],
                "documents_sha256": digest(root / config["documents"]),
                "chunks_sha256": state["chunks_sha256"], "index_sha256": digest(path),
                "retrieval": "word unigram/bigram TF-IDF, sublinear TF, L2 cosine",
                "conflicts": []}
    # Record the actual discrepancy rather than deriving a course count from files.
    faq = next((d for d in documents if d["id"] == "faq.q02"), None)
    weeks = [d for d in documents if d["type"] == "schedule" and
             (12 in d["metadata"].get("chapters", []) or "12 chương" in d["text"])]
    schedule_text = (root / "data/raw/schedule.txt").read_text(encoding="utf-8-sig")
    if faq and "11 chương" in faq["text"] and "12 chương" in schedule_text:
        manifest["conflicts"].append({"attribute": "chapter_count", "documents": [faq["id"]] +
                                      [d["id"] for d in weeks],
                                      "sources": ["fag.txt", "schedule.txt"],
                                      "description": "FAQ states 11 chapters; sample schedule header states 12 chapters."})
    write_json(root / config["manifest"], manifest)
    return manifest


def load_index(root, config):
    import numpy as np
    import sklearn
    from scipy.sparse import load_npz
    from sklearn.feature_extraction.text import TfidfVectorizer
    root = Path(root)
    try:
        manifest = json.loads((root / config["manifest"]).read_text(encoding="utf-8"))
        path = root / config["index"]
        state = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("document_schema_version") != 2
                or manifest["sources"] != source_signature(root)
                or manifest["chunk_chars"] != config["chunk_chars"]
                or manifest["index_sha256"] != digest(path)
                or manifest["documents_sha256"] != digest(root / config["documents"])
                or state["chunks_sha256"] != digest(root / config["chunks"])
                or state["matrix_sha256"] != digest(path.with_suffix(".npz"))
                or state.get("features_sha256") != feature_signature()
                or state["sklearn_version"] != sklearn.__version__):
            raise ValueError("Knowledge/index changed; run: python -B run.py build-kb")
        chunks = read_jsonl(root / config["chunks"])
        retriever = LexicalRetriever.__new__(LexicalRetriever)
        retriever.chunks = chunks
        retriever.boost = config["preferred_source_boost"]
        retriever.vectorizer = TfidfVectorizer(**vectorizer_options(), vocabulary=state["vocabulary"])
        retriever.vectorizer.idf_ = np.asarray(state["idf"], dtype=float)
        retriever.matrix = load_npz(path.with_suffix(".npz"))
        retriever.prepare_subjects()
        if [c["id"] for c in chunks] != state["chunk_ids"] or retriever.matrix.shape != (len(chunks), len(state["vocabulary"])):
            raise ValueError("Index shape/IDs disagree; rebuild KB")
        return retriever, manifest
    except FileNotFoundError as exc:
        raise ValueError("KB/index missing; run: python -B run.py build-kb") from exc
