"""Validate the response contract without repairing facts or inventing citations."""
import json
import re


class GenerationValidationError(ValueError):
    pass


def diagnostic_preview(value, secrets=(), max_chars=4000):
    """Bound rejected model output and redact credentials; never log HTTP data."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"AIza[\w-]{20,}", "[REDACTED]", text)
    text = re.sub(r"(?i)Bearer\s+[\w.\-/+=]+", "Bearer [REDACTED]", text)
    return {"preview": text[:max_chars], "truncated": len(text) > max_chars}


def response_schema(context):
    labels = []
    for line in context.splitlines():
        try:
            block = json.loads(line)
        except ValueError:
            continue
        label = block.get("citation") if isinstance(block, dict) else None
        if isinstance(label, str) and re.fullmatch(r"C\d+", label) and label not in labels:
            labels.append(label)
    citation = {"type": "string"}
    if labels:
        citation["enum"] = labels
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["answered", "insufficient_information"]},
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": citation}},
            "required": ["status", "answer", "citations"]}


def bare_words(text):
    """Words of a text without citation labels or the sample-data prefix."""
    text = re.sub(r"(?i)\[\s*c\d+\s*\]|theo dữ liệu mẫu của project:?", " ", text or "").casefold()
    return re.findall(r"\w+", text)


def echoes_question(answer, question):
    return bool(question) and bare_words(answer) == bare_words(question)


def validate_generation(generation, sources, question=None, render_declared=False):
    if not isinstance(generation, dict):
        raise GenerationValidationError("llm_invalid_response_type")
    if not isinstance(generation.get("status"), str) or generation["status"] not in {"answered", "insufficient_information"}:
        raise GenerationValidationError("llm_invalid_status")
    answer, citations = generation.get("answer"), generation.get("citations")
    if generation["status"] == "insufficient_information" and isinstance(answer, str) and not answer.strip() \
            and isinstance(citations, list) and not citations:
        # A bare refusal is a valid refusal; the caller supplies the message.
        return "insufficient_information", "", []
    if not isinstance(answer, str) or not answer.strip():
        raise GenerationValidationError("llm_empty_answer")
    if generation["status"] == "answered" and echoes_question(answer, question):
        raise GenerationValidationError("llm_echoed_question")
    if not isinstance(citations, list) or any(not isinstance(c, str) for c in citations):
        raise GenerationValidationError("llm_invalid_citations_type")
    canonical = []
    for label in citations:
        match = re.fullmatch(r"\s*(?:\[\s*(C\d+)\s*\]|(C\d+))\s*", label)
        if not match:
            raise GenerationValidationError("llm_invalid_citation_label")
        label = match[1] or match[2]
        if label not in sources:
            raise GenerationValidationError("llm_unknown_citation")
        if label not in canonical:
            canonical.append(label)
    # Normalize equivalent grouped references, retaining every supplied label.
    answer = re.sub(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\]",
                    lambda m: "".join(f"[{c}]" for c in re.findall(r"C\d+", m[1])), answer)
    used = set(re.findall(r"\[(C\d+)\]", answer))
    if render_declared and generation["status"] == "answered" and not used and canonical:
        # Only after a regeneration: the model again declared valid citations
        # but wrote none inline, so render the declared labels. A mismatch
        # between inline and declared labels is still rejected below; no label
        # is ever added that the model did not declare.
        answer = answer.rstrip() + " " + "".join(f"[{c}]" for c in canonical)
        used = set(canonical)
    if used - sources.keys():
        raise GenerationValidationError("llm_unknown_citation")
    if used != set(canonical):
        raise GenerationValidationError("llm_citation_mismatch")
    if generation["status"] == "answered" and not used:
        raise GenerationValidationError("llm_answer_missing_citations")
    return generation["status"], answer.strip(), canonical
