"""Derive a semantic frame from a licensed QUERY constituent.

Everything in the frame comes from the accepted parse tree and the intent
schema: the QUERY label gives intent/question_type/operation, ATTR_* nodes
give the requested attribute, and typed lexical nodes give the entities.
No KB answers, evaluation labels, LLM calls or substring heuristics are used.
"""

_NORMALIZED_INTENTS = {"DEFINITION_QUERY": "GET_TOPIC", "PROCESS_QUERY": "GET_TOPIC",
                       "COMPARISON_QUERY": "GET_TOPIC"}
_QUESTION_TYPES = {"DEFINITION_QUERY": "definition", "PROCESS_QUERY": "process",
                   "COMPARISON_QUERY": "comparison", "CONTEXT_DEPENDENT": "context",
                   "GET_RESOURCE": "list", "GET_ASSIGNMENT": "requirements"}
_OPERATIONS_BY_TYPE = {"definition": "define_topic", "process": "explain_process",
                       "comparison": "compare_topics", "context": "resolve_reference"}
_OPERATIONS_BY_INTENT = {"GET_COURSE_INFO": "get_course_attribute", "GET_SCHEDULE": "get_schedule",
                         "GET_TOPIC": "get_topic", "GET_ASSIGNMENT": "get_assignment",
                         "GET_DEADLINE": "get_deadline", "GET_RESOURCE": "get_resource",
                         "GET_RULE": "get_rule", "GET_INSTRUCTOR": "get_instructor",
                         "GET_LO": "get_learning_outcome"}


def sentence_type(query_node):
    """Read the sentence type from the branch directly under QUERY_*.

    QUERY_GET_SCHEDULE -> SCHEDULE_WH_QUESTION gives "WH_QUESTION"; the other
    values are YES_NO_QUESTION, REQUEST and FOLLOW_UP (see 00_sentence.cfg).
    """
    from ..parsing.generator import sentence_type_of
    for child in query_node.children:
        found = sentence_type_of(getattr(child, "label", ""))
        if found:
            return found
    return None


def _attribute_from_tree(query_node, attribute_symbols):
    """Return the attribute marked by ATTR_* nodes, or None if absent/unclear."""
    found = {attribute_symbols[node.label] for node in query_node.walk()
             if node.label in attribute_symbols}
    return found.pop() if len(found) == 1 else None


def logical_form(frame):
    """Render the frame as a predicate, e.g. GET_DEADLINE(attribute=deadline, assignment=BTL01).

    The predicate is the intent; arguments are the requested attribute, the
    question type when it is not a plain fact, then typed entities in surface
    order. The implicit default course is omitted when other entities exist.
    """
    args = []
    if frame.get("attribute"):
        args.append(f"attribute={frame['attribute']}")
    if frame.get("question_type") not in (None, "factual"):
        args.append(f"type={frame['question_type']}")
    if frame["intent"] == "CONTEXT_DEPENDENT" and frame.get("domain_intent") != "CONTEXT_DEPENDENT":
        args.append(f"frame={frame['domain_intent']}")
    entities = frame.get("entities", [])
    explicit = [e for e in entities if not e.get("implicit")]
    for entity in explicit or entities:
        role = next((m.get("role") for m in frame.get("mentions", [])
                     if m["id"] == entity["id"] and m.get("role")), None)
        args.append(f"{role or entity['type']}={entity['id']}")
    return f"{frame['intent']}({', '.join(args)})"


def interpret(query_node, parse_result, spec=None, schema=None):
    """Return JSON data for one parsed question."""
    spec = spec or {}
    schema = schema or {}
    legacy_intent = parse_result["intent_type"]
    intent = _NORMALIZED_INTENTS.get(legacy_intent, legacy_intent)
    mentions = [dict(m) for m in parse_result.get("mentions", [])]
    entities = [dict(e) for e in parse_result.get("entities", [])]
    question_type = spec.get("question_type") or _QUESTION_TYPES.get(legacy_intent, "factual")
    attribute = spec.get("attribute") or _attribute_from_tree(
        query_node, schema.get("attribute_symbols", {}))
    operation = (spec.get("operation") or _OPERATIONS_BY_TYPE.get(question_type)
                 or _OPERATIONS_BY_INTENT.get(intent, "unknown"))

    if question_type == "comparison":
        # Operand roles follow surface order: in "so sánh mô hình đó với CRF",
        # the unresolved reference is the left operand and CRF the right one.
        compared = [m for m in mentions if m["type"] in {"concept", "reference"}]
        for index, mention in enumerate(compared):
            mention["role"] = "left" if index == 0 else "right" if index == 1 else "additional"

    references = [dict(m) for m in mentions if m["type"] == "reference"]
    exempt = set(schema.get("context_exempt_references", []))
    unresolved = [r for r in references if r["id"] not in exempt]
    domain_intent = intent
    if unresolved and intent != "CONTEXT_DEPENDENT":
        # The construction says what is asked, but not about which object;
        # the KB cannot be queried before the dialogue layer resolves it.
        intent = "CONTEXT_DEPENDENT"

    frame = {"intent": intent, "domain_intent": domain_intent,
             "sentence_type": sentence_type(query_node),
             "question_type": question_type, "attribute": attribute,
             "operation": operation, "entities": entities, "mentions": mentions,
             "references": references,
             "needs_context": bool(parse_result.get("needs_context") or references),
             "query_label": query_node.label}
    if question_type == "comparison":
        frame["arguments"] = [{"role": m["role"], "type": m["type"], "id": m["id"]}
                              for m in mentions if "role" in m]
    frame["logical_form"] = logical_form(frame)
    return frame
