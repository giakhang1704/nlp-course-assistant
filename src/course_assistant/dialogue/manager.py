"""One state per session, pending slot filling, and grounded state transitions."""
from copy import deepcopy
from dataclasses import asdict
import re
from .catalog import EntityCatalog, token_text
from .query_builder import build_dialogue_query
from .resolver import ReferenceResolver, compatible
from .state import DialogueState, PendingClarification


def unique(entities):
    result = []
    for entity in entities:
        if entity.key not in {e.key for e in result}:
            result.append(entity)
    return result


class DialogueAssistant:
    def __init__(self, assistant, max_context_turns=3, pending_turns=2):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (max_context_turns, pending_turns)):
            raise ValueError("Dialogue turn windows must be positive integers")
        self.assistant = assistant
        self.catalog = EntityCatalog(assistant)
        self.resolver = ReferenceResolver(self.catalog, max_context_turns)
        self.pending_turns = pending_turns
        self.sessions = {}

    def state(self, session_id):
        if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
            raise ValueError("session_id must be a non-empty string of at most 128 characters")
        return self.sessions.setdefault(session_id, DialogueState(session_id))

    def ask(self, question, session_id="default", retrieve_only=False):
        state = self.state(session_id)
        state.turn_id += 1
        before = state.snapshot()
        parsed = self.assistant.parser.parse(question)
        context_used, from_pending = [], False
        requested_question = question
        result = self.assistant.new_result(question, parsed)
        if token_text(question).strip() in {"reset", "/ reset", "/reset"}:
            state.clear()
            result.update(status="session_reset", answer="Đã bắt đầu lại ngữ cảnh hội thoại.")
            return self._record(result, state, before, question, None, [], False)
        if parsed["query"] is None or parsed["ambiguity"]:
            result = self.assistant.ask(question, retrieve_only=retrieve_only)
            state.pending_clarification = None
            return self._record(result, state, before, question, None, [], False)
        overrides, final_overrides = {}, set()
        pending = state.pending_clarification
        if pending and state.turn_id - pending.created_turn <= self.pending_turns:
            replies = self.catalog.reply_entities(question, pending.kind)
            replies = unique(e for e in replies if compatible(e, pending.kind))
            if len(replies) == (2 if pending.kind == "pair" else 1):
                requested_question = pending.question
                parsed_request = self.assistant.parser.parse(requested_question)
                overrides = deepcopy(pending.bindings)
                final_overrides = set(overrides)
                for reply in replies:
                    reply.from_turn = state.turn_id
                overrides[pending.reference_key] = replies
                from_pending = True
            else:
                parsed_request = parsed
        else:
            parsed_request = parsed
        state.pending_clarification = None
        resolution = self.resolver.resolve(requested_question, parsed_request, state, overrides, final_overrides)
        context_used = [entity for ref in resolution.references for entity in
                       (ref.get("antecedents") or [e.to_dict() for e in resolution.bindings.get(ref["key"], [])])]
        if resolution.status != "resolved":
            result.update(status=resolution.status, answer=resolution.reason)
            if resolution.status == "clarification_needed":
                state.pending_clarification = PendingClarification(
                    requested_question, resolution.unresolved_key or "ellipsis", resolution.required_kind,
                    resolution.candidates, deepcopy(resolution.bindings), state.turn_id)
            else:
                state.focus, state.related_entities, state.comparison_pair = [], [], []
            return self._record(result, state, before, resolution.effective_question,
                                resolution, context_used, from_pending, requested_question)
        effective = resolution.effective_question
        effective_parse = self.assistant.parser.parse(effective)
        if effective_parse["query"] is None or effective_parse["ambiguity"] or effective_parse["needs_context"]:
            result.update(status="clarification_needed", answer="Chưa diễn giải được câu hỏi; vui lòng nêu rõ tên đối tượng và yêu cầu.")
            return self._record(result, state, before, effective, resolution, context_used, from_pending, requested_question)
        mentions = self.catalog.scan(requested_question)
        located = [(m["start"], m["entity"]) for m in mentions]
        for ref in resolution.references:
            located += [(ref["start"], e) for e in resolution.bindings.get(ref["key"], [])]
        ordered = [e for _, e in sorted(located, key=lambda item: item[0])]
        targets = unique(ordered)
        query = build_dialogue_query(effective, effective_parse, targets, self.catalog, bool(resolution.references) or from_pending)
        generation_question = query.get("generation_question", effective)
        result = self.assistant.answer_query(question, parsed, query, retrieve_only=retrieve_only, generation_question=generation_question)
        result["generation_question"] = generation_question
        result["effective_parse_result"] = effective_parse
        state.pending_clarification = None
        if result["status"] in {"answered", "retrieved"}:
            sources = result["source_map"].values() if retrieve_only else result["citations"]
            used = {s["document_id"] for s in sources}
            confirmed = []
            for entity in targets:
                # A remembered entity keeps only the documents cited on its
                # turn; accept any KB document that supports it on later turns.
                evidence = (set(entity.source_document_ids) | set(self.catalog.supporting_documents(entity))) & used
                if not evidence and entity.type == "week":
                    evidence = {d for related in self.catalog.related(entity) for d in related.source_document_ids} & used
                if evidence:
                    entity = deepcopy(entity)
                    entity.source_document_ids = sorted(evidence)
                    entity.from_turn = state.turn_id
                    confirmed.append(entity)
            # Retrieval can understand a question whose name is absent from the
            # lexicon. Remember the subjects of actually cited slide documents,
            # not arbitrary words or named entities in the generated answer.
            # Keep multiple subjects as candidates: the resolver will clarify
            # a singular reference instead of guessing the first result.
            if not targets:
                for doc_id in sorted(used):
                    topic = self.catalog.entities.get(("topic", doc_id))
                    if topic is not None:
                        topic = deepcopy(topic)
                        topic.source_document_ids = [doc_id]
                        topic.from_turn = state.turn_id
                        confirmed.append(topic)
                result["focus_origin"] = "cited_documents" if confirmed else None
            # A request for a book can mention its subject as well. Preserve the
            # cited resource separately so 'cuốn đó' is typed correctly.
            for doc_id in sorted(used):
                doc = self.catalog.documents.get(doc_id, {})
                week = doc.get("metadata", {}).get("week") if doc.get("type") == "schedule" and re.search(r"tuần (?:nào|mấy)|checkpoint", token_text(question)) else None
                entity = self.catalog.entities.get(("week", f"WEEK_{week:02d}")) if week is not None else self.catalog.entities.get(("resource", doc_id))
                if entity:
                    entity = deepcopy(entity)
                    entity.source_document_ids = [doc_id]
                    entity.from_turn = state.turn_id
                    confirmed.append(entity)
            confirmed = unique(confirmed)
            if re.search(r"\bbước\b", token_text(question)) and result.get("answer"):
                steps = unique(m["entity"] for m in self.catalog.scan(result["answer"]) if
                               m["entity"].type == "step" and set(m["entity"].source_document_ids) & used)
                if len(steps) == 1:
                    steps[0].from_turn = state.turn_id
                    confirmed = steps
            # A narrow named concept (have, term) takes focus over a broader
            # category mentioned alongside it (auxiliary, Logical Form).
            confirmed = [e for e in confirmed if not any(
                e.key != other.key and other.type == e.type and
                set(self.catalog.supporting_documents(other)) < set(self.catalog.supporting_documents(e))
                for other in confirmed)]
            bound_keys = {e.key for entities in resolution.bindings.values() for e in entities}
            comparison_request = bool(re.search(r"so sánh|so hai|khác nhau|khác .* thế nào", token_text(effective)))
            if bound_keys and not comparison_request:
                bound = [e for e in confirmed if e.key in bound_keys]
                if bound:
                    # Asking about a property of NER does not switch the subject
                    # to every vocabulary word ('entity') appearing in the query.
                    confirmed = bound + [e for e in confirmed if e.type == "resource" and e.key not in bound_keys]
            if re.search(r"tuần (?:nào|mấy)", token_text(question)):
                weeks = [e for e in confirmed if e.type == "week"]
                if weeks:
                    confirmed = weeks
            state.focus = confirmed
            state.related_entities = unique(e for target in confirmed for e in self.catalog.related(target))
            for entity in state.related_entities:
                entity.from_turn = state.turn_id
            comparison = (effective_parse.get("semantic_frame") or {}).get("question_type") == "comparison" or bool(
                re.search(r"so sánh|khác nhau|khác .* thế nào", token_text(effective)))
            previous = [e for e in state.recent_entities if state.turn_id - e.from_turn <= self.resolver.max_context_turns]
            named = unique(e for e in list(reversed(previous)) + confirmed if e.type in {"concept", "algorithm", "model"})
            state.comparison_pair = deepcopy(confirmed) if comparison and len(confirmed) == 2 else deepcopy(named[-2:]) if len(named) >= 2 else []
            state.last_grounded_turn = state.turn_id
            state.recent_entities = unique(confirmed + state.recent_entities)
            state.recent_entities = [e for e in state.recent_entities
                                     if state.turn_id - e.from_turn <= self.resolver.max_context_turns][:12]
        else:
            # A refusal/new unsupported subject must not silently reuse old focus.
            state.focus, state.related_entities, state.comparison_pair = [], [], []
        return self._record(result, state, before, effective, resolution, context_used, from_pending, requested_question)

    def _record(self, result, state, before, effective, resolution, context_used, from_pending, requested_question=None):
        result.update(session_id=state.session_id, turn_id=state.turn_id,
                      original_question=result["question"], requested_question=requested_question or result["question"],
                      effective_question=effective, context_used=context_used,
                      resolved_from_pending=from_pending, state_before=before, state_after=state.snapshot(),
                      resolution=None if resolution is None else {
                          "status": resolution.status, "references": resolution.references,
                          "bindings": {key: [e.to_dict() for e in entities] for key, entities in resolution.bindings.items()},
                          "reason": resolution.reason})
        if requested_question and requested_question != result["question"]:
            result["requested_parse_result"] = self.assistant.parser.parse(requested_question)
        return result
