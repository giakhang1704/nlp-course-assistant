"""Deterministic, lazy enumeration from CFG derivations, with fair sampling."""
from functools import lru_cache
from .tokenizer import detokenize

MAX_SAMPLES = 10_000
# When a derivation repeats an entity, try neighbours that change one slot by up to this many steps.
MAX_NEARBY = 10
# Sentence-type branches under each QUERY_* are named <intent>_<TYPE>.
SENTENCE_TYPES = ("WH_QUESTION", "YES_NO_QUESTION", "REQUEST", "FOLLOW_UP")


def sentence_type_of(label):
    """'SCHEDULE_REQUEST' -> 'REQUEST'; None if the label is not a sentence-type branch."""
    return next((t for t in SENTENCE_TYPES if label.endswith("_" + t)), None)

class SentenceGenerator:
    def __init__(self, grammar):
        self.grammar = grammar

    @lru_cache(maxsize=None)
    def count(self, symbol):
        """Count derivations, not unique strings (ambiguous CFGs can differ)."""
        result = 0
        for rule in self.grammar.by_lhs[symbol]:
            count = 1
            for child in rule.rhs:
                if not child.terminal:
                    count *= self.count(child.name)
            result += count
        return result

    def unrank(self, symbol, rank):
        """Decode an index without materializing the Cartesian product."""
        for rule in self.grammar.by_lhs[symbol]:
            radices = [1 if s.terminal else self.count(s.name) for s in rule.rhs]
            block = 1
            for radix in radices:
                block *= radix
            if rank >= block:
                rank -= block
                continue
            indices = [0] * len(radices)
            for i in reversed(range(len(radices))):
                indices[i] = rank % radices[i]
                rank //= radices[i]
            if self._trace is not None:
                self._trace.add(symbol)
            start = len(self._mentions) if self._mentions is not None else 0
            result = []
            for child, index in zip(rule.rhs, indices):
                result.extend((child.name,) if child.terminal else self.unrank(child.name, index))
            if self._mentions is not None:
                self._record_mentions(symbol, rule, tuple(result), start)
            return tuple(result)
        raise IndexError("Derivation index outside grammar")

    _trace = None
    _mentions = None

    def _record_mentions(self, symbol, rule, tokens, start):
        """Remember which entity each lexical phrase denotes.

        A gloss such as "suy luận diễn dịch (Deductive)" names one concept twice;
        it counts as one mention, and a gloss naming a different concept
        ("cfg (pcfg)") is marked as a contradiction.
        """
        mapping = self.grammar.lexicon.get(symbol)
        if mapping is not None:
            self._mentions.append(mapping[tokens].get("id"))
        elif symbol == "NP_CONCEPT" and [s.name for s in rule.rhs if s.terminal] == ["(", ")"] \
                and len(self._mentions) - start == 2:
            head, gloss = self._mentions[start], self._mentions[start + 1]
            del self._mentions[start + 1]
            if head != gloss:
                self._mentions.append(None)
                self._contradiction = True

    _contradiction = False

    def _repeats_entity(self):
        ids = [m for m in self._mentions if m is not None]
        return self._contradiction or len(ids) != len(set(ids))

    def _fair_derivations(self, symbol):
        """Rotate over every construction.

        UTTERANCE -> QUERY_<intent> -> <intent>_<TYPE> are single-symbol
        choices; the constructions are the productions below them. One
        derivation is taken from each construction per round, so every
        question and request form of every intent appears in the samples,
        even when an earlier construction has many lexical combinations.
        Yields (tokens, symbols used in the derivation).
        """
        def constructions(name):
            for rule in self.grammar.by_lhs[name]:
                child = rule.rhs[0] if len(rule.rhs) == 1 else None
                if child is not None and not child.terminal and (
                        child.name.startswith("QUERY_") or sentence_type_of(child.name)):
                    yield from constructions(child.name)
                else:
                    yield rule

        streams = []
        for top in self.grammar.by_lhs[symbol]:
            for rule in constructions(top.rhs[0].name) if len(top.rhs) == 1 else [top]:
                radices = [1 if s.terminal else self.count(s.name) for s in rule.rhs]
                total = 1
                for radix in radices:
                    total *= radix
                streams.append([rule, radices, total, 0])
        while streams:
            for stream in list(streams):
                rule, radices, total, rank = stream
                stream[3] += 1
                tokens, trace, repeated = self._decode(rule, radices, rank)
                if repeated:
                    # Use a neighbouring derivation instead (same construction,
                    # one slot changed), so the construction keeps its turn.
                    tokens, trace = self._nearby_alternative(rule, radices, total, rank) or (None, None)
                if tokens is not None:
                    yield tokens, trace
                if stream[3] >= total:
                    streams.remove(stream)

    def _nearby_alternative(self, rule, radices, total, rank):
        """First derivation without a repeated entity that differs from `rank`
        in one slot by at most MAX_NEARBY steps; None if there is none."""
        weight = 1
        weights = []
        for radix in reversed(radices):
            weights.append(weight)
            weight *= radix
        weights.reverse()
        for slot, (radix, w) in enumerate(zip(radices, weights)):
            digit = (rank // w) % radix
            for step in range(1, MAX_NEARBY + 1):
                if digit + step >= radix:
                    break
                tokens, trace, repeated = self._decode(rule, radices, rank + step * w)
                if not repeated:
                    return tokens, trace
        return None

    def _decode(self, rule, radices, rank):
        value = rank
        indices = [0] * len(radices)
        for i in reversed(range(len(radices))):
            indices[i] = value % radices[i]
            value //= radices[i]
        self._trace, self._mentions, self._contradiction = {rule.lhs}, [], False
        result = []
        for child, index in zip(rule.rhs, indices):
            result.extend((child.name,) if child.terminal else self.unrank(child.name, index))
        repeated = self._repeats_entity()
        trace, self._trace, self._mentions = self._trace, None, None
        return tuple(result), trace, repeated

    def generate(self, limit=MAX_SAMPLES):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SAMPLES:
            raise ValueError("limit must be an integer in [1, 10000]")
        seen = set()
        # A CFG cannot require two slots to hold different entities, so sentences
        # such as "So sánh cfg và cfg" are grammatical but useless as samples;
        # _fair_derivations skips them (every yielded sample is still in the grammar).
        for tokens, used in self._fair_derivations("UTTERANCE"):
            # S -> OPENER UTTERANCE CLOSER with OPENER -> epsilon. CLOSER is "."
            # for a request (an <intent>_REQUEST branch) and "?" otherwise;
            # the parser API checks both CLOSER productions exist.
            is_request = any(sentence_type_of(label) == "REQUEST" for label in used)
            sentence = detokenize(tokens + ("." if is_request else "?",))
            if sentence in seen:
                continue
            seen.add(sentence)
            yield sentence
            if len(seen) >= limit:
                break
