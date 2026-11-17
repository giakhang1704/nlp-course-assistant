"""Serializable memory; stable document IDs rather than per-answer C1 labels."""
from dataclasses import asdict, dataclass, field


@dataclass
class MemoryEntity:
    type: str
    id: str
    label: str
    source_document_ids: list[str] = field(default_factory=list)
    from_turn: int = 0

    @property
    def key(self):
        return self.type, self.id

    def to_dict(self):
        return asdict(self)


@dataclass
class PendingClarification:
    question: str
    reference_key: str
    kind: str
    candidates: list[dict]
    bindings: dict
    created_turn: int


@dataclass
class DialogueState:
    session_id: str
    turn_id: int = 0
    last_grounded_turn: int = 0
    focus: list[MemoryEntity] = field(default_factory=list)
    related_entities: list[MemoryEntity] = field(default_factory=list)
    recent_entities: list[MemoryEntity] = field(default_factory=list)
    comparison_pair: list[MemoryEntity] = field(default_factory=list)
    pending_clarification: PendingClarification | None = None

    def clear(self):
        self.focus.clear()
        self.related_entities.clear()
        self.recent_entities.clear()
        self.comparison_pair.clear()
        self.pending_clarification = None
        self.last_grounded_turn = 0

    def snapshot(self):
        return asdict(self)
