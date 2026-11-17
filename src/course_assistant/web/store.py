"""Small SQLite store for local browser-owned conversations and dialogue state."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import sqlite3
import uuid
from ..dialogue.state import DialogueState, MemoryEntity, PendingClarification


def now():
    return datetime.now(timezone.utc).isoformat()


def restore_state(value):
    state = DialogueState(value["session_id"])
    for key in ("turn_id", "last_grounded_turn"):
        setattr(state, key, value[key])
    for key in ("focus", "related_entities", "recent_entities", "comparison_pair"):
        setattr(state, key, [MemoryEntity(**e) for e in value[key]])
    pending = value.get("pending_clarification")
    if pending:
        pending["bindings"] = {key: [MemoryEntity(**e) for e in entities]
                               for key, entities in pending["bindings"].items()}
        state.pending_clarification = PendingClarification(**pending)
    return state


class ChatStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS owners (id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, state TEXT);
                CREATE TABLE IF NOT EXISTS turns (
                    conversation_id TEXT NOT NULL, id TEXT NOT NULL, question TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, id));
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def owner(self, token):
        with self.connection() as db:
            if token and db.execute("SELECT 1 FROM owners WHERE id=?", (token,)).fetchone():
                return token
            token = secrets.token_urlsafe(32)
            db.execute("INSERT INTO owners VALUES (?)", (token,))
            return token

    def list(self, owner):
        with self.connection() as db:
            return [dict(r) for r in db.execute(
                "SELECT id,title,created_at,updated_at FROM conversations WHERE owner=? ORDER BY updated_at DESC", (owner,))]

    def create(self, owner):
        stamp, identity = now(), str(uuid.uuid4())
        with self.connection() as db:
            if db.execute("SELECT count(*) FROM conversations WHERE owner=?", (owner,)).fetchone()[0] >= 100:
                raise ValueError("Bạn đã có 100 cuộc trò chuyện. Hãy xóa bớt để tạo mới.")
            db.execute("INSERT INTO conversations VALUES (?,?,?,?,?,NULL)",
                       (identity, owner, "Cuộc trò chuyện mới", stamp, stamp))
        return {"id": identity, "title": "Cuộc trò chuyện mới", "created_at": stamp, "updated_at": stamp}

    def get(self, owner, identity):
        with self.connection() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=? AND owner=?", (identity, owner)).fetchone()
            if row is None:
                raise KeyError(identity)
            result = dict(row)
            result["turns"] = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM turns WHERE conversation_id=? ORDER BY created_at,rowid", (identity,))]
            result.pop("owner")
            result["state"] = json.loads(result["state"]) if result["state"] else None
            return result

    def save(self, owner, identity, turn, state):
        with self.connection() as db:
            count = db.execute("SELECT count(*) FROM turns WHERE conversation_id=?", (identity,)).fetchone()[0]
            db.execute("INSERT INTO turns VALUES (?,?,?,?,?)", (identity, turn["id"], turn["question"],
                       json.dumps(turn, ensure_ascii=False), turn["created_at"]))
            title = turn["question"].replace("\n", " ").strip()[:65] if count == 0 else None
            db.execute("UPDATE conversations SET state=?,updated_at=?,title=COALESCE(?,title) WHERE id=? AND owner=?",
                       (json.dumps(state, ensure_ascii=False), now(), title, identity, owner))

    def rename(self, owner, identity, title):
        self.get(owner, identity)
        with self.connection() as db:
            db.execute("UPDATE conversations SET title=? WHERE id=? AND owner=?", (title, identity, owner))

    def reset(self, owner, identity):
        self.get(owner, identity)
        with self.connection() as db:
            db.execute("DELETE FROM turns WHERE conversation_id=?", (identity,))
            db.execute("UPDATE conversations SET state=NULL,title=?,updated_at=? WHERE id=? AND owner=?",
                       ("Cuộc trò chuyện mới", now(), identity, owner))

    def delete(self, owner, identity):
        self.get(owner, identity)
        with self.connection() as db:
            db.execute("DELETE FROM turns WHERE conversation_id=?", (identity,))
            db.execute("DELETE FROM conversations WHERE id=? AND owner=?", (identity, owner))
