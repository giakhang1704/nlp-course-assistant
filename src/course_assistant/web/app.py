"""FastAPI adapter. One process serializes the shared parser, memory and limiter."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import threading
from uuid import UUID
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from ..config import PROJECT_ROOT
from ..dialogue.manager import DialogueAssistant
from ..rag.qa import CourseAssistant
from ..generation.llm_client import load_environment
from .presentation import WEB_PROMPT, present_turn
from .store import ChatStore, now, restore_state


class MessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    request_id: UUID


class RenameInput(BaseModel):
    title: str = Field(min_length=1, max_length=100)


def create_app(assistant=None, db_path=None):
    load_environment(PROJECT_ROOT)
    runtime = {"ready": False, "error": None, "manager": None}
    gate = threading.RLock()
    store = ChatStore(db_path or os.environ.get("CHAT_DB_PATH", str(PROJECT_ROOT / "output/web/chat.sqlite3")))

    def initialize():
        try:
            qa = assistant or CourseAssistant()
            qa.prompt = WEB_PROMPT
            runtime["manager"] = DialogueAssistant(qa)
            runtime["ready"] = True
        except (OSError, ValueError):
            runtime["error"] = "Không tải được dữ liệu môn học. Chạy python -B run.py build-kb rồi khởi động lại."

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(initialize)
        yield

    app = FastAPI(title="Course Assistant", lifespan=lifespan)
    app.state.runtime, app.state.store = runtime, store
    origins = os.environ.get("WEB_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                       allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def check_origin(request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "PATCH", "DELETE"} and origin and origin not in origins:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Origin không hợp lệ."}, status_code=403)
        return await call_next(request)

    def owner(request, response):
        token = store.owner(request.cookies.get("ca_browser"))
        response.set_cookie("ca_browser", token, httponly=True, samesite="strict", max_age=60 * 60 * 24 * 30,
                            path="/api", secure=os.environ.get("COOKIE_SECURE") == "1")
        return token

    def get_conversation(who, identity):
        try:
            return store.get(who, identity)
        except KeyError:
            raise HTTPException(404, "Không tìm thấy cuộc trò chuyện.")

    @app.get("/api/health")
    def health():
        return {"ready": runtime["ready"], "error": runtime["error"], "course": "CO3085"}

    @app.get("/api/course")
    def course():
        if not runtime["ready"]:
            raise HTTPException(503, runtime["error"] or "Đang tải dữ liệu.")
        docs = runtime["manager"].catalog.documents.values()
        return {"id": "CO3085", "name": "Xử lý Ngôn ngữ Tự nhiên", "chapters": sorted([
            {"number": d["metadata"]["chapter"], "title": d["title"]}
            for d in docs if d["type"] == "topic_summary"], key=lambda d: d["number"])}

    @app.get("/api/conversations")
    def list_conversations(request: Request, response: Response):
        return store.list(owner(request, response))

    @app.post("/api/conversations", status_code=201)
    def new_conversation(request: Request, response: Response):
        try:
            return store.create(owner(request, response))
        except ValueError as exc:
            raise HTTPException(429, str(exc))

    @app.get("/api/conversations/{identity}")
    def conversation(identity: str, request: Request, response: Response):
        result = get_conversation(owner(request, response), identity)
        result.pop("state")
        return result

    @app.post("/api/conversations/{identity}/messages")
    def message(identity: str, body: MessageInput, request: Request, response: Response):
        who = owner(request, response)
        text = body.content.strip()
        if not text:
            raise HTTPException(422, "Hãy nhập câu hỏi.")
        if not runtime["ready"]:
            raise HTTPException(503, runtime["error"] or "Đang tải dữ liệu.")
        with gate:
            conv = get_conversation(who, identity)
            request_id = str(body.request_id)
            existing = next((t for t in conv["turns"] if t["id"] == request_id), None)
            if existing:
                if existing["question"] != text:
                    raise HTTPException(409, "Mã lượt gửi đã được sử dụng.")
                return existing
            manager = runtime["manager"]
            # Reload bounded state from SQLite; idle sessions do not accumulate in RAM.
            if conv["state"]:
                manager.sessions[identity] = restore_state(conv["state"])
            try:
                result = manager.ask(text, identity)
                turn = present_turn(result, request_id, now())
                store.save(who, identity, turn, result["state_after"])
            finally:
                manager.sessions.pop(identity, None)
            return turn

    @app.patch("/api/conversations/{identity}")
    def rename(identity: str, body: RenameInput, request: Request, response: Response):
        who = owner(request, response)
        get_conversation(who, identity)
        title = body.title.strip()
        if not title:
            raise HTTPException(422, "Tên cuộc trò chuyện không được để trống.")
        with gate:
            store.rename(who, identity, title)
        return {"ok": True}

    @app.post("/api/conversations/{identity}/reset")
    def reset(identity: str, request: Request, response: Response):
        who = owner(request, response)
        with gate:
            get_conversation(who, identity)
            store.reset(who, identity)
            if runtime["manager"]:
                runtime["manager"].sessions.pop(identity, None)
        return {"ok": True}

    @app.delete("/api/conversations/{identity}")
    def delete(identity: str, request: Request, response: Response):
        who = owner(request, response)
        with gate:
            get_conversation(who, identity)
            store.delete(who, identity)
        return {"ok": True}

    dist = PROJECT_ROOT / "web/frontend/dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        return FileResponse(PROJECT_ROOT / "web/frontend/public/favicon.svg", media_type="image/svg+xml")

    @app.get("/")
    def index():
        if (dist / "index.html").is_file():
            return FileResponse(dist / "index.html")
        return {"message": "Chạy frontend bằng npm run dev trong thư mục web/frontend."}

    return app
