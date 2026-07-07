"""P2 后端:REST + WebSocket(进度推送),前端 GUI 的唯一数据源(规格 §3)。

设计:
- 无数据库,直接读写项目目录的 JSON(与 CLI 完全同一套 pipeline 代码)。
- GET /api/state 一次拿全(edl + scorecards + beats),GUI 启动即加载。
- PUT /api/edl 落盘 GUI 时间线微调;POST /api/chat 走 NL agent(同样写 agent_log)。
- POST /api/render 后台线程渲染,进度经 WebSocket /ws 广播;同时只允许一个任务。
- /media/ 静态挂载项目目录(proxies/output/music),浏览器只播 proxy。
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import load_settings
from ..pipeline import project as proj
from ..pipeline.project import Project
from ..schemas.models import BeatsFile, EdlFile, EventsFile, ScorecardsFile


class ChatRequest(BaseModel):
    instruction: str


class RenderRequest(BaseModel):
    preview: bool = True


class _Broadcaster:
    """把后台线程的事件安全地广播给所有 WebSocket 客户端。"""

    def __init__(self) -> None:
        self.connections: list[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.loop = asyncio.get_running_loop()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)

    def send(self, payload: dict[str, Any]) -> None:
        """线程安全;无事件循环/无连接时静默丢弃。"""
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        text = json.dumps(payload, ensure_ascii=False)

        async def _fanout() -> None:
            for ws in list(self.connections):
                try:
                    await ws.send_text(text)
                except Exception:
                    self.disconnect(ws)

        asyncio.run_coroutine_threadsafe(_fanout(), loop)


class RenderJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.busy = False
        self.last: dict[str, Any] = {"status": "idle"}

    def start(self, target) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=target, daemon=True).start()
        return True

    def finish(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.busy = False
            self.last = result


def create_app(project_root: Path) -> FastAPI:
    project = Project(Path(project_root))
    app = FastAPI(title="vmontage", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    broadcaster = _Broadcaster()
    job = RenderJob()
    app.state.project = project

    def settings() -> dict:
        return load_settings(project.root)

    def _load_or_none(name: str, cls):
        try:
            return project.load(name, cls).model_dump()
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------- 状态
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "project": str(project.root)}

    @app.get("/api/state")
    def state() -> dict:
        return {
            "project": str(project.root),
            "edl": _load_or_none(proj.EDL_JSON, EdlFile),
            "scorecards": _load_or_none(proj.SCORECARDS_JSON, ScorecardsFile),
            "beats": _load_or_none(proj.BEATS_JSON, BeatsFile),
            "events": _load_or_none(proj.EVENTS_JSON, EventsFile),
            "render": job.last,
        }

    @app.put("/api/edl")
    def put_edl(edl: EdlFile) -> dict:
        project.save(proj.EDL_JSON, edl)
        broadcaster.send({"type": "edl_updated", "source": "gui"})
        return {"ok": True}

    # ------------------------------------------------------------- NL agent
    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict:
        from ..agent.dsl import AgentSession
        from ..agent.llm import LlmError, OpenRouterClient, run_agent

        try:
            client = OpenRouterClient(settings())
        except LlmError as e:
            raise HTTPException(400, str(e))
        try:
            session = AgentSession(project, settings())
        except FileNotFoundError as e:
            raise HTTPException(409, str(e))
        ops: list[str] = []

        def on_event(line: str) -> None:
            ops.append(line)
            broadcaster.send({"type": "agent_op", "text": line})

        summary = run_agent(session, req.instruction, client, on_event=on_event)
        broadcaster.send({"type": "edl_updated", "source": "agent"})
        return {"summary": summary, "ops": ops}

    @app.post("/api/undo")
    def undo() -> dict:
        from ..agent.dsl import AgentSession, DslError

        try:
            result = AgentSession(project, settings()).undo()
        except (DslError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        broadcaster.send({"type": "edl_updated", "source": "undo"})
        return {"result": result}

    # ------------------------------------------------------------- 渲染
    @app.post("/api/render")
    def start_render(req: RenderRequest) -> dict:
        from ..pipeline.render import render as run_render

        try:
            edl = project.load(proj.EDL_JSON, EdlFile)
        except FileNotFoundError as e:
            raise HTTPException(409, str(e))
        name = "preview.mp4" if req.preview else "final.mp4"
        cfg = settings()

        def work() -> None:
            broadcaster.send({"type": "render_progress", "fraction": 0.0})
            try:
                out, warnings = run_render(
                    project, edl, project.root / "output" / name, cfg,
                    preview=req.preview,
                    on_progress=lambda f: broadcaster.send(
                        {"type": "render_progress", "fraction": round(f, 3)}))
                result = {"status": "done", "path": f"output/{name}",
                          "warnings": warnings}
                broadcaster.send({"type": "render_done", **result})
            except Exception as e:  # 渲染失败也要解锁并通知
                result = {"status": "error", "error": str(e)[-2000:]}
                broadcaster.send({"type": "render_error", **result})
            job.finish(result)

        if not job.start(work):
            raise HTTPException(409, "已有渲染任务在进行中。")
        return {"started": True, "output": f"output/{name}"}

    @app.get("/api/render/status")
    def render_status() -> dict:
        return {"busy": job.busy, "last": job.last}

    # ------------------------------------------------------------- WebSocket
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await broadcaster.connect(ws)
        try:
            await ws.send_text(json.dumps({"type": "hello",
                                           "project": str(project.root)}))
            while True:
                await ws.receive_text()  # 客户端消息忽略,仅保活
        except WebSocketDisconnect:
            broadcaster.disconnect(ws)

    # 静态媒体:浏览器只播 proxy;output 用于导出预览回放
    app.mount("/media", StaticFiles(directory=project.root), name="media")
    return app
