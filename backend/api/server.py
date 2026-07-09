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

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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


class DirectRequest(BaseModel):
    style_hint: Optional[str] = None
    with_frames: bool = True


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

    # ------------------------------------------------------------- 素材导入与检测
    @app.post("/api/upload")
    async def upload(files: list[UploadFile]) -> dict:
        """浏览器直传素材 → sources/;后台生成 proxy。"""
        from ..pipeline.ffmpeg_utils import spawn_ffmpeg
        from ..pipeline.ingest import VIDEO_EXTS

        saved = []
        proxy_cfg = settings()["render"]["proxy"]
        for f in files:
            name = Path(f.filename or "clip.mp4").name
            if Path(name).suffix.lower() not in VIDEO_EXTS:
                raise HTTPException(400, f"不支持的格式: {name}")
            dest = project.root / "sources" / name
            with open(dest, "wb") as out:
                while chunk := await f.read(8 * 1024 * 1024):
                    out.write(chunk)
            proxy_out = project.root / "proxies" / (dest.stem + ".mp4")
            spawn_ffmpeg(["-i", str(dest), "-vf", f"scale=-2:{proxy_cfg['height']}",
                          "-c:v", "libx264", "-preset", "veryfast",
                          "-b:v", str(proxy_cfg["bitrate"]),
                          "-c:a", "aac", "-b:a", "96k", str(proxy_out)])
            saved.append(f"sources/{name}")
        broadcaster.send({"type": "uploaded", "sources": saved})
        return {"saved": saved}

    @app.post("/api/detect")
    def start_detect() -> dict:
        """后台检测新素材(已检测过的跳过)→ 更新 events/scorecards;
        无 EDL 时自动无音乐 auto-cut,让时间线与聊天立即可用。"""
        from ..config import load_roi_profile
        from ..pipeline.align import build_edl
        from ..pipeline.detector import Detector
        from ..pipeline.ffmpeg_utils import ffprobe_meta
        from ..pipeline.ingest import VIDEO_EXTS
        from ..pipeline.scorer import score_clips
        from ..pipeline.semantic import build_scorecards

        cfg = settings()
        videos = sorted(f for f in (project.root / "sources").iterdir()
                        if f.suffix.lower() in VIDEO_EXTS)
        if not videos:
            raise HTTPException(409, "sources/ 为空,请先上传素材。")

        def work() -> None:
            try:
                events = (project.load(proj.EVENTS_JSON, EventsFile)
                          if project.exists(proj.EVENTS_JSON) else EventsFile())
                done_sources = {s.source for s in events.sources}
                todo = [v for v in videos if f"sources/{v.name}" not in done_sources]
                for i, v in enumerate(todo):
                    broadcaster.send({"type": "detect_progress",
                                      "file": v.name, "done": i, "total": len(todo)})
                    meta = ffprobe_meta(v)
                    profile = load_roi_profile(meta["width"], meta["height"],
                                               cfg, project.root)
                    det = Detector(cfg, profile,
                                   project_templates_dir=project.root / "templates")
                    events.sources.append(det.detect(v, f"sources/{v.name}"))
                project.save(proj.EVENTS_JSON, events)
                cards = score_clips(build_scorecards(events, cfg), cfg)
                project.save(proj.SCORECARDS_JSON, cards)
                if not project.exists(proj.EDL_JSON):
                    beats = (project.load(proj.BEATS_JSON, BeatsFile)
                             if project.exists(proj.BEATS_JSON) else None)
                    edl, _ = build_edl(cards, beats, events, cfg,
                                       target_duration_s=1e9)
                    project.save(proj.EDL_JSON, edl)
                sel = sum(1 for c in cards.clips if c.selected)
                result = {"status": "done", "clips": len(cards.clips), "selected": sel}
                broadcaster.send({"type": "detect_done", **result})
                broadcaster.send({"type": "edl_updated", "source": "detect"})
            except Exception as e:
                result = {"status": "error", "error": str(e)[-2000:]}
                broadcaster.send({"type": "detect_error", **result})
            job.finish(result)

        if not job.start(work):
            raise HTTPException(409, "已有任务在进行中。")
        return {"started": True, "videos": len(videos)}

    @app.post("/api/direct")
    def start_direct(req: DirectRequest) -> dict:
        """后台运行 MLLM 导演编排(2~3 分钟)→ storyboard.json + edl.json。"""
        from ..agent.director import direct as run_direct
        from ..agent.llm import LlmError, OpenRouterClient

        try:
            client = OpenRouterClient(settings())
        except LlmError as e:
            raise HTTPException(400, str(e))

        def work() -> None:
            try:
                sb, edl, warnings = run_direct(project, settings(), client,
                                               style_hint=req.style_hint,
                                               with_frames=req.with_frames)
                result = {"status": "done", "shots": len(sb.shots),
                          "duration_s": edl.target_duration_s, "warnings": warnings}
                broadcaster.send({"type": "direct_done", **result})
                broadcaster.send({"type": "edl_updated", "source": "director"})
            except Exception as e:
                result = {"status": "error", "error": str(e)[-2000:]}
                broadcaster.send({"type": "direct_error", **result})
            job.finish(result)

        if not job.start(work):
            raise HTTPException(409, "已有任务在进行中。")
        broadcaster.send({"type": "direct_started"})
        return {"started": True}

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
