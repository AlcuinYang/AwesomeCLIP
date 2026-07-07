"""P2 后端 API 测试:状态读取、EDL 落盘、undo/chat 错误路径、后台渲染、WS 推送。"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.api.server import create_app
from backend.config import load_settings
from backend.pipeline import project as proj
from backend.pipeline.align import build_edl
from backend.pipeline.beat import analyze_music
from backend.pipeline.ingest import ingest
from backend.pipeline.project import Project
from backend.pipeline.scorer import score_clips
from backend.pipeline.semantic import build_scorecards
from backend.schemas.models import EventsFile, SourceEvents, VideoMeta

from .conftest import make_kill
from .test_e2e_render import CLIP_DUR, _make_click_track, _make_video


@pytest.fixture(scope="module")
def api_project(tmp_path_factory) -> Project:
    tmp = tmp_path_factory.mktemp("api")
    raw = tmp / "raw"
    raw.mkdir()
    for i in range(2):
        _make_video(raw / f"clip_{i}.mp4", i)
    bgm = tmp / "bgm.wav"
    _make_click_track(bgm)

    p = Project.init(tmp / "project")
    settings = load_settings(p.root)
    ingest(p, raw, bgm, settings, make_proxies=False)
    events = EventsFile(sources=[
        SourceEvents(
            source=f"sources/clip_{i}.mp4",
            video_meta=VideoMeta(width=640, height=360, fps=30, duration_s=CLIP_DUR),
            events=[make_kill(8.0 + i), make_kill(10.0 + i)],
        ) for i in range(2)
    ])
    p.save(proj.EVENTS_JSON, events)
    cards = score_clips(build_scorecards(events, settings), settings)
    p.save(proj.SCORECARDS_JSON, cards)
    beats = analyze_music(p.resolve_media("music/bgm.wav"), "music/bgm.wav", settings)
    p.save(proj.BEATS_JSON, beats)
    edl, _ = build_edl(cards, beats, events, settings, target_duration_s=30)
    p.save(proj.EDL_JSON, edl)
    return p


@pytest.fixture(scope="module")
def client(api_project) -> TestClient:
    return TestClient(create_app(api_project.root))


def test_health_and_state(client):
    assert client.get("/api/health").json()["ok"] is True
    state = client.get("/api/state").json()
    assert state["edl"]["timeline"]
    assert state["scorecards"]["clips"]
    assert state["beats"]["bpm"] > 0
    assert state["render"]["status"] == "idle"


def test_put_edl_roundtrip_and_ws(client):
    state = client.get("/api/state").json()
    edl = state["edl"]
    edl["timeline"][0]["audio"]["game_volume"] = 0.5
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        assert client.put("/api/edl", json=edl).json()["ok"] is True
        msg = ws.receive_json()
        assert msg == {"type": "edl_updated", "source": "gui"}
    assert client.get("/api/state").json()["edl"]["timeline"][0]["audio"]["game_volume"] == 0.5


def test_undo_without_log_is_400(client):
    resp = client.post("/api/undo")
    assert resp.status_code == 400


def test_chat_without_key_is_400(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    resp = client.post("/api/chat", json={"instruction": "总长压到10秒"})
    assert resp.status_code == 400
    assert "OPENROUTER_API_KEY" in resp.json()["detail"]


def test_media_static(client):
    resp = client.get("/media/edl.json")
    assert resp.status_code == 200
    assert resp.json()["timeline"]


def test_render_background_job(client, api_project):
    resp = client.post("/api/render", json={"preview": True})
    assert resp.json()["started"] is True
    # 忙碌时二次提交被拒
    assert client.post("/api/render", json={"preview": True}).status_code in (200, 409)
    deadline = time.time() + 60
    while time.time() < deadline:
        status = client.get("/api/render/status").json()
        if not status["busy"] and status["last"]["status"] != "idle":
            break
        time.sleep(0.3)
    assert status["last"]["status"] == "done", status
    out = api_project.root / status["last"]["path"]
    assert out.exists() and out.stat().st_size > 10_000
    # 渲染产物可经静态挂载访问(GUI 回放导出预览)
    assert client.get(f"/media/{status['last']['path']}").status_code == 200
