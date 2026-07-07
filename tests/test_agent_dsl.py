"""P1 验收:附录 A 五条中文指令对应的 DSL 落地 + undo + map_section + agent 循环。

LLM 网络层用 httpx.MockTransport 模拟(离线);真实 OpenRouter 只在有
OPENROUTER_API_KEY 时手工验证。
"""
from __future__ import annotations

import json

import httpx
import pytest

from backend.agent.dsl import AgentSession, DslError
from backend.config import load_settings
from backend.pipeline import project as proj
from backend.pipeline.align import build_edl
from backend.pipeline.project import Project
from backend.pipeline.scorer import score_clips
from backend.pipeline.semantic import build_scorecards
from backend.schemas.models import (
    AliveStateEvent, BeatsFile, EnergyPoint, EventsFile, RoundEndEvent,
    SourceEvents, VideoMeta,
)

from .conftest import make_kill


def _beats() -> BeatsFile:
    return BeatsFile(
        music="music/bgm.wav", bpm=120.0,
        beats_t=[round(i * 0.5, 4) for i in range(200)],
        energy_curve=[EnergyPoint(t=round(i * 0.5, 3), rms=round(0.01 * i, 4))
                      for i in range(200)],
    )


@pytest.fixture
def session(tmp_path) -> AgentSession:
    """4 个片段:A=残局三杀带 flick(9 分)、B=三杀(3)、C=双杀(2)、D=单杀(0,未选)。"""
    p = Project.init(tmp_path / "project")
    settings = load_settings(p.root)
    events = EventsFile(sources=[SourceEvents(
        source="sources/game.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=100),
        events=[
            AliveStateEvent(frame=600, t=10.0, ally_alive=1, enemy_alive=3),
            make_kill(12.0, headshot=True, angvel=300.0),
            make_kill(14.0), make_kill(16.0),
            RoundEndEvent(frame=1200, t=20.0, won=True),
            make_kill(40.0), make_kill(42.0), make_kill(44.0),
            make_kill(60.0), make_kill(62.0),
            make_kill(80.0),
        ])])
    p.save(proj.EVENTS_JSON, events)
    cards = score_clips(build_scorecards(events, settings), settings)
    p.save(proj.SCORECARDS_JSON, cards)
    beats = _beats()
    p.save(proj.BEATS_JSON, beats)
    edl, _ = build_edl(cards, beats, events, settings, target_duration_s=60)
    p.save(proj.EDL_JSON, edl)
    return AgentSession(p, settings)


def _order(s: AgentSession) -> list[str]:
    return [e.clip_id for e in s.edl.timeline]


def _assert_cuts_on_beat(s: AgentSession):
    for e in s.edl.timeline[1:]:
        assert min(abs(e.timeline_start_t - b) for b in s.beats.beats_t) <= 0.05, \
            f"{e.clip_id} 切点 {e.timeline_start_t} 不在拍点上"


def test_fixture_shape(session):
    tags = {c.clip_id: c.tags for c in session.cards.clips}
    assert tags["clip_001"] == ["multikill_3", "clutch_1v3", "flick"]
    assert tags["clip_002"] == ["multikill_3"]
    assert tags["clip_003"] == ["multikill_2"]
    assert tags["clip_004"] == []
    assert _order(session) == ["clip_001", "clip_002", "clip_003"]


# ---------------------------------------------------- 附录 A 五条验收指令

def test_a1_only_clutch_and_triple_plus(session):
    """「只要残局和三杀以上的片段」→ select only clutch,再 add multikill>=3。"""
    session.apply("select", {"mode": "only", "tags_any": ["clutch"]})
    session.apply("select", {"mode": "add", "min_multikill": 3})
    selected = {c.clip_id for c in session.cards.clips if c.selected}
    assert selected == {"clip_001", "clip_002"}
    assert set(_order(session)) == {"clip_001", "clip_002"}
    _assert_cuts_on_beat(session)


def test_a2_best_first_rest_by_time(session):
    """「把分数最高的放开头,其余按时间顺序」→ reorder。"""
    session.apply("reorder", {"order": ["clip_003", "clip_002", "clip_001"]})
    assert _order(session) == ["clip_003", "clip_002", "clip_001"]
    # 最高分 clip_001 开头,其余(002 span 37.5s 起、003 span 57.5s 起)按时间
    session.apply("reorder", {"order": ["clip_001", "clip_002", "clip_003"]})
    assert _order(session) == ["clip_001", "clip_002", "clip_003"]
    _assert_cuts_on_beat(session)


def test_a3_fit_to_duration(session):
    """「总长压到一分钟」→ fit_duration;此处用更紧的 12s 逼出收缩+丢弃。"""
    before_total = sum(e.out_t - e.in_t for e in session.edl.timeline)
    assert before_total > 12
    result = session.apply("fit_duration", {"target_s": 12})
    total = sum(e.out_t - e.in_t for e in session.edl.timeline)
    assert total <= 12 + 0.5  # 拍点填缝余量
    assert "clip_003" not in _order(session)  # 最低分被丢
    card3 = next(c for c in session.cards.clips if c.clip_id == "clip_003")
    assert card3.selected is False
    assert session.edl.target_duration_s == 12
    _assert_cuts_on_beat(session)


def test_a4_trim_more_pre_kill(session):
    """「第二个片段击杀前多留一秒」→ trim(clip='2', in_delta_s=-1)。"""
    before_in = session.edl.timeline[1].in_t
    session.apply("trim", {"clip": "2", "in_delta_s": -1.0})
    assert session.edl.timeline[1].in_t == pytest.approx(before_in - 1.0, abs=1e-3)
    _assert_cuts_on_beat(session)


def test_a5_frame_drop_on_last(session):
    """「给最后一个片段开抽帧,强度调低一点」→ set_effect(last, frame_drop, 0.3)。"""
    session.apply("set_effect", {"target": "last", "effect": "frame_drop",
                                 "enabled": True, "strength": 0.3})
    last = session.edl.timeline[-1]
    assert last.effects.frame_drop is True
    assert last.effects.frame_drop_strength == 0.3
    assert all(not e.effects.frame_drop for e in session.edl.timeline[:-1])


# ---------------------------------------------------- 其余操作与撤销

def test_undo_restores_state_and_log(session):
    p = session.project
    before_in = session.edl.timeline[1].in_t
    session.apply("trim", {"clip": "2", "in_delta_s": -1.0})
    session.apply("set_effect", {"target": "last", "effect": "frame_drop",
                                 "enabled": True})
    log = (p.root / proj.AGENT_LOG).read_text().splitlines()
    assert len(log) == 2
    session.apply("undo", {})
    assert session.edl.timeline[-1].effects.frame_drop is False
    session.apply("undo", {})
    assert session.edl.timeline[1].in_t == pytest.approx(before_in)
    assert (p.root / proj.AGENT_LOG).read_text().strip() == ""
    with pytest.raises(DslError):
        session.apply("undo", {})


def test_undo_after_select_restores_scorecards(session):
    session.apply("select", {"mode": "only", "tags_any": ["clutch"]})
    assert len(_order(session)) == 1
    session.apply("undo", {})
    assert _order(session) == ["clip_001", "clip_002", "clip_003"]
    assert sum(c.selected for c in session.cards.clips) == 3


def test_deselect_and_speed_ramp_anchor(session):
    session.apply("deselect", {"clip_ids": ["clip_003"]})
    assert _order(session) == ["clip_001", "clip_002"]
    session.apply("set_effect", {"target": "clip_001", "effect": "speed_ramp"})
    ramp = session.edl.timeline[0].effects.speed_ramp
    assert ramp is not None
    assert ramp.anchor_t == 12.0  # 第一个击杀帧
    assert ramp.factor == 0.5


def test_map_section_energy(session):
    """能量曲线单调上升 → 最高分片段应排到最后。"""
    session.apply("map_section", {"strategy": "energy"})
    assert _order(session)[-1] == "clip_001"
    _assert_cuts_on_beat(session)


def test_dsl_errors(session):
    with pytest.raises(DslError):
        session.apply("select", {"mode": "only", "min_score": 99})
    with pytest.raises(DslError):
        session.apply("trim", {"clip": "9"})
    with pytest.raises(DslError):
        session.apply("nonexistent_op", {})
    # 失败的操作不写日志
    assert not (session.project.root / proj.AGENT_LOG).exists()


def test_persistence_reload(session):
    """操作落盘:新建 session 能看到上一个 session 的改动。"""
    session.apply("trim", {"clip": "1", "in_delta_s": -0.5})
    fresh = AgentSession(session.project, session.settings)
    assert fresh.edl.timeline[0].in_t == session.edl.timeline[0].in_t


def test_custom_api_key_env(session, monkeypatch):
    """api_key_env 可配置:直连 Kimi/火山方舟等厂商时用各自的 key 环境变量。"""
    import copy

    from backend.agent.llm import LlmError, OpenRouterClient

    settings = copy.deepcopy(session.settings)
    settings["agent"]["api_key_env"] = "MOONSHOT_API_KEY"
    settings["agent"]["base_url"] = "https://api.moonshot.cn/v1"
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(LlmError, match="MOONSHOT_API_KEY"):
        OpenRouterClient(settings)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    client = OpenRouterClient(settings)
    assert client.api_key == "sk-test"
    assert client.base_url == "https://api.moonshot.cn/v1"


def test_kimi_k26_compat(session, monkeypatch):
    """Kimi K2.6 约束:temperature=null 不发送、extra_body 并入、
    多步工具调用透传 reasoning_content。"""
    import copy

    from backend.agent.llm import OpenRouterClient, run_agent

    settings = copy.deepcopy(session.settings)
    settings["agent"]["temperature"] = None
    settings["agent"]["extra_body"] = {"thinking": {"type": "disabled"}}
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            msg = {"role": "assistant", "content": None,
                   "reasoning_content": "让我压缩时长…",
                   "tool_calls": [{
                       "id": "c1", "type": "function",
                       "function": {"name": "fit_duration",
                                    "arguments": json.dumps({"target_s": 15})}}]}
        else:
            msg = {"role": "assistant", "content": "已压到 15 秒。"}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
    summary = run_agent(session, "压到15秒", client)
    assert summary == "已压到 15 秒。"
    assert "temperature" not in bodies[0]
    assert bodies[0]["thinking"] == {"type": "disabled"}
    # 第二轮请求的上下文中,assistant 消息必须保留 reasoning_content
    assistant = next(m for m in bodies[1]["messages"] if m["role"] == "assistant")
    assert assistant.get("reasoning_content") == "让我压缩时长…"


# ---------------------------------------------------- mock LLM 的 agent 循环

def test_run_agent_with_mock_llm(session, monkeypatch):
    from backend.agent.llm import OpenRouterClient, run_agent

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            body = json.loads(request.content)
            assert any(t["function"]["name"] == "fit_duration" for t in body["tools"])
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "fit_duration",
                             "arguments": json.dumps({"target_s": 12})}}]}
        else:
            msg = {"role": "assistant", "content": "已把总长压到 12 秒。"}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = OpenRouterClient(session.settings,
                              transport=httpx.MockTransport(handler))
    summary = run_agent(session, "总长压到12秒", client)
    assert summary == "已把总长压到 12 秒。"
    assert calls["n"] == 2
    total = sum(e.out_t - e.in_t for e in session.edl.timeline)
    assert total <= 12.5
    # 工具调用写入了 agent_log
    log = (session.project.root / proj.AGENT_LOG).read_text().splitlines()
    assert len(log) == 1 and json.loads(log[0])["op"] == "fit_duration"
