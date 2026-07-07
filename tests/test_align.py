from __future__ import annotations

import pytest

from backend.pipeline.align import build_edl
from backend.pipeline.scorer import score_clips
from backend.pipeline.semantic import build_scorecards
from backend.schemas.models import BeatsFile, EventsFile, SourceEvents, VideoMeta

from .conftest import make_kill


def _beats(interval: float = 0.5, n: int = 400) -> BeatsFile:
    return BeatsFile(music="music/bgm.mp3", bpm=60.0 / interval,
                     beats_t=[round(i * interval, 4) for i in range(n)])


def _events_multi() -> EventsFile:
    """三段素材,各一个双杀簇,分数相同。"""
    sources = []
    for i in range(3):
        sources.append(SourceEvents(
            source=f"sources/clip_{i}.mp4",
            video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=60),
            events=[make_kill(20.0), make_kill(22.0)],
        ))
    return EventsFile(sources=sources)


def _cards(events, settings):
    return score_clips(build_scorecards(events, settings), settings)


def test_cuts_snap_to_beats(settings):
    events = _events_multi()
    cards = _cards(events, settings)
    beats = _beats()
    edl, warnings = build_edl(cards, beats, events, settings, target_duration_s=60)
    assert len(edl.timeline) == 3
    assert edl.timeline[0].timeline_start_t == 0.0
    grid = set(beats.beats_t)
    cursor = 0.0
    for entry in edl.timeline[1:]:
        # 验收 #4:切换点与拍点偏差 <= 50ms
        assert entry.snap.mode == "cut_on_beat"
        assert min(abs(entry.timeline_start_t - b) for b in grid) <= 0.05
        # 时间线无空洞:上一片段延长后正好接上
    for i, entry in enumerate(edl.timeline):
        assert entry.timeline_start_t == pytest.approx(cursor, abs=1e-3)
        cursor = entry.timeline_start_t + (entry.out_t - entry.in_t)


def test_fit_duration_drops_low_score(settings):
    events = _events_multi()
    # 给其中一段加一个 flick 使其分数更高
    events.sources[1].events[0].headshot = True
    events.sources[1].events[0].pre_kill_angular_velocity_deg_s = 300.0
    cards = _cards(events, settings)
    beats = _beats()
    # 每段约 2 + 2.5 + 1.2 = 5.7s;目标 10s → 收缩后仍需丢 1 段
    edl, warnings = build_edl(cards, beats, events, settings, target_duration_s=10)
    kept_ids = [e.clip_id for e in edl.timeline]
    assert len(kept_ids) == 2
    high = cards.clips[0]
    assert high.clip_id in kept_ids  # 高分保留
    total = sum(e.out_t - e.in_t for e in edl.timeline)
    assert total <= 10 + 0.5  # 允许最后一段拍点填缝的余量


def test_target_clamped_to_music_length(settings):
    """目标时长超过音乐长度时收紧到最后一个拍点(尾段才有拍点可吸附)。"""
    events = _events_multi()
    cards = _cards(events, settings)
    beats = _beats(interval=0.5, n=40)  # 音乐只有 ~20s
    edl, warnings = build_edl(cards, beats, events, settings, target_duration_s=60)
    total = sum(e.out_t - e.in_t for e in edl.timeline)
    assert total <= beats.beats_t[-1] + 0.5
    assert any("音乐长度" in w for w in warnings)


def test_cap_marathon_clip(settings):
    """65s 马拉松击杀簇裁到 max_clip_s 内,窗口取杀数最密段。"""
    from backend.pipeline.align import Clip, cap_clip_length

    c = Clip(clip_id="c", source="s", in_t=0.0, out_t=65.0,
             anchors=[2.5, 14.0, 25.0, 27.0, 28.5, 30.0, 45.0, 60.0],
             score=5.0, source_duration=70.0)
    cap_clip_length(c, settings)
    assert c.out_t - c.in_t <= 18.0 + 1e-6
    # 最密窗口是 25.0-30.0 那四杀
    assert c.anchors[0] >= 14.0 and 25.0 in c.anchors and 30.0 in c.anchors
    assert len(c.anchors) >= 4


def test_no_selected_raises(settings):
    events = _events_multi()
    cards = _cards(events, settings)
    for c in cards.clips:
        c.selected = False
    with pytest.raises(ValueError):
        build_edl(cards, _beats(), events, settings, 60)


def test_anchor_align_mode(settings):
    events = _events_multi()
    cards = _cards(events, settings)
    beats = _beats()
    edl, _ = build_edl(cards, beats, events, settings, 60, anchor_align=True)
    grid = beats.beats_t
    for entry in edl.timeline:
        if entry.snap.mode != "anchor_align":
            continue
        # 第一个击杀帧(源内 20.0s)在时间线上应落在拍点
        card = next(c for c in cards.clips if c.clip_id == entry.clip_id)
        anchor_timeline = entry.timeline_start_t + (card.anchor_ts[0] - entry.in_t)
        assert min(abs(anchor_timeline - b) for b in grid) <= 0.05
    assert any(e.snap.mode == "anchor_align" for e in edl.timeline)
