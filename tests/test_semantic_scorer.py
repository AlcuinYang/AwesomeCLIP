from __future__ import annotations

from backend.pipeline.scorer import score_clips
from backend.pipeline.semantic import build_scorecards
from backend.schemas.models import EventsFile, SourceEvents, VideoMeta

from .conftest import make_kill


def test_clutch_multikill_flick_tags(clutch_events, settings):
    cards = build_scorecards(clutch_events, settings)
    assert len(cards.clips) == 2  # 三杀簇 + 孤立单杀(间隔 > merge_gap)

    main = cards.clips[0]
    assert main.tags == ["multikill_3", "clutch_1v3", "flick"] or \
        set(main.tags) == {"multikill_3", "clutch_1v3", "flick"}
    assert main.anchor_ts == [30.57, 32.02, 33.50]
    # 证据卡字段完整(验收 #3)
    assert main.evidence.alive_state.enemy == 3
    assert main.evidence.round_won.t == 34.92
    assert len(main.evidence.kills) == 3
    assert main.evidence.kills[0].pre_kill_angular_velocity_deg_s == 236.0
    # 片段外扩:pre 2.5 / post 1.2
    assert abs(main.span.start_t - (30.57 - 2.5)) < 1e-6
    assert abs(main.span.end_t - (33.50 + 1.2)) < 1e-6

    lone = cards.clips[1]
    assert lone.tags == []


def test_merge_gap_clusters(settings):
    src = SourceEvents(
        source="sources/a.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=100),
        events=[make_kill(10.0), make_kill(13.0), make_kill(30.0)],
    )
    cards = build_scorecards(EventsFile(sources=[src]), settings)
    assert len(cards.clips) == 2
    assert cards.clips[0].tags == ["multikill_2"]


def test_multikill_window(settings):
    # 间隔 4s < merge_gap=5 全部并簇;8s 滑动窗口内最多 3 杀(10,14,18)
    src = SourceEvents(
        source="sources/a.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=100),
        events=[make_kill(t) for t in [10.0, 14.0, 18.0, 22.0]],
    )
    cards = build_scorecards(EventsFile(sources=[src]), settings)
    assert len(cards.clips) == 1
    assert cards.clips[0].tags == ["multikill_3"]


def test_scoring_and_selection(clutch_events, settings):
    cards = score_clips(build_scorecards(clutch_events, settings), settings)
    main = cards.clips[0]  # 排序后最高分在前
    # clutch 1v3: 3*1.5=4.5; multikill_3: 3*1.0=3.0; flick: 1.5; round_context: 0
    assert main.score.breakdown["clutch"] == 4.5
    assert main.score.breakdown["multikill"] == 3.0
    assert main.score.breakdown["flick"] == 1.5
    assert main.score.total == 9.0
    assert main.selected is True

    lone = cards.clips[1]
    assert lone.score.total == 0.0
    assert lone.selected is False  # 低于 min_score=2.0
