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
    assert len(cards.clips) == 2  # 13→30 间隔 17s > merge_gap=12,分簇
    assert cards.clips[0].tags == ["multikill_2"]


def test_multikill_is_cluster_total(settings):
    """用户定义:片段内不连续击杀也算——1,2,1 分布(间隔 <12s 且中间没死)= 四杀。"""
    src = SourceEvents(
        source="sources/a.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=100),
        events=[make_kill(10.0),                    # 1
                make_kill(20.0), make_kill(21.5),   # 2
                make_kill(30.0)],                   # 1
    )
    cards = build_scorecards(EventsFile(sources=[src]), settings)
    # 全簇 = 四杀;跨度 23.7s > max_clip_s=18 → 切成段但标签共享、击杀全保留
    assert all(c.tags == ["multikill_4"] for c in cards.clips)
    all_anchors = sorted(a for c in cards.clips for a in c.anchor_ts)
    assert all_anchors == [10.0, 20.0, 21.5, 30.0]


def test_round_boundary_breaks_cluster(settings):
    """回合切换断簇:跨回合的击杀不能并成一个多杀(用户指出的 test5 问题)。"""
    from backend.schemas.models import RoundBoundaryEvent

    src = SourceEvents(
        source="sources/a.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=100),
        events=[make_kill(10.0), make_kill(12.0),
                RoundBoundaryEvent(frame=900, t=15.0),
                make_kill(18.0), make_kill(20.0), make_kill(22.0)],
    )
    cards = build_scorecards(EventsFile(sources=[src]), settings)
    assert len(cards.clips) == 2
    assert cards.clips[0].tags == ["multikill_2"]
    assert cards.clips[1].tags == ["multikill_3"]


def test_long_cluster_splits_keeping_all_kills(settings):
    """超长簇分段:五杀跨 30s,不丢第 5 杀——切成多段,各段共享 multikill_5。"""
    kills = [10.0, 12.0, 13.5, 15.0, 32.0]  # 第 4-5 杀之间 17s 空档
    src = SourceEvents(
        source="sources/a.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=100),
        events=[make_kill(t) for t in kills],
    )
    # 间隔 17s > merge_gap=12 会分簇;本测试针对切段逻辑,临时放大 merge_gap
    import copy
    s = copy.deepcopy(settings)
    s["semantic"]["clip"]["merge_gap_s"] = 20.0
    cards = build_scorecards(EventsFile(sources=[src]), s)
    assert len(cards.clips) == 2
    assert all(c.tags == ["multikill_5"] for c in cards.clips)  # 共享全簇标签
    all_anchors = sorted(a for c in cards.clips for a in c.anchor_ts)
    assert all_anchors == kills  # 一个击杀都不丢
    assert all(c.span.end_t - c.span.start_t <= 18.0 + 1e-6 for c in cards.clips)


def test_death_breaks_cluster(settings):
    """自己被杀强制断簇:死前 2 杀 + 死后 2 杀 = 两个双杀片段,不是四杀。"""
    from backend.schemas.models import DeathEvent

    src = SourceEvents(
        source="sources/a.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=100),
        events=[make_kill(10.0), make_kill(12.0),
                DeathEvent(frame=900, t=15.0),
                make_kill(18.0), make_kill(20.0)],
    )
    cards = build_scorecards(EventsFile(sources=[src]), settings)
    assert len(cards.clips) == 2
    assert all(c.tags == ["multikill_2"] for c in cards.clips)


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
