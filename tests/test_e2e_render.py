"""合成素材端到端:ingest → (伪造 events) → semantic → score → beat → align → render。

对应验收 #1(流程跑通)、#4(切点对拍)、#6(改 edl 重渲无需 detect)。
检测本身在 test_detector_synthetic 中单独验证(真实 ROI 需真素材校准)。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.config import load_settings
from backend.pipeline import project as proj
from backend.pipeline.align import build_edl
from backend.pipeline.beat import analyze_music
from backend.pipeline.ffmpeg_utils import ffprobe_meta
from backend.pipeline.ingest import ingest
from backend.pipeline.project import Project
from backend.pipeline.render import render
from backend.pipeline.scorer import score_clips
from backend.pipeline.semantic import build_scorecards
from backend.schemas.models import EdlFile, EventsFile, SourceEvents, VideoMeta

from .conftest import make_kill

CLIP_DUR = 20


def _make_video(path: Path, hue_shift: int) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={CLIP_DUR}",
        "-f", "lavfi", "-i", f"sine=frequency={300 + hue_shift * 100}:duration={CLIP_DUR}",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
        str(path),
    ], check=True)


def _make_click_track(path: Path, bpm: float = 120.0, duration: float = 60.0) -> None:
    sr = 22050
    y = np.zeros(int(sr * duration), dtype=np.float32)
    interval = 60.0 / bpm
    click_len = int(0.03 * sr)
    tt = np.arange(click_len) / sr
    click = (np.sin(2 * np.pi * 1000 * tt) * np.exp(-tt * 120)).astype(np.float32)
    t = 0.0
    while t < duration - 0.1:
        i = int(t * sr)
        y[i:i + click_len] += click
        t += interval
    sf.write(str(path), y, sr)


@pytest.fixture(scope="module")
def pipeline_project(tmp_path_factory) -> tuple[Project, dict]:
    tmp = tmp_path_factory.mktemp("e2e")
    raw = tmp / "raw"
    raw.mkdir()
    for i in range(3):
        _make_video(raw / f"clip_{i}.mp4", i)
    bgm = tmp / "bgm.wav"
    _make_click_track(bgm)

    p = Project.init(tmp / "project")
    settings = load_settings(p.root)
    result = ingest(p, raw, bgm, settings, make_proxies=False)
    assert len(result.sources) == 3 and result.music is not None

    # 伪造 L1 事件(合成视频无真实 HUD):每段一个双杀簇
    events = EventsFile(sources=[
        SourceEvents(
            source=f"sources/clip_{i}.mp4",
            video_meta=VideoMeta(width=640, height=360, fps=30, duration_s=CLIP_DUR),
            events=[make_kill(8.0 + i), make_kill(10.0 + i)],
        ) for i in range(3)
    ])
    p.save(proj.EVENTS_JSON, events)
    cards = score_clips(build_scorecards(events, settings), settings)
    for c in cards.clips:  # 双杀分 2.0 恰好等于 min_score,显式确认
        assert c.selected
    p.save(proj.SCORECARDS_JSON, cards)

    beats = analyze_music(p.resolve_media("music/bgm.wav"), "music/bgm.wav", settings)
    p.save(proj.BEATS_JSON, beats)
    assert 100 <= beats.bpm <= 140, f"点击音轨应测出 ~120 BPM,实际 {beats.bpm}"

    edl, warnings = build_edl(cards, beats, events, settings, target_duration_s=30)
    p.save(proj.EDL_JSON, edl)
    return p, settings


def test_full_pipeline_render(pipeline_project):
    p, settings = pipeline_project
    edl = p.load(proj.EDL_JSON, EdlFile)
    out, warnings = render(p, edl, p.root / "output" / "preview.mp4", settings,
                           preview=True)
    assert out.exists()
    meta = ffprobe_meta(out)
    expected = sum(e.out_t - e.in_t for e in edl.timeline)
    assert meta["duration_s"] == pytest.approx(expected, abs=0.5)
    assert meta["height"] == 720  # preview 固定 720p(规格 §5.8)
    # 切换点全部落在拍点(验收 #4)
    beats = p.load(proj.BEATS_JSON, __import__("backend.schemas.models",
                                               fromlist=["BeatsFile"]).BeatsFile)
    for entry in edl.timeline[1:]:
        assert entry.snap.mode == "cut_on_beat"
        assert min(abs(entry.timeline_start_t - b) for b in beats.beats_t) <= 0.05


def test_edit_edl_and_rerender(pipeline_project):
    """验收 #6:改 edl.json 的 in_t 后直接重渲,无需重新 detect。"""
    p, settings = pipeline_project
    edl = p.load(proj.EDL_JSON, EdlFile)
    edl.timeline[0].in_t = round(edl.timeline[0].in_t + 1.0, 3)
    p.save(proj.EDL_JSON, edl)
    edl2 = p.load(proj.EDL_JSON, EdlFile)
    out, _ = render(p, edl2, p.root / "output" / "preview2.mp4", settings,
                    preview=True)
    meta = ffprobe_meta(out)
    expected = sum(e.out_t - e.in_t for e in edl2.timeline)
    assert meta["duration_s"] == pytest.approx(expected, abs=0.5)


def test_effects_render(pipeline_project):
    """frame_drop 保时长;speed_ramp 慢放按预期增加时长。"""
    from backend.pipeline.effects import clip_extra_duration
    from backend.schemas.models import SpeedRamp

    p, settings = pipeline_project
    edl = p.load(proj.EDL_JSON, EdlFile)
    edl.timeline[0].effects.frame_drop = True
    edl.timeline[0].effects.frame_drop_strength = 0.5
    anchor = (edl.timeline[1].in_t + edl.timeline[1].out_t) / 2
    edl.timeline[1].effects.speed_ramp = SpeedRamp(anchor_t=anchor)
    extra = sum(clip_extra_duration(e, False) for e in edl.timeline)
    assert extra == pytest.approx(0.8, abs=1e-6)  # (0.3+0.5)*(1/0.5-1)

    out, _ = render(p, edl, p.root / "output" / "preview_fx.mp4", settings,
                    preview=True)
    meta = ffprobe_meta(out)
    expected = sum(e.out_t - e.in_t for e in edl.timeline) + extra
    assert meta["duration_s"] == pytest.approx(expected, abs=0.6)
