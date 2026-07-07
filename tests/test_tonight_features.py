"""今晚实操相关功能:多路径 ingest、视频抽帧校准、run 一键流程、VLM 数据集导出。"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from backend.cli import app
from backend.config import load_settings
from backend.pipeline import project as proj
from backend.pipeline.export_dataset import export_dataset
from backend.pipeline.ingest import collect_videos, ingest
from backend.pipeline.project import Project
from backend.schemas.models import (
    EventsFile, KillEvent, RoundEndEvent, SourceEvents, VideoMeta,
)

from .test_e2e_render import CLIP_DUR, _make_click_track, _make_video

runner = CliRunner()


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("media")
    d = tmp / "batch"
    d.mkdir()
    for i in range(2):
        _make_video(d / f"clip_{i}.mp4", i)
    single = tmp / "extra.mp4"
    _make_video(single, 3)
    bgm = tmp / "bgm.wav"
    _make_click_track(bgm)
    return {"dir": d, "single": single, "bgm": bgm}


def test_collect_videos_mixed_inputs(media):
    videos = collect_videos([media["dir"], media["single"]])
    assert [v.name for v in videos] == ["clip_0.mp4", "clip_1.mp4", "extra.mp4"]
    # 重复输入去重
    videos = collect_videos([media["dir"], media["dir"] / "clip_0.mp4"])
    assert len(videos) == 2


def test_ingest_multiple_paths(media, tmp_path):
    p = Project.init(tmp_path / "proj")
    settings = load_settings(p.root)
    result = ingest(p, [media["dir"], media["single"]], media["bgm"],
                    settings, make_proxies=False)
    assert len(result.sources) == 3
    assert (p.root / "sources" / "extra.mp4").exists()


def test_calibrate_from_video_frame(media, tmp_path, settings):
    """视频 + --at 抽帧 → ROI 标注图(640x360 不在校准分辨率里应报错并提示)。"""
    from backend.pipeline.calibrate import calibrate

    with pytest.raises(ValueError, match="--at"):
        calibrate(media["single"])  # 视频不给 --at 要报错
    with pytest.raises(ValueError, match="不支持的分辨率"):
        calibrate(media["single"], at_s=2.0)  # 抽帧成功但分辨率未校准 → 明确报错
    png = media["single"].with_name("extra_t2.png")
    assert png.exists() and png.stat().st_size > 0  # 抽帧产物本身存在


def test_run_one_shot(media, tmp_path):
    """vmontage run:多输入一键到 preview。用带假击杀横幅的 1080p 合成视频,
    走真实 detect(1080p ROI)→ auto-cut → render 全链。"""
    import shutil

    from tests.test_detector_synthetic import FIXTURES

    src_video = FIXTURES / "synthetic_kills.mp4"
    if not src_video.exists():
        pytest.skip("先跑 test_detector_synthetic 生成合成视频")
    batch = tmp_path / "batch1080"
    batch.mkdir()
    shutil.copy(src_video, batch / "game_a.mp4")
    shutil.copy(src_video, batch / "game_b.mp4")

    root = tmp_path / "oneshot"
    result = runner.invoke(app, [
        "run", str(batch),
        "--music", str(media["bgm"]), "--target", "20", "--dir", str(root),
    ])
    assert result.exit_code == 0, result.output
    assert (root / "output" / "preview.mp4").exists()
    edl = json.loads((root / proj.EDL_JSON).read_text())
    assert len(edl["timeline"]) >= 1


def test_export_vlm_dataset(tmp_path):
    from tests.test_detector_synthetic import FIXTURES

    video = FIXTURES / "synthetic_kills.mp4"
    if not video.exists():
        pytest.skip("先跑 test_detector_synthetic 生成合成视频")
    p = Project.init(tmp_path / "proj")
    (p.root / "sources" / "synthetic_kills.mp4").symlink_to(video)
    events = EventsFile(sources=[SourceEvents(
        source="sources/synthetic_kills.mp4",
        video_meta=VideoMeta(width=1920, height=1080, fps=30, duration_s=12),
        events=[
            KillEvent(frame=120, t=4.0, headshot=True),
            KillEvent(frame=162, t=5.4),
            RoundEndEvent(frame=300, t=10.0, won=True),
        ])])
    p.save(proj.EVENTS_JSON, events)
    settings = load_settings(p.root)
    stats = export_dataset(p, events, p.root / "dataset", settings)
    # 正样本 5(2 杀×2 + 回合 1)+ 负样本(12s 短视频受 3s 间隔限制,至少 1)
    assert stats["train"] + stats["test"] >= 6
    labels = [json.loads(l) for l in
              (p.root / "dataset" / "labels.jsonl").read_text().splitlines()]
    assert all((p.root / "dataset" / r["image"]).exists() for r in labels)
    kill_pos = [r for r in labels if r["label"]["kill_banner"]]
    assert any(r["label"]["headshot"] for r in kill_pos)
    assert any(r["label"]["round_end"] == "won" for r in labels)
    negs = [r for r in labels if not r["label"]["kill_banner"]
            and r["label"]["round_end"] is None and not r["label"]["death"]]
    assert negs, "应有负样本"
    # sft.jsonl 与 labels 行数一致且为合法对话格式
    sft = [json.loads(l) for l in
           (p.root / "dataset" / "sft.jsonl").read_text().splitlines()]
    assert len(sft) == len(labels)
    assert sft[0]["messages"][0]["role"] == "user"
    assert json.loads(sft[0]["messages"][1]["content"])  # assistant 内容是合法 JSON
