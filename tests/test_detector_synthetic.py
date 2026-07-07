"""合成帧/合成视频上的 L1 检测器测试(信息流金框方案,无真实素材时的最低保障)。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config import load_roi_profile
from backend.pipeline.detector import Detector, Roi

FIXTURES = Path(__file__).parent / "fixtures"
W, H, FPS = 1920, 1080, 30
GOLD = (40, 200, 250)  # BGR ≈ HSV H~27 S~215 V~250,落入金框阈值


def _profile(settings):
    return load_roi_profile(W, H, settings)


def _blank_frame() -> np.ndarray:
    return np.full((H, W, 3), 40, dtype=np.uint8)  # 深灰背景


def _draw_ring(frame: np.ndarray, right_frac: float, y_frac: float,
               w: int = 54, h: int = 32, thickness: int = 3,
               row_color: tuple = (90, 120, 90), label: str = "ROW") -> None:
    """画一条信息流行:行底色条 + 文字纹理(供指纹区分)+ 金色空心高亮框。"""
    x1 = int(right_frac * W)
    x0 = x1 - w
    y0 = int(y_frac * H)
    bar_x0 = int(0.72 * W)
    cv2.rectangle(frame, (bar_x0, y0), (int(0.985 * W), y0 + h), row_color, -1)
    # 每行独特的文字纹理(真实行有玩家名/武器图标,指纹靠内容区分行)
    cv2.putText(frame, label * 3, (bar_x0 + 70, y0 + h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + h), GOLD, thickness)


def _draw_gold_blob(frame: np.ndarray, right_frac: float, y_frac: float) -> None:
    """实心金色块(模拟金发头像),不应被识别为金框。"""
    x1 = int(right_frac * W)
    cv2.rectangle(frame, (x1 - 20, int(y_frac * H)), (x1, int(y_frac * H) + 12),
                  GOLD, -1)


def test_feed_ring_detection(settings):
    det = Detector(settings, _profile(settings))
    frame = _blank_frame()
    assert det._feed_rings(frame) == []
    _draw_ring(frame, right_frac=0.87, y_frac=0.07)   # 我的击杀(行中部)
    _draw_ring(frame, right_frac=0.97, y_frac=0.105)  # 我被击杀(行尾)
    _draw_gold_blob(frame, right_frac=0.96, y_frac=0.14)  # 金发干扰,应被过滤
    rings = det._feed_rings(frame)
    kinds = sorted(r.kind for r in rings)
    assert kinds == ["death", "kill"], rings


def test_count_alive_components(settings):
    profile = _profile(settings)
    det = Detector(settings, profile)
    roi = Roi.from_list(profile["rois"]["scoreboard_ally"])
    frame = _blank_frame()
    crop_shape = roi.crop(frame).shape
    cells = np.full(crop_shape, 40, dtype=np.uint8)
    ch, cw = crop_shape[:2]
    # 画 3 个彩色头像块(高饱和度),留间隔
    aw = cw // 7
    for i in [0, 2, 4]:
        x0 = i * cw // 5 + 2
        cells[2:ch - 2, x0:x0 + aw] = (200, 60, 30)
    assert det._count_alive(cells) == 3
    assert det._count_alive(np.full(crop_shape, 40, dtype=np.uint8)) == 0


@pytest.fixture(scope="module")
def synthetic_video(settings) -> Path:
    """12s 1080p30 合成视频:t=4s 一杀,t=5.4s 二杀(第二行金框),t=9s 被杀。"""
    FIXTURES.mkdir(exist_ok=True)
    path = FIXTURES / "synthetic_kills.mp4"
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    try:
        for f in range(12 * FPS):
            t = f / FPS
            noise = rng.integers(0, 90, size=(H // 8, W // 8, 3), dtype=np.uint8)
            frame = cv2.resize(noise, (W, H), interpolation=cv2.INTER_NEAREST)
            # 金框脉冲:每秒亮 0.7s 灭 0.3s(模拟真实的呼吸发光,考验去抖)
            pulse = (t % 1.0) < 0.7
            if 4.0 <= t < 10.0 and (pulse or t < 4.5):      # 第一杀的行
                _draw_ring(frame, right_frac=0.87, y_frac=0.065,
                           row_color=(90, 120, 90), label="AAA")
            if 5.4 <= t < 11.0 and (pulse or t < 5.9):      # 第二杀的行(叠在下方,
                # 与上一行同 x 对齐:同一击杀者的连杀行等宽,金框会粘连,考验切分)
                _draw_ring(frame, right_frac=0.87, y_frac=0.10,
                           row_color=(120, 90, 110), label="BBB")
            if 9.0 <= t < 12.0 and (pulse or t < 9.5):      # 我被击杀的行(行尾金框)
                _draw_ring(frame, right_frac=0.97, y_frac=0.135,
                           row_color=(80, 100, 130), label="CCC")
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_detect_synthetic_video(settings, synthetic_video):
    det = Detector(settings, _profile(settings))
    result = det.detect(synthetic_video, "sources/synthetic_kills.mp4")
    kills = [e for e in result.events if e.type == "kill"]
    deaths = [e for e in result.events if e.type == "death"]
    assert len(kills) == 2, f"应检出 2 次击杀: {[(e.type, e.t) for e in result.events]}"
    assert len(deaths) == 1, f"应检出 1 次被杀: {[(e.type, e.t) for e in result.events]}"
    # 精定位:全帧率精度
    assert abs(kills[0].t - 4.0) < 0.2
    assert abs(kills[1].t - 5.4) < 0.2
    assert abs(deaths[0].t - 9.0) < 0.2
    assert kills[0].confidence == 0.9
    assert result.video_meta.fps == pytest.approx(FPS, abs=0.1)
