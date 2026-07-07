"""合成帧/合成视频上的 L1 检测器测试(无真实素材时的最低保障)。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config import load_roi_profile
from backend.pipeline.detector import Detector, Roi, _hsv_banner_ratio

FIXTURES = Path(__file__).parent / "fixtures"
W, H, FPS = 1920, 1080, 30


def _profile(settings):
    return load_roi_profile(W, H, settings)


def _blank_frame() -> np.ndarray:
    return np.full((H, W, 3), 40, dtype=np.uint8)  # 深灰背景


def _paint_roi(frame: np.ndarray, roi: Roi, color: tuple[int, int, int],
               inset: float = 0.2) -> None:
    x0 = int((roi.x + roi.w * inset) * W)
    x1 = int((roi.x + roi.w * (1 - inset)) * W)
    y0 = int((roi.y + roi.h * inset) * H)
    y1 = int((roi.y + roi.h * (1 - inset)) * H)
    frame[y0:y1, x0:x1] = color

BANNER_RED = (48, 40, 235)  # BGR,HSV 落入默认红色阈值


def test_hsv_ratio(settings):
    profile = _profile(settings)
    roi = Roi.from_list(profile["rois"]["kill_banner"])
    frame = _blank_frame()
    assert _hsv_banner_ratio(roi.crop(frame), settings["detector"]["banner_hsv"]) < 0.001
    _paint_roi(frame, roi, BANNER_RED)
    assert _hsv_banner_ratio(roi.crop(frame), settings["detector"]["banner_hsv"]) > 0.3


def test_count_alive(settings):
    profile = _profile(settings)
    det = Detector(settings, profile)
    roi = Roi.from_list(profile["rois"]["scoreboard_ally"])
    frame = _blank_frame()
    crop_shape = roi.crop(frame).shape
    cells = np.full(crop_shape, 40, dtype=np.uint8)
    w = crop_shape[1]
    # 5 格中 3 格涂高饱和度颜色(存活),2 格保持灰(阵亡)
    for i in [0, 2, 4]:
        cells[:, i * w // 5:(i + 1) * w // 5] = (200, 60, 30)
    assert det._count_alive(cells) == 3
    assert det._count_alive(np.full(crop_shape, 40, dtype=np.uint8)) == 0


@pytest.fixture(scope="module")
def synthetic_video(settings) -> Path:
    """12s 1080p30 合成视频:t=4s 与 t=5.4s 两次击杀横幅(第二次叠加行),背景带噪声。"""
    FIXTURES.mkdir(exist_ok=True)
    path = FIXTURES / "synthetic_kills.mp4"
    profile = load_roi_profile(W, H, settings)
    roi = Roi.from_list(profile["rois"]["kill_banner"])
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    try:
        for f in range(12 * FPS):
            t = f / FPS
            frame = _blank_frame()
            # 背景噪声块,给光流一些可跟踪纹理
            noise = rng.integers(0, 90, size=(H // 8, W // 8, 3), dtype=np.uint8)
            frame[:] = cv2.resize(noise, (W, H), interpolation=cv2.INTER_NEAREST)
            if 4.0 <= t < 8.0:      # 第一条横幅,持续 4s
                _paint_roi(frame, roi, BANNER_RED, inset=0.35)
            if 5.4 <= t < 8.0:      # 第二条叠加(面积跳升 → 多杀再触发)
                _paint_roi(frame, roi, BANNER_RED, inset=0.15)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_detect_synthetic_video(settings, synthetic_video):
    profile = _profile(settings)
    det = Detector(settings, profile)
    result = det.detect(synthetic_video, "sources/synthetic_kills.mp4")
    kills = [e for e in result.events if e.type == "kill"]
    assert len(kills) == 2, f"应检出 2 次击杀,实际 {[(e.type, e.t) for e in result.events]}"
    # 精定位:第一次击杀应在 4.0s 附近(全帧率精度)
    assert abs(kills[0].t - 4.0) < 0.2
    assert abs(kills[1].t - 5.4) < 0.4
    # HSV-only 降级模式的置信度
    assert kills[0].confidence == 0.6
    assert result.video_meta.fps == pytest.approx(FPS, abs=0.1)
