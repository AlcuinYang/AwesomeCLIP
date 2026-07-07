"""calibrate:从用户样例帧截取模板资产 + 输出 ROI 标注图(规格 §5.2,一等公民)。

用法(非交互):
  vmontage calibrate frame.png                # 输出 ROI 标注图核对坐标
  vmontage calibrate kill_frame.png --kind kill        # 截取击杀横幅模板
  vmontage calibrate death_frame.png --kind death
  vmontage calibrate hs_frame.png --kind headshot --sub 0.62,0.15,0.10,0.55
  vmontage calibrate won_frame.png --kind round-won
  vmontage calibrate lost_frame.png --kind round-lost

--sub x,y,w,h 为 ROI 内的相对子区域,用于截取更紧的模板(如爆头小图标)。
模板写入 assets/templates/,文件名来自 roi_*.yaml 的 templates 表。
"""
from __future__ import annotations

from pathlib import Path

import cv2

from ..config import load_roi_profile, load_settings
from .detector import ASSETS_TEMPLATES, Roi

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

KIND_TO_ROI = {
    "kill": ("kill_banner", "kill_banner"),
    "death": ("kill_banner", "death_banner"),
    "headshot": ("kill_banner", "headshot_icon"),
    "round-won": ("round_end_banner", "round_won"),
    "round-lost": ("round_end_banner", "round_lost"),
}


def extract_frame(video: Path, at_s: float) -> Path:
    """从录像抽一帧(无损 png)供校准;输出在视频旁边,便于反复使用。"""
    from .ffmpeg_utils import run_ffmpeg

    out = video.with_name(f"{video.stem}_t{at_s:g}.png")
    run_ffmpeg(["-ss", f"{at_s:.3f}", "-i", str(video), "-frames:v", "1", str(out)])
    return out


def calibrate(frame_path: Path, kind: str | None = None,
              sub: tuple[float, float, float, float] | None = None,
              project_dir: Path | None = None,
              at_s: float | None = None) -> Path:
    frame_path = Path(frame_path)
    if frame_path.suffix.lower() in VIDEO_EXTS:
        if at_s is None:
            raise ValueError("输入是视频,请用 --at <秒> 指定要抽取的画面时间点"
                             "(选一帧正显示击杀横幅/回合结算的画面)。")
        frame_path = extract_frame(frame_path, at_s)
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取帧图像: {frame_path}")
    h, w = img.shape[:2]
    settings = load_settings(project_dir)
    profile = load_roi_profile(w, h, settings, project_dir)

    if kind is None:
        # 标注全部 ROI 供人工核对
        vis = img.copy()
        for name, rect in profile["rois"].items():
            roi = Roi.from_list(rect)
            x0, y0 = int(roi.x * w), int(roi.y * h)
            x1, y1 = int((roi.x + roi.w) * w), int((roi.y + roi.h) * h)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, name, (x0, max(16, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = frame_path.with_name(frame_path.stem + "_roi_overview.png")
        cv2.imwrite(str(out), vis)
        return out

    if kind not in KIND_TO_ROI:
        raise ValueError(f"未知 kind: {kind}(可选 {', '.join(KIND_TO_ROI)})")
    roi_name, template_key = KIND_TO_ROI[kind]
    crop = Roi.from_list(profile["rois"][roi_name]).crop(img)
    if sub is not None:
        sx, sy, sw, sh = sub
        ch, cw = crop.shape[:2]
        crop = crop[int(sy * ch):int((sy + sh) * ch), int(sx * cw):int((sx + sw) * cw)]
    if crop.size == 0:
        raise ValueError("截取区域为空,请检查 ROI/--sub 参数。")
    ASSETS_TEMPLATES.mkdir(parents=True, exist_ok=True)
    out = ASSETS_TEMPLATES / profile["templates"][template_key]
    cv2.imwrite(str(out), crop)
    return out
