"""L1 检测器:规则 CV(HSV + 模板匹配 + 稀疏光流),输出原子事件流(规格 §5.2)。

策略:
- 主扫描:ffmpeg rawvideo 管道按 sample_fps(默认 6fps)取帧,HSV 检测击杀横幅出现、
  存活状态变化、回合结束横幅。
- 精定位:横幅上升沿检出后,用 cv2.VideoCapture 在 ±refine_window_s 邻域内全帧率
  找到横幅起始帧。
- 方向区分:kill/death 模板匹配;模板缺失时降级为 HSV-only(全部记为 kill,
  confidence 压到 0.6)并告警——运行 `vmontage calibrate` 生成模板后恢复。
- 多杀再触发:横幅持续期间 HSV 命中面积显著跳升(新横幅行叠加)视为又一次击杀。
- flick 特征:每个 kill 取击杀前 pre_window 的帧对,屏幕中心区域稀疏光流估计
  视角角速度(度/秒)写入事件。
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from ..schemas.models import (
    AliveStateEvent, DeathEvent, Event, KillEvent, RoundEndEvent, SourceEvents, VideoMeta,
)
from .ffmpeg_utils import ffprobe_meta

ASSETS_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates"


# ------------------------------------------------------------------ 基础工具

@dataclass(frozen=True)
class Roi:
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_list(cls, v: list[float]) -> "Roi":
        return cls(*v)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x0, y0 = int(self.x * fw), int(self.y * fh)
        x1, y1 = int((self.x + self.w) * fw), int((self.y + self.h) * fh)
        return frame[y0:y1, x0:x1]


class TemplateSet:
    """assets/templates(全局)与 project/templates(项目级,优先)下的模板资产。"""

    def __init__(self, names: dict[str, str], extra_dir: Optional[Path] = None):
        self._cache: dict[str, Optional[np.ndarray]] = {}
        for key, filename in names.items():
            img = None
            for base in filter(None, [extra_dir, ASSETS_TEMPLATES]):
                p = Path(base) / filename
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is not None:
                        break
            self._cache[key] = img

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._cache.get(key)

    def has(self, *keys: str) -> bool:
        return all(self._cache.get(k) is not None for k in keys)


def _hsv_banner_ratio(roi_bgr: np.ndarray, hsv_cfg: dict) -> float:
    """ROI 内命中横幅 HSV 阈值的像素占比(红色跨 0 度,双段合并)。"""
    if roi_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_cfg["lower"]), np.array(hsv_cfg["upper"]))
    if "lower2" in hsv_cfg:
        mask |= cv2.inRange(hsv, np.array(hsv_cfg["lower2"]), np.array(hsv_cfg["upper2"]))
    return float(np.count_nonzero(mask)) / mask.size


def _template_score(roi_bgr: np.ndarray, template: np.ndarray) -> float:
    th, tw = template.shape[:2]
    rh, rw = roi_bgr.shape[:2]
    if rh < th or rw < tw:
        # 模板比 ROI 大(分辨率不符)按不匹配处理
        return 0.0
    res = cv2.matchTemplate(roi_bgr, template, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


# ------------------------------------------------------------------ 帧采样

def iter_sampled_frames(video: Path, sample_fps: float,
                        width: int, height: int) -> Iterator[tuple[float, np.ndarray]]:
    """ffmpeg 管道按 sample_fps 输出 BGR 帧,yield (t, frame)。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg")
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error",
         "-i", str(video), "-vf", f"fps={sample_fps}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=width * height * 3 * 4,
    )
    frame_size = width * height * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            # fps 滤镜输出帧 i 对应源时间约 i/sample_fps(取窗口起点)
            yield idx / sample_fps, frame
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


# ------------------------------------------------------------------ 检测器

class Detector:
    def __init__(self, settings: dict, roi_profile: dict,
                 project_templates_dir: Optional[Path] = None):
        self.cfg = settings["detector"]
        self.rois = {k: Roi.from_list(v) for k, v in roi_profile["rois"].items()}
        self.templates = TemplateSet(roi_profile.get("templates", {}), project_templates_dir)
        self.warnings: list[str] = []
        if not self.templates.has("kill_banner", "death_banner"):
            self.warnings.append(
                "缺少 kill/death 横幅模板 → HSV-only 降级模式:所有横幅记为 kill,"
                "confidence=0.6。运行 `vmontage calibrate <样例帧>` 生成模板。")
        if not self.templates.has("round_won", "round_lost"):
            self.warnings.append("缺少回合结束横幅模板,round_end 事件不产出。")

    # ---------------------------------------------------------- 主入口
    def detect(self, video: Path, rel_source: str) -> SourceEvents:
        meta = ffprobe_meta(video)
        vm = VideoMeta(width=meta["width"], height=meta["height"],
                       fps=meta["fps"], duration_s=meta["duration_s"])
        banner_hits, alive_events, round_events = self._scan(video, vm)
        cap = cv2.VideoCapture(str(video))
        try:
            kill_events = self._refine_banners(cap, vm, banner_hits)
            for ev in kill_events:
                if isinstance(ev, KillEvent):
                    ev.pre_kill_angular_velocity_deg_s = self._angular_velocity(cap, vm, ev.t)
        finally:
            cap.release()
        events: list[Event] = sorted(
            [*kill_events, *alive_events, *round_events], key=lambda e: e.t)
        return SourceEvents(source=rel_source, video_meta=vm, events=events)

    # ---------------------------------------------------------- 采样扫描
    def _scan(self, video: Path, vm: VideoMeta):
        cfg = self.cfg
        hsv_cfg = cfg["banner_hsv"]
        sample_fps = float(cfg["sample_fps"])
        min_ratio = float(hsv_cfg["min_area_ratio"])

        # (t, 精定位用的面积阈值):上升沿用基础阈值;面积跳升(多杀)用抬高阈值,
        # 否则回溯会撞上仍在显示的上一条横幅
        banner_hits: list[tuple[float, float]] = []
        alive_events: list[AliveStateEvent] = []
        round_events: list[RoundEndEvent] = []

        prev_present = False
        rolling_min_ratio = 0.0
        last_alive: Optional[tuple[int, int]] = None
        last_alive_t = -1e9
        prev_round_present = False

        roi_banner = self.rois["kill_banner"]
        roi_ally = self.rois["scoreboard_ally"]
        roi_enemy = self.rois["scoreboard_enemy"]
        roi_round = self.rois["round_end_banner"]
        check_round = self.templates.has("round_won", "round_lost")

        for t, frame in iter_sampled_frames(video, sample_fps, vm.width, vm.height):
            ratio = _hsv_banner_ratio(roi_banner.crop(frame), hsv_cfg)
            present = ratio >= min_ratio
            if present and not prev_present:
                banner_hits.append((t, min_ratio))
                rolling_min_ratio = ratio
            elif present:
                # 多杀:横幅持续期间面积显著跳升 → 新横幅行叠加,再触发一次
                rolling_min_ratio = min(rolling_min_ratio, ratio)
                if rolling_min_ratio > 0 and ratio >= rolling_min_ratio * 1.6:
                    banner_hits.append((t, rolling_min_ratio * 1.3))
                    rolling_min_ratio = ratio
            prev_present = present

            ally = self._count_alive(roi_ally.crop(frame))
            enemy = self._count_alive(roi_enemy.crop(frame))
            if (ally, enemy) != last_alive and \
                    t - last_alive_t >= float(cfg["alive_min_change_interval_s"]):
                alive_events.append(AliveStateEvent(
                    frame=int(t * vm.fps), t=round(t, 3),
                    ally_alive=ally, enemy_alive=enemy))
                last_alive = (ally, enemy)
                last_alive_t = t

            if check_round:
                crop = roi_round.crop(frame)
                won_s = _template_score(crop, self.templates.get("round_won"))
                lost_s = _template_score(crop, self.templates.get("round_lost"))
                thr = float(cfg["template_match_threshold"])
                round_present = max(won_s, lost_s) >= thr
                if round_present and not prev_round_present:
                    round_events.append(RoundEndEvent(
                        frame=int(t * vm.fps), t=round(t, 3),
                        won=won_s >= lost_s, confidence=round(max(won_s, lost_s), 3)))
                prev_round_present = round_present
        return banner_hits, alive_events, round_events

    def _count_alive(self, roi_bgr: np.ndarray) -> int:
        """记分区一侧头像格:按列均分,平均饱和度高于阈值的格子数=存活数。"""
        cells = int(self.cfg["alive_cells"])
        if roi_bgr.size == 0:
            return 0
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        w = sat.shape[1]
        alive = 0
        for i in range(cells):
            cell = sat[:, i * w // cells:(i + 1) * w // cells]
            if cell.size and float(cell.mean()) >= float(self.cfg["alive_saturation_threshold"]):
                alive += 1
        return alive

    # ---------------------------------------------------------- 精定位 + 方向
    def _refine_banners(self, cap: cv2.VideoCapture, vm: VideoMeta,
                        banner_hits: list[tuple[float, float]]) -> list[Event]:
        cfg = self.cfg
        hsv_cfg = cfg["banner_hsv"]
        window = float(cfg["refine_window_s"])
        min_gap = float(cfg["kill_min_gap_s"])
        roi_banner = self.rois["kill_banner"]
        has_dir_templates = self.templates.has("kill_banner", "death_banner")
        thr = float(cfg["template_match_threshold"])

        events: list[Event] = []
        last_t = -1e9
        for t_hit, ratio_threshold in banner_hits:
            # ±window 内全帧率向前回溯,找第一个过阈帧(阈值随命中类型而定)
            f_start = max(0, int((t_hit - window) * vm.fps))
            f_end = int((t_hit + window) * vm.fps)
            precise_frame, precise_img = None, None
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
            for f in range(f_start, f_end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if _hsv_banner_ratio(roi_banner.crop(frame), hsv_cfg) >= ratio_threshold:
                    precise_frame, precise_img = f, frame
                    break
            if precise_frame is None:
                precise_frame = int(t_hit * vm.fps)
                precise_img = _read_frame_at(cap, precise_frame)
                if precise_img is None:
                    continue
            t = precise_frame / vm.fps
            if t - last_t < min_gap:
                continue
            last_t = t

            crop = roi_banner.crop(precise_img)
            if has_dir_templates:
                kill_s = _template_score(crop, self.templates.get("kill_banner"))
                death_s = _template_score(crop, self.templates.get("death_banner"))
                if max(kill_s, death_s) < thr:
                    continue  # HSV 误报,模板双双不认
                if death_s > kill_s:
                    events.append(DeathEvent(frame=precise_frame, t=round(t, 3),
                                             confidence=round(death_s, 3)))
                    continue
                confidence = round(kill_s, 3)
            else:
                confidence = 0.6  # HSV-only 降级
            headshot = False
            hs_tpl = self.templates.get("headshot_icon")
            if hs_tpl is not None:
                headshot = _template_score(crop, hs_tpl) >= float(cfg["headshot_match_threshold"])
            events.append(KillEvent(frame=precise_frame, t=round(t, 3),
                                    headshot=headshot, confidence=confidence))
        return events

    # ---------------------------------------------------------- flick 光流
    def _angular_velocity(self, cap: cv2.VideoCapture, vm: VideoMeta,
                          kill_t: float) -> Optional[float]:
        """击杀前帧对的稀疏光流 → 视角角速度(度/秒)。"""
        w0, w1 = self.cfg["flick"]["pre_window_s"]  # 例 [0.10, 0.30]
        f_a = int((kill_t - float(w1)) * vm.fps)
        f_b = int((kill_t - float(w0)) * vm.fps)
        if f_a < 0 or f_b <= f_a:
            return None
        img_a = _read_frame_at(cap, f_a)
        img_b = _read_frame_at(cap, f_b)
        if img_a is None or img_b is None:
            return None
        roi = self.rois["center_flow"]
        gray_a = cv2.cvtColor(roi.crop(img_a), cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(roi.crop(img_b), cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(
            gray_a, maxCorners=int(self.cfg["flick"]["max_features"]),
            qualityLevel=0.01, minDistance=8)
        if pts is None or len(pts) < 8:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, pts, None)
        good = status.reshape(-1) == 1
        if good.sum() < 8:
            return None
        disp = (nxt.reshape(-1, 2) - pts.reshape(-1, 2))[good]
        # 相机旋转 → 背景整体平移;取中位位移幅值抗前景干扰
        median_px = float(np.median(np.linalg.norm(disp, axis=1)))
        dt = (f_b - f_a) / vm.fps
        deg_per_px = float(self.cfg["flick"]["hfov_deg"]) / vm.width
        return round(median_px * deg_per_px / dt, 1)
