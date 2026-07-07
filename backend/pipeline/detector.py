"""L1 检测器 v2:击杀信息流金框检测(规则 CV),输出原子事件流。

方案(2026-07 依据用户真实素材重设计,见 DECISIONS DX35):
- 主信号:右上击杀信息流中,"我"参与的行头像带金色高亮空心边框。
  金框是空心环(fill 低、bbox 中心无命中),可与金发头像/金色皮肤区分;
  金框右缘位于行尾(帧宽 feed_victim_x_frac 之外)= 我被击杀,否则 = 我的击杀。
  该信号无需任何模板即可工作;多杀 = 多行金框,天然支持。
- 主扫描:ffmpeg rawvideo 管道按 sample_fps(默认 6fps)取帧;金框计数上升 → 事件。
- 精定位:cv2.VideoCapture 在 ±refine_window_s 邻域内全帧率找计数首次到达的帧。
- 存活状态:顶部记分区 ROI 内数高饱和度头像连通块(阵亡头像灰化/消失)。
- 回合结束:胜利/失败横幅模板匹配(可选,缺模板则不产出并告警)。
- 爆头:信息流新行内模板匹配爆头图标(可选)。
- flick 特征:每个 kill 取击杀前帧对,屏幕中心稀疏光流估计视角角速度(度/秒)。
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


def _template_score(roi_bgr: np.ndarray, template: np.ndarray) -> float:
    th, tw = template.shape[:2]
    rh, rw = roi_bgr.shape[:2]
    if rh < th or rw < tw or th == 0 or tw == 0:
        return 0.0
    res = cv2.matchTemplate(roi_bgr, template, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


@dataclass(frozen=True)
class FeedRing:
    """信息流中的一个金色高亮框(帧坐标 bbox)。"""
    kind: str          # "kill" | "death"
    x: int
    y: int
    w: int
    h: int


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
        if not self.templates.has("round_won", "round_lost"):
            self.warnings.append("缺少回合结束横幅模板(可选),round_end 事件不产出;"
                                 "可用 `vmontage calibrate <帧> --kind round-won/round-lost` 生成。")
        if not self.templates.has("headshot_icon"):
            self.warnings.append("缺少爆头图标模板(可选),headshot 一律为 False;"
                                 "flick 判定因此不可用。")

    # ---------------------------------------------------------- 金框识别
    def _feed_rings(self, frame: np.ndarray) -> list[FeedRing]:
        roi = self.rois["kill_feed"]
        crop = roi.crop(frame)
        if crop.size == 0:
            return []
        fh, fw = frame.shape[:2]
        hsv_cfg = self.cfg["feed_highlight_hsv"]
        ring_cfg = self.cfg["feed_ring"]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_cfg["lower"]), np.array(hsv_cfg["upper"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        rings: list[FeedRing] = []
        x_off = int(roi.x * fw)
        y_off = int(roi.y * fh)
        min_h = ring_cfg["min_h_frac"] * fh
        max_h = ring_cfg["max_h_frac"] * fh

        def accept(x: int, y: int, w: int, h: int) -> bool:
            sub = mask[y:y + h, x:x + w]
            if sub.size == 0 or sub.mean() / 255 > float(ring_cfg["max_fill"]):
                return False
            cx0, cy0 = x + w // 4, y + h // 4
            center = mask[cy0:cy0 + max(1, h // 2), cx0:cx0 + max(1, w // 2)]
            return not (center.size and
                        center.mean() / 255 > float(ring_cfg["max_center_density"]))

        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            if not (ring_cfg["min_w_frac"] * fw <= w <= ring_cfg["max_w_frac"] * fw):
                continue
            # 连杀时相邻行金框只差 1-3px,闭合后会粘成一个 2-3 倍高的块:按行高切开逐段验环
            typical_h = (min_h + max_h) / 2
            n_rows = max(1, round(h / typical_h))
            if h < min_h or h / n_rows < min_h * 0.6 or h / n_rows > max_h:
                continue
            sub_h = h // n_rows
            for j in range(n_rows):
                sy = y + j * sub_h
                if not accept(x, sy, w, sub_h):
                    continue
                right_frac = (x_off + x + w) / fw
                kind = ("death" if right_frac >= float(self.cfg["feed_victim_x_frac"])
                        else "kill")
                rings.append(FeedRing(kind=kind, x=x_off + x, y=y_off + sy,
                                      w=w, h=sub_h))
        return rings

    def _row_strip(self, frame: np.ndarray, ring: FeedRing,
                   tall: bool = False) -> np.ndarray:
        """金框所在行的内容指纹:行横条(feed 左缘→帧右缘)灰度缩略图。

        tall=True 返回上下各加半行的加高探测条(匹配时供指纹垂直滑动,
        容忍行的像素级位移)。
        """
        fh, fw = frame.shape[:2]
        cfg = self.cfg["feed_row_track"]
        sw, sh = int(cfg["strip_w"]), int(cfg["strip_h"])
        x0 = int(self.rois["kill_feed"].x * fw)
        pad = ring.h // 2 if tall else 0
        y0 = max(0, ring.y - pad)
        y1 = min(fh, ring.y + ring.h + pad)
        band = cv2.cvtColor(frame[y0:y1, x0:fw], cv2.COLOR_BGR2GRAY)
        return cv2.resize(band, (sw, sh * 2 if tall else sh))

    @staticmethod
    def _strip_corr(probe_tall: np.ndarray, strip: np.ndarray) -> float:
        """指纹在加高探测条内垂直滑动的最大归一化相关。"""
        res = cv2.matchTemplate(probe_tall, strip, cv2.TM_CCOEFF_NORMED)
        v = float(res.max())
        return 0.0 if np.isnan(v) else v

    # ---------------------------------------------------------- 主入口
    def detect(self, video: Path, rel_source: str) -> SourceEvents:
        meta = ffprobe_meta(video)
        vm = VideoMeta(width=meta["width"], height=meta["height"],
                       fps=meta["fps"], duration_s=meta["duration_s"])
        hits, alive_events, round_events = self._scan(video, vm)
        cap = cv2.VideoCapture(str(video))
        try:
            feed_events = self._refine_hits(cap, vm, hits)
            for ev in feed_events:
                if isinstance(ev, KillEvent):
                    ev.pre_kill_angular_velocity_deg_s = self._angular_velocity(cap, vm, ev.t)
        finally:
            cap.release()
        events: list[Event] = sorted(
            [*feed_events, *alive_events, *round_events], key=lambda e: e.t)
        return SourceEvents(source=rel_source, video_meta=vm, events=events)

    # ---------------------------------------------------------- 采样扫描
    def _scan(self, video: Path, vm: VideoMeta):
        cfg = self.cfg
        sample_fps = float(cfg["sample_fps"])

        # 去抖(DX36):金框高亮与信息流行同寿命(约 5-6s,脉冲发光);按 TTL 记账,
        # 窗口内计数瞬时回落不算行消失,计数超过在账行数才判定新事件。
        # hits: (t, kind, 该行指纹, 当时同类金框数),精定位时按指纹+计数找行的诞生帧。
        hits: list[tuple[float, str, np.ndarray, int]] = []
        alive_events: list[AliveStateEvent] = []
        round_events: list[RoundEndEvent] = []

        highlight_s = float(cfg["feed_highlight_s"])
        active: dict[str, list[float]] = {"kill": [], "death": []}
        last_alive: Optional[tuple[int, int]] = None
        last_alive_t = -1e9
        prev_round_present = False
        check_round = self.templates.has("round_won", "round_lost")
        roi_ally = self.rois["scoreboard_ally"]
        roi_enemy = self.rois["scoreboard_enemy"]
        roi_round = self.rois["round_end_banner"]

        for t, frame in iter_sampled_frames(video, sample_fps, vm.width, vm.height):
            rings = self._feed_rings(frame)
            counts = {"kill": [r for r in rings if r.kind == "kill"],
                      "death": [r for r in rings if r.kind == "death"]}
            for kind, kind_rings in counts.items():
                active[kind] = [t0 for t0 in active[kind] if t - t0 < highlight_s]
                extra = len(kind_rings) - len(active[kind])
                if extra <= 0:
                    continue
                # 新行出现在信息流下方,取 y 最大的 extra 个作为候选
                for ring in sorted(kind_rings, key=lambda r: r.y)[-extra:]:
                    hits.append((t, kind, self._row_strip(frame, ring),
                                 len(kind_rings)))
                    active[kind].append(t)

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
        return hits, alive_events, round_events

    def _count_alive(self, roi_bgr: np.ndarray) -> int:
        """记分区一侧:数高饱和度头像连通块(阵亡头像灰化/消失)。"""
        if roi_bgr.size == 0:
            return 0
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] >= int(self.cfg["alive_saturation_threshold"])) &
                (hsv[:, :, 2] >= int(self.cfg["alive_value_threshold"]))).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        min_area = float(self.cfg["alive_min_area_frac"]) * roi_bgr.shape[0] * roi_bgr.shape[1]
        return sum(1 for i in range(1, n) if stats[i][4] >= min_area)

    # ---------------------------------------------------------- 精定位
    def _refine_hits(self, cap: cv2.VideoCapture, vm: VideoMeta,
                     hits: list[tuple[float, str, "np.ndarray", int]]) -> list[Event]:
        cfg = self.cfg
        window = float(cfg["refine_window_s"])
        min_gap = float(cfg["kill_min_gap_s"])
        match_thr = float(cfg["feed_row_track"]["refine_match"])
        events: list[Event] = []
        last_t = {"kill": -1e9, "death": -1e9}

        for t_hit, kind, strip, count_at_hit in hits:
            f_start = max(0, int((t_hit - window) * vm.fps))
            f_end = int((t_hit + window) * vm.fps)
            precise_frame, precise_img, new_ring = None, None, None
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
            for f in range(f_start, f_end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                rings_f = [r for r in self._feed_rings(frame) if r.kind == kind]
                match = None
                if len(rings_f) >= count_at_hit:  # 计数到位 + 指纹匹配双条件
                    match = next(
                        (r for r in rings_f
                         if self._strip_corr(self._row_strip(frame, r, tall=True),
                                             strip) >= match_thr),
                        None)
                if match is not None:
                    precise_frame, precise_img, new_ring = f, frame, match
                    break
            if precise_frame is None:
                precise_frame = int(t_hit * vm.fps)
                precise_img = _read_frame_at(cap, precise_frame)
                if precise_img is None:
                    continue
            t = precise_frame / vm.fps
            if t - last_t[kind] < min_gap:
                continue
            last_t[kind] = t

            if kind == "death":
                events.append(DeathEvent(frame=precise_frame, t=round(t, 3),
                                         confidence=0.9))
                continue
            headshot = self._headshot_in_row(precise_img, new_ring)
            events.append(KillEvent(frame=precise_frame, t=round(t, 3),
                                    headshot=headshot, confidence=0.9))
        return events

    def _headshot_in_row(self, frame: np.ndarray, ring: Optional[FeedRing]) -> bool:
        """在新增信息流行的横条区域内匹配爆头图标模板(可选能力)。"""
        tpl = self.templates.get("headshot_icon")
        if tpl is None or ring is None:
            return False
        fh, fw = frame.shape[:2]
        roi = self.rois["kill_feed"]
        y0 = max(0, ring.y - ring.h // 3)
        y1 = min(fh, ring.y + ring.h + ring.h // 3)
        x0 = int(roi.x * fw)
        row = frame[y0:y1, x0:fw]
        return _template_score(row, tpl) >= float(self.cfg["headshot_match_threshold"])

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
