"""align:贪心放置片段,切换点吸附拍点 → edl.json(规格 §5.6,D3)。

默认 cut_on_beat:按 EDL 顺序(打分降序)放置,每个片段的 timeline_start_t 吸附到
>= 当前游标的最近拍点;吸附产生的空隙由上一片段延长 out_t 填补(渲染是顺序拼接,
时间线上不允许空洞;延长量来自源素材 out_t 之后的富余画面)。

可选 anchor_align(默认关):第一个击杀帧对齐拍点,反推 in_t,片段间无缝拼接。

总时长超出 target_duration_s:先把低分片段的 pre_kill 收缩到 pre_kill_min,
仍超则从最低分开始丢弃(至少保留 1 个)。

place_clips / fit_clips / clips_from_cards 同时服务于 P0 的 build_edl 与
P1 agent DSL(改单后重吸附)。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from ..schemas.models import (
    BeatsFile, ClipAudio, ClipEffects, EdlFile, EventsFile, GlobalEffects,
    RenderSettings, ScorecardsFile, SnapInfo, TimelineEntry,
)

EPS = 1e-3


@dataclass
class Clip:
    """放置算法的输入单元(与 EDL entry 一一对应的轻量盒子)。"""
    clip_id: str
    source: str
    in_t: float
    out_t: float
    anchors: list[float]
    score: float
    source_duration: float
    effects: ClipEffects = field(default_factory=ClipEffects)
    audio: ClipAudio | None = None


def clips_from_cards(cards: ScorecardsFile, events: EventsFile,
                     only_selected: bool = True) -> list[Clip]:
    src_durations = {s.source: s.video_meta.duration_s for s in events.sources}
    return [
        Clip(clip_id=c.clip_id, source=c.source, in_t=c.span.start_t,
             out_t=c.span.end_t, anchors=list(c.anchor_ts), score=c.score.total,
             source_duration=src_durations.get(c.source, c.span.end_t))
        for c in cards.clips if c.selected or not only_selected
    ]


def cap_clip_length(c: Clip, settings: dict) -> None:
    """超长片段(马拉松击杀簇)裁到杀数最密的窗口,保住多杀精华。"""
    clip_cfg = settings["semantic"]["clip"]
    max_len = float(clip_cfg.get("max_clip_s", 0) or 0)
    if max_len <= 0 or c.out_t - c.in_t <= max_len or not c.anchors:
        return
    pre = min(float(clip_cfg["pre_kill_s"]), c.anchors[0] - c.in_t)
    post = min(float(clip_cfg["post_kill_s"]), c.out_t - c.anchors[-1])
    inner = max(1.0, max_len - pre - post)
    # 滑窗:选覆盖击杀数最多(平分取更短)的锚点窗口
    best = (1, 0.0, 0, 0)  # (count, -span, i, j)
    j = 0
    for i in range(len(c.anchors)):
        j = max(j, i)
        while j + 1 < len(c.anchors) and c.anchors[j + 1] - c.anchors[i] <= inner:
            j += 1
        cand = (j - i + 1, -(c.anchors[j] - c.anchors[i]), i, j)
        if cand > best:
            best = cand
    _, _, i, j = best
    c.in_t = round(max(c.in_t, c.anchors[i] - pre), 3)
    c.out_t = round(min(c.out_t, c.anchors[j] + post), 3)
    c.anchors = [a for a in c.anchors if c.in_t <= a <= c.out_t]


def fit_clips(clips: list[Clip], target: float, settings: dict,
              warnings: list[str]) -> list[Clip]:
    """封顶超长片段 → 收缩 pre_kill → 丢低分片段,直到总长 <= target(至少保留 1 个)。"""
    clip_cfg = settings["semantic"]["clip"]
    pre_min = float(clip_cfg["pre_kill_min_s"])
    for c in clips:
        cap_clip_length(c, settings)
    total = sum(c.out_t - c.in_t for c in clips)
    if total <= target:
        return clips
    for c in sorted(clips, key=lambda c: c.score):
        if total <= target:
            break
        if not c.anchors:
            continue
        new_in = min(c.anchors[0] - pre_min, c.out_t - EPS)
        if new_in > c.in_t:
            total -= new_in - c.in_t
            c.in_t = round(new_in, 3)
    kept = sorted(clips, key=lambda c: c.score, reverse=True)
    while len(kept) > 1 and sum(c.out_t - c.in_t for c in kept) > target:
        dropped = kept.pop()
        warnings.append(f"超出目标时长,丢弃低分片段 {dropped.clip_id}"
                        f"(score={dropped.score})")
    kept_ids = {c.clip_id for c in kept}
    return [c for c in clips if c.clip_id in kept_ids]


def _next_beat(beats: list[float], t: float) -> float | None:
    i = bisect.bisect_left(beats, t - EPS)
    return beats[i] if i < len(beats) else None


def place_clips(clips: list[Clip], beats_t: list[float], settings: dict,
                anchor_align: bool = False) -> tuple[list[TimelineEntry], list[str]]:
    """按顺序放置片段,切点吸附拍点;返回时间线(无空洞)与警告。"""
    warnings: list[str] = []
    align_cfg = settings["align"]
    max_shift = float(align_cfg["max_snap_shift_s"])
    fill_gap = bool(align_cfg["fill_gap_by_extending_out"])
    audio_cfg = settings["audio"]

    entries: list[TimelineEntry] = []
    cursor = 0.0
    for c in clips:
        snap = SnapInfo(mode="none", cut_beat_t=None)
        if anchor_align and c.anchors:
            # 高级模式:反推 in_t 使第一个击杀帧落在拍点,片段间无缝拼接。
            # 拍点从 [游标+pre_min, 游标+默认 pre] 区间内取最接近默认留白的,
            # 区间内没有则向后取第一个(留白加长,受 max_shift 限制)。
            pre_default = max(c.anchors[0] - c.in_t, 0.0)
            pre_min = min(float(settings["semantic"]["clip"]["pre_kill_min_s"]),
                          pre_default)
            i = bisect.bisect_right(beats_t, cursor + pre_default + EPS) - 1
            if i >= 0 and beats_t[i] >= cursor + pre_min - EPS:
                beat = beats_t[i]
            else:
                beat = _next_beat(beats_t, cursor + pre_min)
            if beat is not None and beat - cursor <= pre_default + max_shift:
                new_in = c.anchors[0] - (beat - cursor)
                if new_in >= 0:
                    c.in_t = round(new_in, 3)
                    snap = SnapInfo(mode="anchor_align", cut_beat_t=round(beat, 4))
            if snap.mode != "anchor_align":
                warnings.append(f"{c.clip_id}: anchor_align 找不到合适拍点,保持原 in_t。")
            start = cursor
        elif not entries:
            start = 0.0  # 第一段从时间线 0 开始,音乐同起点
        else:
            beat = _next_beat(beats_t, cursor)
            if beat is None or beat - cursor > max_shift:
                warnings.append(f"{c.clip_id}: 游标 {cursor:.2f}s 附近无可用拍点,不吸附。")
                start = cursor
            else:
                gap = beat - cursor
                prev = entries[-1]
                prev_clip = clips[len(entries) - 1]
                capacity = max(0.0, prev_clip.source_duration - prev.out_t)
                if fill_gap and gap > EPS:
                    ext = min(gap, capacity)
                    prev.out_t = round(prev.out_t + ext, 3)
                    cursor += ext
                if abs(beat - cursor) <= 0.05:  # 验收:切换点与拍点偏差 <= 50ms
                    start = cursor
                    snap = SnapInfo(mode="cut_on_beat", cut_beat_t=round(beat, 4))
                else:
                    warnings.append(
                        f"{c.clip_id}: 上一片段素材不足以延长填缝(差 {beat - cursor:.2f}s),"
                        f"切点不在拍点上。")
                    start = cursor
        entries.append(TimelineEntry(
            clip_id=c.clip_id, source=c.source,
            in_t=round(c.in_t, 3), out_t=round(c.out_t, 3),
            timeline_start_t=round(start, 4), snap=snap,
            effects=c.effects,
            audio=c.audio or ClipAudio(game_volume=float(audio_cfg["game_volume"])),
        ))
        cursor = start + (c.out_t - c.in_t)
    return entries, warnings


def build_edl(cards: ScorecardsFile, beats: BeatsFile, events: EventsFile,
              settings: dict, target_duration_s: float,
              anchor_align: bool = False) -> tuple[EdlFile, list[str]]:
    warnings: list[str] = []
    clips = clips_from_cards(cards, events)
    if not clips:
        raise ValueError("没有 selected 的片段,请检查 scorecards.json 或调低 scorer.min_score。")
    # 成片不能比音乐长:超出部分没有拍点可吸附,BGM 也会提前收尾
    if beats.beats_t and target_duration_s > beats.beats_t[-1]:
        warnings.append(f"目标时长 {target_duration_s:.0f}s 超过音乐长度,"
                        f"收紧到最后一个拍点 {beats.beats_t[-1]:.1f}s。")
        target_duration_s = beats.beats_t[-1]
    clips = fit_clips(clips, target_duration_s, settings, warnings)
    entries, place_warnings = place_clips(clips, beats.beats_t, settings,
                                          anchor_align=anchor_align)
    warnings += place_warnings

    render_cfg = settings["render"]
    edl = EdlFile(
        music=beats.music,
        target_duration_s=target_duration_s,
        global_effects=GlobalEffects(frame_drop=False),
        timeline=entries,
        render=RenderSettings(codec=render_cfg["codec"],
                              crf_equivalent=int(render_cfg["crf_equivalent"])),
    )
    return edl, warnings
