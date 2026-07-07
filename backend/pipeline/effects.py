"""per-clip 特效 → ffmpeg 滤镜链片段(规格 §5.7,默认全关)。

- frame_drop(抽帧顿挫):select 按比例丢帧 + fps 补帧,时长不变。
- speed_ramp(击杀慢放):击杀帧前后局部 0.5x,通过 trim×3 + concat 实现;
  注意慢放会使片段实际时长增加 (pre+post)·(1/factor-1),该增量由 clip_extra_duration
  返回,渲染与对齐方需知晓(规格 §9:特效与卡点叠加的时长校验)。
"""
from __future__ import annotations

from ..schemas.models import ClipEffects, TimelineEntry


def clip_extra_duration(entry: TimelineEntry, global_frame_drop: bool) -> float:
    """该片段应用特效后相对 (out_t - in_t) 的时长增量。"""
    ramp = entry.effects.speed_ramp
    if ramp is None:
        return 0.0
    seg = _ramp_segment(entry)
    if seg is None:
        return 0.0
    a0, a1 = seg
    return (a1 - a0) * (1.0 / ramp.factor - 1.0)


def _ramp_segment(entry: TimelineEntry) -> tuple[float, float] | None:
    """慢放区间(源内时间),裁剪到 [in_t, out_t];退化为空返回 None。"""
    ramp = entry.effects.speed_ramp
    a0 = max(entry.in_t, ramp.anchor_t - ramp.pre_s)
    a1 = min(entry.out_t, ramp.anchor_t + ramp.post_s)
    return (a0, a1) if a1 - a0 > 1e-3 else None


def _frame_drop_filter(strength: float, fps: float) -> str:
    """丢帧比例 strength → 保留每 K 帧中的 1 帧,再 fps 补帧保持时长。"""
    keep_every = max(2, round(1.0 / max(1e-6, 1.0 - min(strength, 0.9))))
    return f"select='not(mod(n\\,{keep_every}))',fps={fps:.6f}"


def build_video_chain(entry: TimelineEntry, input_label: str, out_label: str,
                      fps: float, scale: str, global_frame_drop: bool) -> str:
    """单片段视频滤镜链:trim → 特效 → scale/fps 归一。返回 filter_complex 片段。"""
    eff: ClipEffects = entry.effects
    base = (f"trim=start={entry.in_t:.3f}:end={entry.out_t:.3f},"
            f"setpts=PTS-STARTPTS")
    post = f"{scale},fps={fps:.6f},setsar=1,format=yuv420p"

    ramp = eff.speed_ramp
    seg = _ramp_segment(entry) if ramp else None
    if seg is not None:
        a0, a1 = seg
        slow = 1.0 / ramp.factor
        parts = []
        labels = []
        pieces = [(entry.in_t, a0, 1.0), (a0, a1, slow), (a1, entry.out_t, 1.0)]
        idx = 0
        for (s, e, factor) in pieces:
            if e - s <= 1e-3:
                continue
            lbl = f"{out_label}_r{idx}"
            parts.append(
                f"[{input_label}]trim=start={s:.3f}:end={e:.3f},"
                f"setpts=(PTS-STARTPTS)*{factor:.4f}[{lbl}]")
            labels.append(lbl)
            idx += 1
        chain = ";".join(parts)
        chain += f";{''.join(f'[{l}]' for l in labels)}concat=n={len(labels)}:v=1:a=0[{out_label}_rc]"
        mid = f"[{out_label}_rc]"
        filters = []
    else:
        chain = ""
        mid = f"[{input_label}]"
        filters = [base]

    if eff.frame_drop or global_frame_drop:
        filters.append(_frame_drop_filter(eff.frame_drop_strength, fps))
    filters.append(post)
    tail = f"{mid}{','.join(filters)}[{out_label}]"
    return f"{chain};{tail}" if chain else tail
