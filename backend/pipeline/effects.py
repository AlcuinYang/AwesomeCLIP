"""per-clip 特效 → ffmpeg 滤镜链片段。

- frame_drop(抽帧顿挫):select 按比例丢帧 + fps 补帧,时长不变。
- 变速:speed_spans(导演的间隙压缩,factor>1 快进)与 speed_ramp(击杀慢放,
  锚点式)统一为分段 trim+setpts+concat;时长变化由 clip_extra_duration 显式给出。
- 音频与视频分段严格对齐:变速段按配置静音(aevalsrc 补长)或 atempo 变速保留。
"""
from __future__ import annotations

from ..schemas.models import ClipEffects, SpeedSpan, TimelineEntry

MIN_PIECE_S = 0.05


def _spans(entry: TimelineEntry) -> list[SpeedSpan]:
    """合并 speed_spans 与遗留 speed_ramp,裁剪到 [in_t, out_t],按时间排序。"""
    spans = list(entry.effects.speed_spans)
    ramp = entry.effects.speed_ramp
    if ramp is not None:
        spans.append(SpeedSpan(from_t=ramp.anchor_t - ramp.pre_s,
                               to_t=ramp.anchor_t + ramp.post_s,
                               factor=ramp.factor))
    out = []
    for s in sorted(spans, key=lambda s: s.from_t):
        a = max(entry.in_t, s.from_t)
        b = min(entry.out_t, s.to_t)
        if b - a > MIN_PIECE_S and s.factor > 0 and abs(s.factor - 1.0) > 1e-6:
            if out and a < out[-1].to_t:  # 重叠跨度:后者从前者结束处开始
                a = out[-1].to_t
                if b - a <= MIN_PIECE_S:
                    continue
            out.append(SpeedSpan(from_t=a, to_t=b, factor=s.factor))
    return out


def pieces_of(entry: TimelineEntry) -> list[tuple[float, float, float]]:
    """片段的分段计划 [(源内起, 源内止, 播放倍速)],覆盖 [in_t, out_t] 无缝无重叠。"""
    pieces: list[tuple[float, float, float]] = []
    cursor = entry.in_t
    for s in _spans(entry):
        if s.from_t - cursor > MIN_PIECE_S:
            pieces.append((cursor, s.from_t, 1.0))
        pieces.append((s.from_t, s.to_t, s.factor))
        cursor = s.to_t
    if entry.out_t - cursor > MIN_PIECE_S:
        pieces.append((cursor, entry.out_t, 1.0))
    return pieces or [(entry.in_t, entry.out_t, 1.0)]


def rendered_duration(entry: TimelineEntry) -> float:
    """应用变速后的实际输出时长。"""
    return sum((e - s) / f for s, e, f in pieces_of(entry))


def clip_extra_duration(entry: TimelineEntry, global_frame_drop: bool) -> float:
    """相对 (out_t - in_t) 的时长增量(慢放为正,压缩为负)。"""
    return rendered_duration(entry) - (entry.out_t - entry.in_t)


def _frame_drop_filter(strength: float, fps: float) -> str:
    keep_every = max(2, round(1.0 / max(1e-6, 1.0 - min(strength, 0.9))))
    return f"select='not(mod(n\\,{keep_every}))',fps={fps:.6f}"


def build_video_chain(entry: TimelineEntry, input_label: str, out_label: str,
                      fps: float, scale: str, global_frame_drop: bool) -> str:
    """单片段视频滤镜链:分段 trim/变速 → concat → 特效 → 归一。"""
    eff: ClipEffects = entry.effects
    post = f"{scale},fps={fps:.6f},setsar=1,format=yuv420p"
    pieces = pieces_of(entry)

    if len(pieces) == 1 and pieces[0][2] == 1.0:
        chain = ""
        mid = f"[{input_label}]"
        filters = [f"trim=start={entry.in_t:.3f}:end={entry.out_t:.3f},"
                   f"setpts=PTS-STARTPTS"]
    else:
        parts, labels = [], []
        for i, (s, e, f) in enumerate(pieces):
            lbl = f"{out_label}_p{i}"
            parts.append(f"[{input_label}]trim=start={s:.3f}:end={e:.3f},"
                         f"setpts=(PTS-STARTPTS)/{f:.4f}[{lbl}]")
            labels.append(lbl)
        chain = ";".join(parts)
        chain += (f";{''.join(f'[{l}]' for l in labels)}"
                  f"concat=n={len(labels)}:v=1:a=0[{out_label}_pc]")
        mid = f"[{out_label}_pc]"
        filters = []

    if eff.frame_drop or global_frame_drop:
        filters.append(_frame_drop_filter(eff.frame_drop_strength, fps))
    filters.append(post)
    tail = f"{mid}{','.join(filters)}[{out_label}]"
    return f"{chain};{tail}" if chain else tail


def build_audio_chain(entry: TimelineEntry, input_label: str, out_label: str,
                      mute_speed_spans: bool) -> str:
    """单片段音频滤镜链,与视频分段严格对齐。

    变速段:mute_speed_spans=True 用等长静音(压缩段安静,主流做法);
    False 则 atempo 变速保留(ffmpeg atempo 支持 0.5~100)。
    """
    fmt = "aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo"
    pieces = pieces_of(entry)
    if len(pieces) == 1 and pieces[0][2] == 1.0:
        return (f"[{input_label}]atrim=start={entry.in_t:.3f}:end={entry.out_t:.3f},"
                f"asetpts=PTS-STARTPTS,{fmt},"
                f"volume={entry.audio.game_volume:.3f}[{out_label}]")
    parts, labels = [], []
    for i, (s, e, f) in enumerate(pieces):
        lbl = f"{out_label}_a{i}"
        out_dur = (e - s) / f
        if f != 1.0 and mute_speed_spans:
            parts.append(f"aevalsrc=0:d={out_dur:.3f}:s=48000,"
                         f"aformat=sample_rates=48000:channel_layouts=stereo[{lbl}]")
        elif f != 1.0:
            parts.append(f"[{input_label}]atrim=start={s:.3f}:end={e:.3f},"
                         f"asetpts=PTS-STARTPTS,atempo={min(max(f, 0.5), 100):.4f},"
                         f"{fmt}[{lbl}]")
        else:
            parts.append(f"[{input_label}]atrim=start={s:.3f}:end={e:.3f},"
                         f"asetpts=PTS-STARTPTS,{fmt}[{lbl}]")
        labels.append(lbl)
    chain = ";".join(parts)
    chain += (f";{''.join(f'[{l}]' for l in labels)}"
              f"concat=n={len(labels)}:v=0:a=1,"
              f"volume={entry.audio.game_volume:.3f}[{out_label}]")
    return chain
