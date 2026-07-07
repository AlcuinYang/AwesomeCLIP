"""render:EDL → ffmpeg filter_complex 单次调用出片(规格 §5.8)。

- 视频:per-clip trim/特效/归一(统一分辨率+帧率)→ concat。
- 音频:per-clip 游戏原声(atrim + volume,无音轨补静音)→ concat,与 BGM amix,
  BGM 尾部 afade 收尾。
- 编码:h264_nvenc,不可用回退 libx264 并警告;preview 走 720p + veryfast。
- 渲染只读 edl.json 与源文件:改 edl 后直接重渲,无需重新 detect(验收 #6)。
"""
from __future__ import annotations

from pathlib import Path

from ..schemas.models import EdlFile
from .effects import build_video_chain, clip_extra_duration
from .ffmpeg_utils import ffprobe_meta, pick_video_codec, run_ffmpeg
from .project import Project


def _target_geometry(edl: EdlFile, metas: dict[str, dict], preview: bool,
                     preview_height: int) -> tuple[int, int, float]:
    """输出宽高与归一帧率:source 模式取首片段源参数(全部片段缩放对齐)。"""
    first = metas[edl.timeline[0].source]
    w, h, fps = first["width"], first["height"], first["fps"]
    res = edl.render.resolution
    if preview:
        h_target = preview_height
    elif res == "source":
        h_target = h
    else:
        h_target = int(res.rstrip("p"))
    w_target = int(round(w * h_target / h / 2) * 2)
    return w_target, h_target - (h_target % 2), fps


def render(project: Project, edl: EdlFile, out_path: Path, settings: dict,
           preview: bool = False,
           on_progress=None) -> tuple[Path, list[str]]:
    if not edl.timeline:
        raise ValueError("EDL 时间线为空。")
    warnings: list[str] = []

    sources = list(dict.fromkeys(e.source for e in edl.timeline))  # 去重保序
    metas = {s: ffprobe_meta(project.resolve_media(s)) for s in sources}
    src_index = {s: i for i, s in enumerate(sources)}
    w, h, fps = _target_geometry(edl, metas, preview,
                                 int(settings["render"]["preview"]["height"]))
    scale = f"scale={w}:{h}:force_original_aspect_ratio=decrease," \
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"

    args: list[str] = []
    for s in sources:
        args += ["-i", str(project.resolve_media(s))]
    music_index = None
    if edl.music:
        music_index = len(sources)
        args += ["-i", str(project.resolve_media(edl.music))]

    total_dur = 0.0
    fc_parts: list[str] = []
    v_labels: list[str] = []
    a_labels: list[str] = []
    for k, entry in enumerate(edl.timeline):
        i = src_index[entry.source]
        vl = f"v{k}"
        fc_parts.append(build_video_chain(
            entry, f"{i}:v", vl, fps, scale,
            global_frame_drop=edl.global_effects.frame_drop))
        v_labels.append(vl)

        dur = entry.out_t - entry.in_t + clip_extra_duration(
            entry, edl.global_effects.frame_drop)
        total_dur += dur
        al = f"a{k}"
        if metas[entry.source]["has_audio"]:
            fc_parts.append(
                f"[{i}:a]atrim=start={entry.in_t:.3f}:end={entry.out_t:.3f},"
                f"asetpts=PTS-STARTPTS,"
                f"apad=whole_dur={dur:.3f},atrim=end={dur:.3f},"
                f"volume={entry.audio.game_volume:.3f}[{al}]")
        else:
            fc_parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                f"atrim=end={dur:.3f}[{al}]")
        a_labels.append(al)

    n = len(edl.timeline)
    fc_parts.append(f"{''.join(f'[{l}]' for l in v_labels)}concat=n={n}:v=1:a=0[vout]")
    fc_parts.append(
        f"{''.join(f'[{l}]' for l in a_labels)}"
        f"concat=n={n}:v=0:a=1,aresample=48000,pan=stereo|c0=c0|c1=c1[agame]")

    audio_cfg = settings["audio"]
    if music_index is not None:
        fade = float(audio_cfg["music_fade_out_s"])
        fc_parts.append(
            f"[{music_index}:a]atrim=end={total_dur:.3f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,pan=stereo|c0=c0|c1=c1,"
            f"volume={float(audio_cfg['music_volume']):.3f},"
            f"afade=t=out:st={max(0.0, total_dur - fade):.3f}:d={fade:.3f}[amus]")
        fc_parts.append("[agame][amus]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        fc_parts.append("[agame]anull[aout]")

    codec, codec_warnings = pick_video_codec(
        "libx264" if preview else edl.render.codec)
    warnings += codec_warnings
    if codec == "h264_nvenc":
        encode = ["-c:v", codec, "-preset", "p4", "-cq", str(edl.render.crf_equivalent)]
    else:
        encode = ["-c:v", "libx264", "-preset", "veryfast" if preview else "medium",
                  "-crf", str(28 if preview else edl.render.crf_equivalent)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    args += [
        "-filter_complex", ";".join(fc_parts),
        "-map", "[vout]", "-map", "[aout]",
        *encode,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    run_ffmpeg(args, total_duration_s=total_dur, on_progress=on_progress)
    return out_path, warnings
