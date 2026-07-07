"""ingest:素材目录 + BGM → 项目;ffprobe 元信息;后台生成 720p proxy(规格 §5.1)。"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .ffmpeg_utils import ffprobe_meta, spawn_ffmpeg
from .project import Project

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


@dataclass
class IngestResult:
    sources: list[Path] = field(default_factory=list)
    music: Path | None = None
    metas: dict[str, dict] = field(default_factory=dict)   # rel path -> ffprobe meta
    proxy_procs: list[subprocess.Popen] = field(default_factory=list)


def _link_into(src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / src.name
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)  # 跨设备/无软链权限时退化为复制
    return dest


def collect_videos(inputs: list[Path]) -> list[Path]:
    """文件/目录混合输入 → 去重排序的视频文件列表(一次可传多个)。"""
    videos: dict[str, Path] = {}
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            found = [p for p in item.iterdir()
                     if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")]
            for v in found:
                videos[v.name] = v
        elif item.suffix.lower() in VIDEO_EXTS:
            if not item.exists():
                raise FileNotFoundError(f"视频文件不存在: {item}")
            videos[item.name] = item
        else:
            raise ValueError(f"不支持的输入: {item}(需要视频文件或目录)")
    return [videos[k] for k in sorted(videos)]


def ingest(project: Project, sources: Path | list[Path], music: Path | None,
           settings: dict, make_proxies: bool = True) -> IngestResult:
    result = IngestResult()
    inputs = [sources] if isinstance(sources, (str, Path)) else list(sources)
    videos = collect_videos(inputs)
    if not videos:
        raise FileNotFoundError(
            f"输入中没有视频文件({'/'.join(sorted(VIDEO_EXTS))})")

    proxy_cfg = settings["render"]["proxy"]
    for v in videos:
        linked = _link_into(v, project.root / "sources")
        rel = f"sources/{linked.name}"
        result.sources.append(linked)
        result.metas[rel] = ffprobe_meta(v)
        if make_proxies:
            proxy_out = project.root / "proxies" / (linked.stem + ".mp4")
            if not proxy_out.exists():
                result.proxy_procs.append(spawn_ffmpeg([
                    "-i", str(v.resolve()),
                    "-vf", f"scale=-2:{proxy_cfg['height']}",
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-b:v", str(proxy_cfg["bitrate"]),
                    "-c:a", "aac", "-b:a", "96k",
                    str(proxy_out),
                ]))

    if music is not None:
        music = Path(music)
        if music.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"不支持的音频格式: {music.suffix}")
        result.music = _link_into(music, project.root / "music")
    return result
