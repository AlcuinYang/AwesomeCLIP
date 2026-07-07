"""ffmpeg / ffprobe 子进程封装(禁止 moviepy,规格 §8)。"""
from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable


class FFmpegNotFound(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise FFmpegNotFound(f"未找到 {binary},请安装 ffmpeg >= 6.0 并加入 PATH。")
    return path


def ffprobe_meta(video: Path) -> dict:
    """返回 {width, height, fps, duration_s, has_audio}。"""
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = vstream.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    if fps <= 0:
        num, den = vstream.get("r_frame_rate", "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    duration = float(info["format"].get("duration") or vstream.get("duration") or 0.0)
    return {
        "width": int(vstream["width"]),
        "height": int(vstream["height"]),
        "fps": fps,
        "duration_s": duration,
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def run_ffmpeg(args: list[str], quiet: bool = True,
               total_duration_s: float | None = None,
               on_progress: "Callable[[float], None] | None" = None) -> None:
    """执行 ffmpeg。给了 on_progress + total_duration_s 时解析 -progress 输出,
    以 0..1 的比例回调(WS 进度推送用)。"""
    cmd = [_require("ffmpeg"), "-hide_banner", "-y"]
    if quiet:
        cmd += ["-loglevel", "error"]
    if on_progress and total_duration_s:
        cmd += ["-progress", "pipe:1", "-nostats"]
        proc = subprocess.Popen(cmd + args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit():
                on_progress(min(1.0, int(value) / 1e6 / total_duration_s))
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"ffmpeg 失败 (exit {proc.returncode}):\n{stderr[-3000:]}")
        on_progress(1.0)
        return
    result = subprocess.run(cmd + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败 (exit {result.returncode}):\n{result.stderr[-3000:]}")


def spawn_ffmpeg(args: list[str]) -> subprocess.Popen:
    """后台运行(proxy 生成不阻塞检测,规格 §5.1)。"""
    cmd = [_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y"] + args
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    out = subprocess.run(
        [_require("ffmpeg"), "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    ).stdout
    return {line.split()[1] for line in out.splitlines()
            if line.strip().startswith(("V", "A")) and len(line.split()) >= 2}


def pick_video_codec(requested: str) -> tuple[str, list[str]]:
    """返回 (实际编码器, 警告列表)。NVENC 不可用回退 libx264(规格 §5.8)。"""
    warnings: list[str] = []
    if requested in available_encoders():
        return requested, warnings
    warnings.append(f"编码器 {requested} 不可用,回退 libx264(速度较慢)。")
    return "libx264", warnings
