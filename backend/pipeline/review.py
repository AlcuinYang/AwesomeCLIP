"""评审辅助:从 EDL 生成 .srt 标注字幕(与 preview.mp4 同名放一起,播放器自动加载)。

播放时实时显示:当前镜头(S1/S2…)、来源、当前处理(常速/压缩xN/慢放),
每个变速段有独立标签(S1-G1…)——用户评审时可直接引用标签给反馈。
"""
from __future__ import annotations

from pathlib import Path

from ..schemas.models import EdlFile
from .effects import pieces_of


def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_review_srt(edl: EdlFile, out_path: Path) -> Path:
    cues: list[tuple[float, float, str]] = []
    cursor = 0.0
    for si, entry in enumerate(edl.timeline, 1):
        gi = 0
        for (a, b, factor) in pieces_of(entry):
            out_dur = (b - a) / factor
            src = Path(entry.source).stem
            if factor == 1.0:
                label = f"S{si} {entry.clip_id} [{src}] 常速 源{a:.1f}~{b:.1f}s"
            else:
                gi += 1
                kind = "压缩" if factor > 1 else "慢放"
                label = (f"S{si}-G{gi} {entry.clip_id} [{src}] "
                         f"◀◀ {kind}x{factor:g} 源{a:.1f}~{b:.1f}s ▶▶")
            cues.append((cursor, cursor + out_dur, label))
            cursor += out_dur
    lines = []
    for i, (t0, t1, text) in enumerate(cues, 1):
        lines += [str(i), f"{_ts(t0)} --> {_ts(t1)}", text, ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
