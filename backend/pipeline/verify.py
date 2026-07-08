"""verify:击杀 node 人工核对表(防漏杀,用户工作流的质检环节)。

对 events.json 里的每个 kill/death,在事件时刻截取信息流区域拼成核对图;
即时回放素材不可能没有击杀,检出 0 杀的素材单独报警。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schemas.models import EventsFile
from .project import Project

TILE_W = 430
COLS = 4
MAX_ROWS_PER_SHEET = 10


def build_verify_sheets(project: Project, events: EventsFile,
                        out_dir: Path) -> tuple[list[Path], list[str]]:
    """生成核对拼图,返回 (图片路径列表, 警告列表)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    tiles: list[np.ndarray] = []

    for src in events.sources:
        rows = [(e.type, e.t) for e in src.events if e.type in ("kill", "death")]
        name = Path(src.source).name
        if not any(k == "kill" for k, _ in rows):
            warnings.append(f"{name}: 未检出任何击杀——即时回放素材通常都有击杀,"
                            f"请检查 HUD 遮挡(网络统计悬浮窗)或分辨率校准。")
        video = project.resolve_media(src.source)
        cap = cv2.VideoCapture(str(video))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or src.video_meta.fps
            for i, (kind, t) in enumerate(rows):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + 0.1) * fps))
                ok, frame = cap.read()
                if not ok:
                    continue
                fh, fw = frame.shape[:2]
                crop = frame[int(0.05 * fh):int(0.26 * fh), int(0.70 * fw):fw]
                scale_h = int(crop.shape[0] * TILE_W / crop.shape[1])
                crop = cv2.resize(crop, (TILE_W, scale_h))
                label = f"{Path(name).stem} {'KILL' if kind == 'kill' else 'DEATH'} {t:.1f}s"
                cv2.rectangle(crop, (0, 0), (220, 22), (0, 0, 0), -1)
                cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1)
                tiles.append(crop)
        finally:
            cap.release()

    paths: list[Path] = []
    if not tiles:
        return paths, warnings
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    per_sheet = COLS * MAX_ROWS_PER_SHEET
    for sheet_i in range(0, len(tiles), per_sheet):
        chunk = tiles[sheet_i:sheet_i + per_sheet]
        grid_rows = []
        for r in range(0, len(chunk), COLS):
            row = chunk[r:r + COLS]
            while len(row) < COLS:
                row.append(np.zeros_like(chunk[0]))
            grid_rows.append(np.hstack(row))
        out = out_dir / f"kill_nodes_{sheet_i // per_sheet + 1}.png"
        cv2.imwrite(str(out), np.vstack(grid_rows))
        paths.append(out)
    return paths, warnings
