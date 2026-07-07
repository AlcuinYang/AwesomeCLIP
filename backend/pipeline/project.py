"""剪辑会话项目目录管理 + JSON 中间产物读写(规格 §3 目录布局)。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

SUBDIRS = ["sources", "proxies", "music", "output"]

EVENTS_JSON = "events.json"
SCORECARDS_JSON = "scorecards.json"
BEATS_JSON = "beats.json"
EDL_JSON = "edl.json"
AGENT_LOG = "agent_log.jsonl"

M = TypeVar("M", bound=BaseModel)


class Project:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    @classmethod
    def init(cls, root: Path) -> "Project":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (root / sub).mkdir(exist_ok=True)
        return cls(root)

    @classmethod
    def find(cls, start: Optional[Path] = None) -> "Project":
        """从 start(默认 cwd)向上找项目根(以 sources/ 目录为标志)。"""
        cur = Path(start or Path.cwd()).resolve()
        for candidate in [cur, *cur.parents]:
            if (candidate / "sources").is_dir():
                return cls(candidate)
        raise FileNotFoundError(
            f"在 {cur} 及其上级目录未找到 vmontage 项目(缺 sources/)。先运行 `vmontage init <dir>`。"
        )

    # ------------------------------------------------------------ JSON IO
    def _path(self, name: str) -> Path:
        return self.root / name

    def save(self, name: str, model: BaseModel) -> Path:
        path = self._path(name)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, name: str, model_cls: Type[M]) -> M:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在,请先运行上游阶段。")
        return model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def resolve_media(self, rel: str) -> Path:
        """把 JSON 里的相对路径(如 sources/clip.mp4)解析为绝对路径,软链接取真身。"""
        p = Path(rel)
        if not p.is_absolute():
            p = self.root / p
        return p.resolve()
