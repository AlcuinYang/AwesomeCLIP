"""配置加载:内置 settings.yaml 为默认值,项目目录同名文件浅合并覆盖。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_DIR = Path(__file__).parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_settings(project_dir: Optional[Path] = None) -> dict[str, Any]:
    with open(CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    if project_dir is not None:
        override_path = Path(project_dir) / "settings.yaml"
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            settings = _deep_merge(settings, override)
    return settings


def load_roi_profile(width: int, height: int, settings: dict[str, Any],
                     project_dir: Optional[Path] = None) -> dict[str, Any]:
    """按视频分辨率取 ROI 校准文件;无匹配时报错并提示校准流程(不静默降级)。"""
    key = f"{width}x{height}"
    resolutions = settings["detector"]["resolutions"]
    if key not in resolutions:
        supported = ", ".join(resolutions)
        raise ValueError(
            f"不支持的分辨率 {key}(已校准: {supported})。"
            f"请复制 backend/config/roi_1440p.yaml 为 roi_{key}.yaml 调整比例坐标,"
            f"并在 settings.yaml 的 detector.resolutions 中注册,"
            f"然后运行 `vmontage calibrate <样例帧>` 截取模板。"
        )
    filename = resolutions[key]
    # 项目目录可放同名文件覆盖(校准结果随项目走)
    for base in ([Path(project_dir)] if project_dir else []) + [CONFIG_DIR]:
        path = Path(base) / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"ROI 校准文件不存在: {filename}")
