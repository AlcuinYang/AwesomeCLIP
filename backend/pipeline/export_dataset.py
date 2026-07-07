"""P3 数据准备:用 L1 规则引擎的检测结果自动标注帧,导出 VLM 微调数据集(规格 §7)。

产出(项目目录/dataset/ 或 --out):
- images/*.jpg            采样帧
- labels.jsonl            每帧结构化标签(规则引擎产出即弱监督真值)
- sft.jsonl               Qwen2.5-VL 风格对话样本(user: 帧+提问 / assistant: JSON 标签)

采样策略:
- 正样本:每个 kill/death/round_end 事件时刻 + 事件后 0.5s(横幅持续期)
- 状态样本:每个 alive_state 事件时刻
- 负样本:距任何事件 >= 3s 的随机时刻,数量 = 正样本数 × negative_ratio
划分:按确定性哈希每 test_every_n 个样本留出为一致性测试集(split=test),
用于训练后校验 VLM 与规则引擎的一致率;职业转播素材只进 test,不进 train(规格 §7)。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from ..schemas.models import (
    AliveStateEvent, DeathEvent, EventsFile, KillEvent, RoundEndEvent,
)
from .project import Project

PROMPT = ("这是 Valorant 对局画面。请以 JSON 输出:kill_banner(画面下方是否正显示"
          "本人击杀横幅)、death(是否正显示被击杀横幅)、headshot(击杀是否爆头)、"
          "round_end(回合结算横幅: won/lost/null)、ally_alive/enemy_alive"
          "(顶部记分区双方存活人数,不可见则 null)。只输出 JSON。")


@dataclass
class Sample:
    t: float
    label: dict


def _samples_for_source(src, negative_ratio: float, min_gap_s: float) -> list[Sample]:
    kills = [e for e in src.events if isinstance(e, KillEvent)]
    deaths = [e for e in src.events if isinstance(e, DeathEvent)]
    rounds = [e for e in src.events if isinstance(e, RoundEndEvent)]
    alive = [e for e in src.events if isinstance(e, AliveStateEvent)]
    event_ts = sorted(e.t for e in [*kills, *deaths, *rounds])

    def base_label() -> dict:
        return {"kill_banner": False, "death": False, "headshot": False,
                "round_end": None, "ally_alive": None, "enemy_alive": None}

    def alive_at(t: float) -> tuple[Optional[int], Optional[int]]:
        last = None
        for a in alive:
            if a.t <= t:
                last = a
        return (last.ally_alive, last.enemy_alive) if last else (None, None)

    samples: list[Sample] = []

    def add(t: float, **overrides) -> None:
        if not 0 <= t < src.video_meta.duration_s:
            return
        label = base_label()
        ally, enemy = alive_at(t)
        label["ally_alive"], label["enemy_alive"] = ally, enemy
        label.update(overrides)
        samples.append(Sample(t=round(t, 3), label=label))

    for k in kills:
        add(k.t, kill_banner=True, headshot=k.headshot)
        add(k.t + 0.5, kill_banner=True, headshot=k.headshot)
    for d in deaths:
        add(d.t, death=True)
        add(d.t + 0.5, death=True)
    for r in rounds:
        add(r.t, round_end="won" if r.won else "lost")

    # 负样本:均匀扫描找距事件足够远的时刻
    n_neg = max(1, int(len(samples) * negative_ratio))
    t, step, added = 1.0, max(1.0, src.video_meta.duration_s / (n_neg * 7 + 1)), 0
    while added < n_neg and t < src.video_meta.duration_s - 0.5:
        if all(abs(t - et) >= min_gap_s for et in event_ts):
            add(t)
            added += 1
        t += step
    return samples


def _split_of(key: str, test_every_n: int) -> str:
    digest = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return "test" if digest % test_every_n == 0 else "train"


def export_dataset(project: Project, events: EventsFile, out_dir: Path,
                   settings: dict) -> dict:
    cfg = settings["dataset"]
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_f = open(out_dir / "labels.jsonl", "w", encoding="utf-8")
    sft_f = open(out_dir / "sft.jsonl", "w", encoding="utf-8")
    stats = {"train": 0, "test": 0, "skipped_unreadable": 0}

    try:
        for src in events.sources:
            video = project.resolve_media(src.source)
            cap = cv2.VideoCapture(str(video))
            try:
                for s in _samples_for_source(
                        src, float(cfg["negative_ratio"]), float(cfg["negative_min_gap_s"])):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s.t * src.video_meta.fps))
                    ok, frame = cap.read()
                    if not ok:
                        stats["skipped_unreadable"] += 1
                        continue
                    stem = Path(src.source).stem
                    name = f"{stem}_{s.t:08.3f}.jpg".replace(":", "-")
                    cv2.imwrite(str(img_dir / name), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg_quality"])])
                    split = _split_of(f"{src.source}:{s.t}", int(cfg["test_every_n"]))
                    stats[split] += 1
                    record = {"image": f"images/{name}", "source": src.source,
                              "t": s.t, "split": split, "label": s.label}
                    labels_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sft_f.write(json.dumps({
                        "split": split,
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "image", "image": f"images/{name}"},
                                {"type": "text", "text": PROMPT}]},
                            {"role": "assistant",
                             "content": json.dumps(s.label, ensure_ascii=False)},
                        ]}, ensure_ascii=False) + "\n")
            finally:
                cap.release()
    finally:
        labels_f.close()
        sft_f.close()
    return stats
