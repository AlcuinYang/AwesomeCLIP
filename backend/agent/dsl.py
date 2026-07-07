"""P1 DSL:自然语言剪辑的操作原语,直接作用于项目 JSON(规格 §7)。

十个操作:select / deselect / reorder / trim / align / fit_duration /
set_effect / set_music / map_section / undo。

- 每次成功操作追加一行 agent_log.jsonl:{ts, op, args, result, before:{edl, scorecards}},
  可回放(按序重放 op+args)也可撤销(undo 弹出最后一行并恢复 before 快照)。
- 改动切点的操作(select/reorder/trim/align/fit_duration/set_music/map_section)
  会自动重跑拍点吸附(place_clips),保证切换点始终落拍。
- 本层不碰网络,LLM function calling 只是把自然语言翻译成对本层的调用(llm.py)。
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..pipeline import project as proj
from ..pipeline.align import Clip, fit_clips, place_clips
from ..pipeline.project import Project
from ..schemas.models import (
    BeatsFile, EdlFile, EventsFile, ScoreCard, ScorecardsFile, SpeedRamp,
    TimelineEntry,
)


class DslError(ValueError):
    """操作参数/状态错误,信息面向 LLM 反馈,可被下一轮修正。"""


class AgentSession:
    def __init__(self, project: Project, settings: dict):
        self.project = project
        self.settings = settings
        self.cards: ScorecardsFile = project.load(proj.SCORECARDS_JSON, ScorecardsFile)
        self.events: EventsFile = project.load(proj.EVENTS_JSON, EventsFile)
        self.beats: BeatsFile = project.load(proj.BEATS_JSON, BeatsFile)
        self.edl: EdlFile = project.load(proj.EDL_JSON, EdlFile)
        # 会话内记住对齐模式(从现有 EDL 推断)
        self.anchor_align = any(e.snap.mode == "anchor_align" for e in self.edl.timeline)

    # ------------------------------------------------------------- 入口
    def apply(self, op: str, args: dict[str, Any]) -> str:
        if op == "undo":
            return self.undo()
        handler = getattr(self, f"op_{op}", None)
        if handler is None:
            raise DslError(f"未知操作: {op}")
        before = {
            "edl": self.edl.model_dump(),
            "scorecards": self.cards.model_dump(),
        }
        result = handler(**args)
        self._persist()
        self._log(op, args, result, before)
        return result

    def _persist(self) -> None:
        self.project.save(proj.EDL_JSON, self.edl)
        self.project.save(proj.SCORECARDS_JSON, self.cards)

    def _log(self, op: str, args: dict, result: str, before: dict) -> None:
        line = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "op": op, "args": args, "result": result, "before": before,
        }
        with open(self.project.root / proj.AGENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def undo(self) -> str:
        log_path = self.project.root / proj.AGENT_LOG
        if not log_path.exists():
            raise DslError("没有可撤销的操作。")
        lines = log_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise DslError("没有可撤销的操作。")
        last = json.loads(lines[-1])
        self.edl = EdlFile.model_validate(last["before"]["edl"])
        self.cards = ScorecardsFile.model_validate(last["before"]["scorecards"])
        self._persist()
        log_path.write_text("\n".join(lines[:-1]) + ("\n" if lines[:-1] else ""),
                            encoding="utf-8")
        return f"已撤销上一步操作: {last['op']}({last['result']})"

    # ------------------------------------------------------------- 工具
    def _card(self, clip_id: str) -> ScoreCard:
        for c in self.cards.clips:
            if c.clip_id == clip_id:
                return c
        raise DslError(f"找不到片段 {clip_id}")

    def _resolve_ref(self, ref: str | int) -> TimelineEntry:
        """时间线片段引用:1 起序号 / clip_id / 'last'。"""
        tl = self.edl.timeline
        if not tl:
            raise DslError("时间线为空。")
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            idx = int(ref)
            if not 1 <= idx <= len(tl):
                raise DslError(f"序号 {idx} 超出范围(时间线共 {len(tl)} 个片段)。")
            return tl[idx - 1]
        if ref == "last":
            return tl[-1]
        if ref == "first":
            return tl[0]
        for e in tl:
            if e.clip_id == ref:
                return e
        raise DslError(f"时间线上没有片段 {ref}")

    def _match(self, card: ScoreCard, *, tags_any: Optional[list[str]] = None,
               min_score: Optional[float] = None,
               min_multikill: Optional[int] = None,
               min_clutch_enemies: Optional[int] = None,
               has_flick: Optional[bool] = None,
               clip_ids: Optional[list[str]] = None) -> bool:
        if clip_ids is not None and card.clip_id not in clip_ids:
            return False
        if min_score is not None and card.score.total < min_score:
            return False
        if tags_any is not None and not any(
                t.startswith(prefix) for t in card.tags for prefix in tags_any):
            return False
        if min_multikill is not None:
            ns = [int(m.group(1)) for t in card.tags
                  if (m := re.fullmatch(r"multikill_(\d+)", t))]
            if not ns or max(ns) < min_multikill:
                return False
        if min_clutch_enemies is not None:
            ns = [int(m.group(1)) for t in card.tags
                  if (m := re.fullmatch(r"clutch_1v(\d+)", t))]
            if not ns or max(ns) < min_clutch_enemies:
                return False
        if has_flick is not None and ("flick" in card.tags) != has_flick:
            return False
        return True

    def _src_duration(self, source: str) -> float:
        for s in self.events.sources:
            if s.source == source:
                return s.video_meta.duration_s
        return float("inf")

    def _entry_to_clip(self, e: TimelineEntry) -> Clip:
        card = self._card(e.clip_id)
        return Clip(clip_id=e.clip_id, source=e.source, in_t=e.in_t, out_t=e.out_t,
                    anchors=list(card.anchor_ts), score=card.score.total,
                    source_duration=self._src_duration(e.source),
                    effects=e.effects, audio=e.audio)

    def _replace(self, clips: Optional[list[Clip]] = None) -> list[str]:
        """用给定(或当前)片段序列重跑拍点吸附,更新时间线。"""
        if clips is None:
            clips = [self._entry_to_clip(e) for e in self.edl.timeline]
        entries, warnings = place_clips(clips, self.beats.beats_t, self.settings,
                                        anchor_align=self.anchor_align)
        self.edl.timeline = entries
        return warnings

    @staticmethod
    def _fmt(warnings: list[str]) -> str:
        return ("；" + "；".join(warnings)) if warnings else ""

    # ------------------------------------------------------------- 操作
    def op_select(self, mode: str = "add", **filt) -> str:
        """mode='only': 仅保留匹配片段;'add': 追加匹配片段。"""
        matched = [c for c in self.cards.clips if self._match(c, **filt)]
        if not matched:
            raise DslError(f"筛选条件没有匹配到任何片段(条件: {filt})。")
        matched_ids = {c.clip_id for c in matched}
        if mode == "only":
            for c in self.cards.clips:
                c.selected = c.clip_id in matched_ids
        elif mode == "add":
            for c in self.cards.clips:
                if c.clip_id in matched_ids:
                    c.selected = True
        else:
            raise DslError(f"未知 mode: {mode}(可选 only/add)")
        # 时间线:保留仍选中的(原顺序),新选中的按分数降序补到末尾
        kept = [self._entry_to_clip(e) for e in self.edl.timeline
                if self._card(e.clip_id).selected]
        kept_ids = {c.clip_id for c in kept}
        added = sorted((c for c in self.cards.clips
                        if c.selected and c.clip_id not in kept_ids),
                       key=lambda c: c.score.total, reverse=True)
        new_clips = kept + [self._card_to_clip(c) for c in added]
        if not new_clips:
            raise DslError("操作后时间线为空,已拒绝。")
        warnings = self._replace(new_clips)
        sel = [c.clip_id for c in self.cards.clips if c.selected]
        return f"已选中 {len(sel)} 个片段: {', '.join(sel)}{self._fmt(warnings)}"

    def _card_to_clip(self, c: ScoreCard) -> Clip:
        return Clip(clip_id=c.clip_id, source=c.source, in_t=c.span.start_t,
                    out_t=c.span.end_t, anchors=list(c.anchor_ts),
                    score=c.score.total,
                    source_duration=self._src_duration(c.source))

    def op_deselect(self, **filt) -> str:
        matched = [c for c in self.cards.clips if c.selected and self._match(c, **filt)]
        if not matched:
            raise DslError(f"筛选条件没有匹配到任何已选片段(条件: {filt})。")
        for c in matched:
            c.selected = False
        remaining = [self._entry_to_clip(e) for e in self.edl.timeline
                     if self._card(e.clip_id).selected]
        if not remaining:
            raise DslError("操作后时间线为空,已拒绝;请改用 select mode='only'。")
        warnings = self._replace(remaining)
        return (f"已移除 {len(matched)} 个片段: "
                f"{', '.join(c.clip_id for c in matched)}{self._fmt(warnings)}")

    def op_reorder(self, order: list[str]) -> str:
        """order 为 clip_id 列表;未提及的片段保持相对顺序跟在后面。"""
        current = {e.clip_id: e for e in self.edl.timeline}
        unknown = [cid for cid in order if cid not in current]
        if unknown:
            raise DslError(f"这些片段不在时间线上: {', '.join(unknown)}")
        rest = [e.clip_id for e in self.edl.timeline if e.clip_id not in order]
        new_order = list(order) + rest
        warnings = self._replace([self._entry_to_clip(current[cid])
                                  for cid in new_order])
        return f"新顺序: {' → '.join(new_order)}{self._fmt(warnings)}"

    def op_trim(self, clip: str | int, in_delta_s: float = 0.0,
                out_delta_s: float = 0.0, in_t: Optional[float] = None,
                out_t: Optional[float] = None) -> str:
        """delta 为相对调整(in_delta_s=-1 即击杀前多留 1s);in_t/out_t 为绝对设置。"""
        entry = self._resolve_ref(clip)
        src_dur = self._src_duration(entry.source)
        new_in = in_t if in_t is not None else entry.in_t + in_delta_s
        new_out = out_t if out_t is not None else entry.out_t + out_delta_s
        new_in = max(0.0, min(new_in, src_dur))
        new_out = max(0.0, min(new_out, src_dur))
        if new_out - new_in < 0.2:
            raise DslError(f"裁剪后片段过短({new_out - new_in:.2f}s)。")
        entry.in_t, entry.out_t = round(new_in, 3), round(new_out, 3)
        warnings = self._replace()
        return (f"{entry.clip_id}: in={entry.in_t:.2f}s out={entry.out_t:.2f}s "
                f"(时长 {entry.out_t - entry.in_t:.2f}s){self._fmt(warnings)}")

    def op_align(self, mode: str) -> str:
        if mode not in ("cut_on_beat", "anchor_align"):
            raise DslError(f"未知对齐模式: {mode}")
        self.anchor_align = mode == "anchor_align"
        warnings = self._replace()
        return f"对齐模式已切换为 {mode}{self._fmt(warnings)}"

    def op_fit_duration(self, target_s: float) -> str:
        if target_s <= 0:
            raise DslError("目标时长必须为正。")
        warnings: list[str] = []
        clips = [self._entry_to_clip(e) for e in self.edl.timeline]
        clips = fit_clips(clips, target_s, self.settings, warnings)
        kept_ids = {c.clip_id for c in clips}
        for c in self.cards.clips:
            if c.selected and c.clip_id not in kept_ids and \
                    any(e.clip_id == c.clip_id for e in self.edl.timeline):
                c.selected = False
        warnings += self._replace(clips)
        self.edl.target_duration_s = target_s
        total = sum(e.out_t - e.in_t for e in self.edl.timeline)
        return (f"目标 {target_s:.0f}s,当前 {len(self.edl.timeline)} 个片段"
                f"共 {total:.1f}s{self._fmt(warnings)}")

    def op_set_effect(self, target: str | int, effect: str,
                      enabled: bool = True, strength: Optional[float] = None,
                      factor: Optional[float] = None) -> str:
        entries = (list(self.edl.timeline) if target == "all"
                   else [self._resolve_ref(target)])
        fx_cfg = self.settings["effects"]
        for e in entries:
            if effect == "frame_drop":
                e.effects.frame_drop = enabled
                if strength is not None:
                    if not 0.1 <= strength <= 0.9:
                        raise DslError("frame_drop 强度需在 0.1~0.9 之间。")
                    e.effects.frame_drop_strength = strength
            elif effect == "speed_ramp":
                if not enabled:
                    e.effects.speed_ramp = None
                else:
                    card = self._card(e.clip_id)
                    anchors = [a for a in card.anchor_ts if e.in_t < a < e.out_t]
                    anchor = anchors[0] if anchors else (e.in_t + e.out_t) / 2
                    ramp_cfg = fx_cfg["speed_ramp"]
                    e.effects.speed_ramp = SpeedRamp(
                        anchor_t=round(anchor, 3),
                        factor=factor if factor is not None else float(ramp_cfg["factor"]),
                        pre_s=float(ramp_cfg["pre_s"]),
                        post_s=float(ramp_cfg["post_s"]))
            else:
                raise DslError(f"未知特效: {effect}(可选 frame_drop/speed_ramp)")
        names = ", ".join(e.clip_id for e in entries)
        state = "开启" if enabled else "关闭"
        extra = f",强度 {strength}" if strength is not None else ""
        return f"{names}: {effect} 已{state}{extra}"

    def op_set_music(self, path: str) -> str:
        from ..pipeline.beat import analyze_music
        from ..pipeline.ingest import _link_into

        music = Path(path)
        if not music.is_absolute():
            candidate = self.project.root / path
            music = candidate if candidate.exists() else music
        if not music.exists():
            raise DslError(f"音乐文件不存在: {path}")
        if music.resolve().parent != (self.project.root / "music").resolve():
            music = _link_into(music, self.project.root / "music")
        rel = f"music/{music.name}"
        self.beats = analyze_music(self.project.resolve_media(rel), rel, self.settings)
        self.project.save(proj.BEATS_JSON, self.beats)
        self.edl.music = rel
        warnings = self._replace()
        return (f"BGM 已换为 {music.name}(BPM {self.beats.bpm},"
                f"{len(self.beats.beats_t)} 拍){self._fmt(warnings)}")

    def op_map_section(self, strategy: str = "energy") -> str:
        """按音乐能量重排:高分片段对应高能量段(能量曲线来自 beats.json)。"""
        if strategy != "energy":
            raise DslError(f"未知策略: {strategy}(V1 仅支持 energy)")
        if not self.beats.energy_curve:
            raise DslError("beats.json 缺少能量曲线,请重跑 analyze-music。")
        clips = [self._entry_to_clip(e) for e in self.edl.timeline]
        n = len(clips)
        if n < 2:
            return "时间线不足 2 个片段,无需重排。"
        total = sum(c.out_t - c.in_t for c in clips)
        seg_energy = []
        for i in range(n):
            lo, hi = total * i / n, total * (i + 1) / n
            vals = [p.rms for p in self.beats.energy_curve if lo <= p.t < hi]
            seg_energy.append(sum(vals) / len(vals) if vals else 0.0)
        # 能量最高的段位放分数最高的片段
        seg_rank = sorted(range(n), key=lambda i: seg_energy[i], reverse=True)
        clip_rank = sorted(range(n), key=lambda i: clips[i].score, reverse=True)
        new_clips: list[Clip | None] = [None] * n
        for seg_i, clip_i in zip(seg_rank, clip_rank):
            new_clips[seg_i] = clips[clip_i]
        warnings = self._replace(new_clips)  # type: ignore[arg-type]
        order = " → ".join(c.clip_id for c in new_clips)  # type: ignore[union-attr]
        return f"已按能量重排: {order}{self._fmt(warnings)}"

    # ------------------------------------------------------------- 上下文
    def context_summary(self) -> str:
        """给 LLM 的当前状态摘要(scorecards + 时间线 + 音乐)。"""
        lines = [f"## 音乐\n{self.beats.music},BPM {self.beats.bpm},"
                 f"拍点间隔约 {60 / self.beats.bpm:.2f}s",
                 f"\n## 全部片段(scorecards,按分数降序)"]
        for c in self.cards.clips:
            lines.append(
                f"- {c.clip_id} score={c.score.total} tags={c.tags} "
                f"span={c.span.start_t:.1f}~{c.span.end_t:.1f}s "
                f"kills={len(c.evidence.kills)} selected={c.selected}"
                + (f" 叙述: {c.narration}" if c.narration else ""))
        total = sum(e.out_t - e.in_t for e in self.edl.timeline)
        lines.append(f"\n## 当前时间线(共 {total:.1f}s,目标 "
                     f"{self.edl.target_duration_s:.0f}s,顺序即成片顺序)")
        for i, e in enumerate(self.edl.timeline, 1):
            fx = []
            if e.effects.frame_drop:
                fx.append(f"frame_drop({e.effects.frame_drop_strength})")
            if e.effects.speed_ramp:
                fx.append("speed_ramp")
            lines.append(f"{i}. {e.clip_id} in={e.in_t:.2f} out={e.out_t:.2f} "
                         f"时长 {e.out_t - e.in_t:.2f}s snap={e.snap.mode}"
                         + (f" 特效: {','.join(fx)}" if fx else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------- 工具 schema
# OpenAI function calling 格式(OpenRouter 兼容),llm.py 直接使用。
FILTER_PROPS = {
    "clip_ids": {"type": "array", "items": {"type": "string"},
                 "description": "指定 clip_id 列表"},
    "tags_any": {"type": "array", "items": {"type": "string"},
                 "description": "标签前缀列表,命中任一即匹配,如 ['clutch'] 匹配所有残局"},
    "min_score": {"type": "number", "description": "分数下限"},
    "min_multikill": {"type": "integer", "description": "多杀数下限,如 3 表示三杀及以上"},
    "min_clutch_enemies": {"type": "integer", "description": "残局对面人数下限,如 2 表示 1v2 及以上"},
    "has_flick": {"type": "boolean", "description": "是否要求含神经枪"},
}

TOOLS: list[dict] = [
    {"name": "select",
     "description": "选中片段进入成片。mode='only' 表示仅保留匹配的(其余移出),'add' 表示追加。",
     "parameters": {"type": "object", "properties": {
         "mode": {"type": "string", "enum": ["only", "add"]}, **FILTER_PROPS}}},
    {"name": "deselect", "description": "把匹配的片段移出成片。",
     "parameters": {"type": "object", "properties": FILTER_PROPS}},
    {"name": "reorder",
     "description": "重排时间线。order 为 clip_id 列表,未提及的保持相对顺序跟在后面。",
     "parameters": {"type": "object", "properties": {
         "order": {"type": "array", "items": {"type": "string"}}},
         "required": ["order"]}},
    {"name": "trim",
     "description": "裁剪某片段的入出点。clip 可为 1 起的时间线序号、clip_id、'first'/'last'。"
                    "in_delta_s 为入点相对调整(负值=向前多留,如击杀前多留 1 秒是 -1);"
                    "out_delta_s 同理;in_t/out_t 为绝对设置(源内时间)。",
     "parameters": {"type": "object", "properties": {
         "clip": {"type": "string"},
         "in_delta_s": {"type": "number"}, "out_delta_s": {"type": "number"},
         "in_t": {"type": "number"}, "out_t": {"type": "number"}},
         "required": ["clip"]}},
    {"name": "align", "description": "切换对齐模式:cut_on_beat(切点吸附拍点,默认)或 anchor_align(击杀帧对齐拍点)。",
     "parameters": {"type": "object", "properties": {
         "mode": {"type": "string", "enum": ["cut_on_beat", "anchor_align"]}},
         "required": ["mode"]}},
    {"name": "fit_duration", "description": "把成片总长压到目标秒数(先收缩击杀前留白,再丢低分片段)。",
     "parameters": {"type": "object", "properties": {
         "target_s": {"type": "number"}}, "required": ["target_s"]}},
    {"name": "set_effect",
     "description": "开关片段特效。target 可为时间线序号、clip_id、'first'/'last'/'all'。"
                    "frame_drop 的 strength 为丢帧比例 0.1~0.9(默认 0.5,调低=更轻微);"
                    "speed_ramp 为击杀慢放,factor 默认 0.5。",
     "parameters": {"type": "object", "properties": {
         "target": {"type": "string"},
         "effect": {"type": "string", "enum": ["frame_drop", "speed_ramp"]},
         "enabled": {"type": "boolean"},
         "strength": {"type": "number"}, "factor": {"type": "number"}},
         "required": ["target", "effect"]}},
    {"name": "set_music", "description": "更换 BGM 并重新分析节拍、重吸附切点。path 为音乐文件路径。",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}}, "required": ["path"]}},
    {"name": "map_section", "description": "按音乐能量重排片段:高分片段放到高能量段落。",
     "parameters": {"type": "object", "properties": {
         "strategy": {"type": "string", "enum": ["energy"]}}}},
    {"name": "undo", "description": "撤销上一步操作。",
     "parameters": {"type": "object", "properties": {}}},
]
