"""MLLM 导演(llm as judge):以导演思维链对片段做"段内二次处理"裁决 → 分镜表。

设计(与用户讨论定稿):
- 输入:簇级片段(不做机械 18s 切分——切不切由导演决定)+ 证据卡 + 每段关键帧。
- 主业:逐个审视片段内击杀间隙,二选一 keep/cut;击杀一个不能丢;
  无叙事曲线、无地图武器多样性这类伪准则;排序为次要项。
  (v3 起废弃 compress:导演看不到间隙内画面,变速点位判断不可靠;且用户参考大量
  高光成片后确认主流做法就是直接剪掉。模型若仍输出 compress,校验器降级为 cut 并告警。)
- 输出:storyboard.json(一等产物,每个裁决必须带 rationale)→ 生成 edl.json。
- 可解释性/数据沉淀:模型完整思考过程(reasoning_content)存入 storyboard.reasoning,
  每次 pass 追加 director_log.jsonl(输入摘要+分镜+思考),供优化提示词与未来偏好训练。
- 程序校验兜底:非法 clip_id / 越界跨度 / 覆盖击杀的处理一律拒绝该项并告警,不静默修。
"""
from __future__ import annotations

import base64
import copy
import datetime
import json
from pathlib import Path
from typing import Optional

import cv2

from ..pipeline import project as proj
from ..pipeline.project import Project
from ..pipeline.scorer import score_clips
from ..pipeline.semantic import build_scorecards
from ..schemas.models import (
    ClipAudio, ClipEffects, EdlFile, EventsFile, GlobalEffects, RenderSettings,
    ScorecardsFile, SnapInfo, SpeedSpan, Storyboard, TimelineEntry,
)
from .llm import LlmError, OpenRouterClient

RUBRIC_VERSION = "v3"

DIRECTOR_PROMPT = """你是电竞(Valorant)集锦剪辑导演。下面给你候选片段清单:每段含击杀时间戳、\
时长、标签、评分,以及关键帧截图(按清单顺序一一对应)。你的任务是对每个片段做"段内二次处理"\
裁决,产出成片分镜表。

核心准则:
1. 击杀一个都不能丢。两杀间隔长绝不是删杀的理由。
2. 逐个审视片段内每个击杀间隙(相邻击杀时间戳之差),二选一:
   - keep:间隙 < 6s,或期间可能有值得看的内容(残局对峙、下包、关键走位)
   - cut:间隙没有观赏价值(赶路、找人、等待)→ 直接剪掉该跨度,硬切拼接
   不允许变速/压缩,只有保留和剪掉两种处理。
   剪切跨度必须远离击杀:from_t 距上一杀 >= {buf_after}s,to_t 距下一杀
   >= {buf_before}s——交火从拉枪到击杀落地的全过程必须完整保留,只剪纯赶路。
   间隙扣除两端缓冲后不足 {min_span}s 就选 keep,不值得处理。
3. 每镜头入点 in_t:第一杀前留 1.5~2.5s(首镜头可短至 0.5s);出点 out_t:尾杀后 0.8~1.2s。
4. 整段平庸(如单杀且无亮点)可弃,进 rejected 并写明理由。
5. 排序:默认强度高的在前;有更好理由可调整。
6. 不使用"地图/武器重复"作为取舍理由;不追求叙事曲线,集锦以密度和爽感为先。
7. 全覆盖:清单里的每个片段都必须出现在 shots 或 rejected 之一,一个都不能漏、不能不表态。

每个裁决必须给 rationale:一句话,引用你看到的具体证据(击杀分布/画面内容),禁止套话。

{style_line}

只输出一个 JSON 对象,不要任何其他文字,格式:
{{"shots": [{{"clip_id": "...", "in_t": 秒, "out_t": 秒,
  "gap_treatments": [{{"from_t": 秒, "to_t": 秒, "action": "keep|cut",
                      "rationale": "..."}}],
  "rationale": "..."}}],
 "rejected": [{{"clip_id": "...", "reason": "..."}}]}}

候选片段清单:
{clips_block}"""


# ------------------------------------------------------------------ 输入构建

def build_director_cards(project: Project, settings: dict) -> ScorecardsFile:
    """簇级证据卡:关掉机械切分(max_clip_s),切不切由导演决定。"""
    events = project.load(proj.EVENTS_JSON, EventsFile)
    s = copy.deepcopy(settings)
    s["semantic"]["clip"]["max_clip_s"] = 0
    return score_clips(build_scorecards(events, s), s)


def _clips_block(cards: ScorecardsFile) -> str:
    lines = []
    for i, c in enumerate(cards.clips, 1):
        if not c.selected:
            continue
        gaps = [round(b - a, 1) for a, b in zip(c.anchor_ts, c.anchor_ts[1:])]
        lines.append(
            f"{i}. {c.clip_id} 来源={c.source.split('/')[-1]} "
            f"跨度={c.span.start_t:.1f}~{c.span.end_t:.1f}s "
            f"击杀时间戳={[round(t, 1) for t in c.anchor_ts]} "
            f"相邻击杀间隙={gaps} 标签={c.tags} 评分={c.score.total}")
    return "\n".join(lines)


def extract_keyframes(project: Project, cards: ScorecardsFile,
                      frame_width: int, jpeg_quality: int) -> list[str]:
    """每个入选片段取第一杀瞬间的关键帧 → base64 jpeg(与清单顺序一致)。"""
    frames: list[str] = []
    caps: dict[str, cv2.VideoCapture] = {}
    try:
        for c in cards.clips:
            if not c.selected:
                continue
            cap = caps.setdefault(
                c.source, cv2.VideoCapture(str(project.resolve_media(c.source))))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60
            t = c.anchor_ts[0] if c.anchor_ts else c.span.start_t
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                frames.append("")
                continue
            h = int(frame.shape[0] * frame_width / frame.shape[1])
            small = cv2.resize(frame, (frame_width, h))
            ok, buf = cv2.imencode(".jpg", small,
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frames.append(base64.b64encode(buf).decode() if ok else "")
    finally:
        for cap in caps.values():
            cap.release()
    return frames


# ------------------------------------------------------------------ 校验

def validate_storyboard(sb: Storyboard, cards: ScorecardsFile,
                        events: EventsFile, settings: dict) -> list[str]:
    """程序校验:剔除非法镜头/处理并返回告警。不静默修正。"""
    warnings: list[str] = []
    by_id = {c.clip_id: c for c in cards.clips}
    src_dur = {s.source: s.video_meta.duration_s for s in events.sources}

    valid_shots = []
    for shot in sb.shots:
        card = by_id.get(shot.clip_id)
        if card is None or not card.selected:
            warnings.append(f"分镜引用了不存在/未入选的片段 {shot.clip_id},已剔除。")
            continue
        dur = src_dur.get(card.source, card.span.end_t)
        shot.in_t = round(max(0.0, min(shot.in_t, dur)), 3)
        shot.out_t = round(max(shot.in_t + 0.5, min(shot.out_t, dur)), 3)
        kills = [t for t in card.anchor_ts if shot.in_t <= t <= shot.out_t]
        if len(kills) < len(card.anchor_ts):
            warnings.append(f"{shot.clip_id}: 镜头范围丢了 "
                            f"{len(card.anchor_ts) - len(kills)} 个击杀(准则1),"
                            f"已扩回全部击杀。")
            shot.in_t = round(min(shot.in_t, card.anchor_ts[0] - 0.5), 3)
            shot.out_t = round(max(shot.out_t, card.anchor_ts[-1] + 0.8), 3)
            kills = card.anchor_ts
        buf_after = float(settings["director"]["buffer_after_kill_s"])
        buf_before = float(settings["director"]["buffer_before_kill_s"])
        min_span = float(settings["director"]["min_span_s"])
        valid_gaps = []
        for g in sorted(shot.gap_treatments, key=lambda g: g.from_t):
            if g.action == "keep":
                continue  # keep 无需落实,只是裁决记录
            if g.action == "compress":
                # 准则 v3:只剪不加速(导演看不到间隙画面,变速点位不可靠)
                warnings.append(f"{shot.clip_id}: compress {g.from_t:.1f}-{g.to_t:.1f} "
                                f"已降级为 cut(准则 v3 只剪不加速)。")
                g.action, g.factor = "cut", None
            if not (shot.in_t <= g.from_t < g.to_t <= shot.out_t):
                warnings.append(f"{shot.clip_id}: 间隙处理 {g.from_t}-{g.to_t} 越界,已剔除。")
                continue
            # 硬缓冲:交火全程(拉枪→击杀落地)必须常速;时间戳晚于真实击杀,
            # 缓冲同时补偿该延迟。越界不剔除而是收紧到合法窗口(告警可见)。
            prev_kill = max((t for t in kills if t <= g.to_t), default=None)
            next_kill = min((t for t in kills if t >= g.from_t), default=None)
            lo = (prev_kill + buf_after) if prev_kill is not None else shot.in_t
            hi = (next_kill - buf_before) if next_kill is not None else shot.out_t
            new_from, new_to = max(g.from_t, lo), min(g.to_t, hi)
            if (new_from, new_to) != (g.from_t, g.to_t):
                warnings.append(
                    f"{shot.clip_id}: {g.action} {g.from_t:.1f}-{g.to_t:.1f} 距击杀太近,"
                    f"收紧为 {new_from:.1f}-{new_to:.1f}(杀后{buf_after}s/杀前{buf_before}s 硬缓冲)。")
                g.from_t, g.to_t = round(new_from, 3), round(new_to, 3)
            if g.to_t - g.from_t < min_span:
                warnings.append(f"{shot.clip_id}: 收紧后跨度不足 {min_span}s,剔除该处理。")
                continue
            if any(g.from_t - 0.2 < t < g.to_t + 0.2 for t in kills):
                warnings.append(f"{shot.clip_id}: 间隙处理 {g.from_t}-{g.to_t} "
                                f"覆盖了击杀时刻(准则1),已剔除。")
                continue
            if valid_gaps and g.from_t < valid_gaps[-1].to_t:
                warnings.append(f"{shot.clip_id}: 间隙处理重叠,已剔除后者。")
                continue
            valid_gaps.append(g)
        shot.gap_treatments = [g for g in shot.gap_treatments if g.action == "keep"] + valid_gaps
        valid_shots.append(shot)
    sb.shots = valid_shots
    if not sb.shots:
        raise LlmError("分镜表校验后没有任何合法镜头,保留原 EDL 不动。")
    return warnings


# ------------------------------------------------------------------ 分镜 → EDL

def storyboard_to_edl(sb: Storyboard, cards: ScorecardsFile,
                      settings: dict) -> EdlFile:
    """无音乐直拼:cut 拆成多条目,compress 转 speed_spans。"""
    by_id = {c.clip_id: c for c in cards.clips}
    audio_cfg = settings["audio"]
    entries: list[TimelineEntry] = []
    cursor = 0.0
    from ..pipeline.effects import rendered_duration

    for shot in sb.shots:
        card = by_id[shot.clip_id]
        cuts = [g for g in shot.gap_treatments if g.action == "cut"]
        compresses = [g for g in shot.gap_treatments if g.action == "compress"]
        # cut 把镜头切成若干连续区间
        segments: list[tuple[float, float]] = []
        pos = shot.in_t
        for g in sorted(cuts, key=lambda g: g.from_t):
            if g.from_t - pos > 0.2:
                segments.append((pos, g.from_t))
            pos = g.to_t
        if shot.out_t - pos > 0.2:
            segments.append((pos, shot.out_t))
        for si, (a, b) in enumerate(segments):
            spans = [SpeedSpan(from_t=g.from_t, to_t=g.to_t, factor=g.factor)
                     for g in compresses if a <= g.from_t and g.to_t <= b]
            entry = TimelineEntry(
                clip_id=card.clip_id if len(segments) == 1
                else f"{card.clip_id}#{si + 1}",
                source=card.source, in_t=round(a, 3), out_t=round(b, 3),
                timeline_start_t=round(cursor, 4),
                snap=SnapInfo(mode="none", cut_beat_t=None),
                effects=ClipEffects(speed_spans=spans),
                audio=ClipAudio(game_volume=float(audio_cfg["game_volume"])),
            )
            entries.append(entry)
            cursor += rendered_duration(entry)

    render_cfg = settings["render"]
    return EdlFile(
        music=None, target_duration_s=round(cursor, 1),
        global_effects=GlobalEffects(frame_drop=False), timeline=entries,
        render=RenderSettings(codec=render_cfg["codec"],
                              crf_equivalent=int(render_cfg["crf_equivalent"])))


# ------------------------------------------------------------------ 主入口

def direct(project: Project, settings: dict, client: OpenRouterClient,
           style_hint: Optional[str] = None,
           with_frames: bool = True) -> tuple[Storyboard, EdlFile, list[str]]:
    dcfg = settings["director"]
    cards = build_director_cards(project, settings)
    events = project.load(proj.EVENTS_JSON, EventsFile)
    if not any(c.selected for c in cards.clips):
        raise LlmError("没有入选片段可供导演编排。")

    style_line = (f"用户风格提示(优先遵循):{style_hint}" if style_hint
                  else "用户未给风格提示,按默认准则执行。")
    prompt = DIRECTOR_PROMPT.format(style_line=style_line,
                                    clips_block=_clips_block(cards),
                                    buf_after=dcfg["buffer_after_kill_s"],
                                    buf_before=dcfg["buffer_before_kill_s"],
                                    min_span=dcfg["min_span_s"])
    content: list[dict] = [{"type": "text", "text": prompt}]
    if with_frames:
        for b64 in extract_keyframes(project, cards,
                                     int(dcfg["frame_width"]),
                                     int(dcfg["jpeg_quality"])):
            if b64:
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}})

    msg = client.chat([{"role": "user", "content": content}],
                      model=client.director_model)
    text = (msg.get("content") or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LlmError(f"导演输出不是合法 JSON,保留原 EDL 不动。原文: {text[:500]}") from e

    sb = Storyboard.model_validate(data)
    sb.style_hint = style_hint
    sb.model = client.director_model
    sb.rubric_version = RUBRIC_VERSION
    sb.created_at = datetime.datetime.now().isoformat(timespec="seconds")
    sb.reasoning = msg.get("reasoning_content")
    warnings = validate_storyboard(sb, cards, events, settings)

    # 落盘:分镜表(旧版留档)+ EDL + 簇级证据卡 + 数据沉淀日志
    sb_path = project.root / "storyboard.json"
    if sb_path.exists():
        sb_path.rename(project.root / "storyboard.prev.json")
    sb_path.write_text(sb.model_dump_json(indent=2), encoding="utf-8")
    project.save(proj.SCORECARDS_JSON, cards)
    edl = storyboard_to_edl(sb, cards, settings)
    project.save(proj.EDL_JSON, edl)
    with open(project.root / "director_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": sb.created_at, "model": sb.model, "style_hint": style_hint,
            "input_clips": _clips_block(cards),
            "storyboard": sb.model_dump(), "warnings": warnings,
        }, ensure_ascii=False) + "\n")
    return sb, edl, warnings
