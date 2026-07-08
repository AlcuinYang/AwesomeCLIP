"""vmontage CLI(规格 §5.9)。

  vmontage init <dir> [--sources DIR] [--music FILE]   # 建项目(可顺带 ingest)
  vmontage ingest <sources_dir> [--music FILE]         # 导入素材/BGM,后台生成 proxy
  vmontage calibrate <frame> [--kind ...]              # 校准 ROI / 截取模板
  vmontage detect                                      # → events.json + scorecards.json
  vmontage analyze-music <mp3>                         # → beats.json
  vmontage auto-cut [--target 60] [--anchor-align]     # → edl.json
  vmontage preview / render                            # → output/
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from .config import load_roi_profile, load_settings
from .pipeline import project as proj
from .pipeline.project import Project
from .schemas.models import BeatsFile, EdlFile, EventsFile, ScorecardsFile

app = typer.Typer(help="Valorant 高光集锦工具 — 检测/卡点/渲染 CLI", no_args_is_help=True)

_project_opt = typer.Option(None, "--project", "-p", help="项目目录(默认从 cwd 向上查找)")


def _get_project(project: Optional[Path]) -> Project:
    return Project(project) if project else Project.find()


def _warn(warnings: list[str]) -> None:
    for w in warnings:
        typer.secho(f"[警告] {w}", fg=typer.colors.YELLOW)


@app.command()
def init(directory: Path,
         sources: Optional[Path] = typer.Option(None, help="素材目录,给了就顺带 ingest"),
         music: Optional[Path] = typer.Option(None, help="BGM 文件")):
    """创建项目目录结构。"""
    p = Project.init(directory)
    typer.echo(f"项目已创建: {p.root}")
    if sources:
        _ingest(p, [sources], music)


def _ingest(p: Project, sources: list[Path], music: Optional[Path]):
    from .pipeline.ingest import ingest as run_ingest
    settings = load_settings(p.root)
    result = run_ingest(p, sources, music, settings)
    typer.echo(f"已导入 {len(result.sources)} 段素材"
               + (f",BGM: {result.music.name}" if result.music else ""))
    if result.proxy_procs:
        typer.echo(f"{len(result.proxy_procs)} 个 720p proxy 正在后台生成(不阻塞检测)。")


@app.command()
def ingest(sources: list[Path] = typer.Argument(..., help="视频文件或目录,可传多个"),
           music: Optional[Path] = typer.Option(None, help="BGM 文件"),
           project: Optional[Path] = _project_opt):
    """导入素材(多个文件/目录均可)与 BGM;后台生成 proxy。"""
    _ingest(_get_project(project), sources, music)


@app.command()
def run(sources: list[Path] = typer.Argument(..., help="视频文件或目录,可传多个"),
        music: Optional[Path] = typer.Option(None, "--music", "-m",
                                             help="BGM 文件;省略则无音乐粗剪"),
        target: float = typer.Option(60, "--target", help="目标总时长(秒)"),
        anchor_align: bool = typer.Option(False, "--anchor-align"),
        directory: Optional[Path] = typer.Option(
            None, "--dir", help="项目目录(默认 ./project_<时间戳>)"),
        final: bool = typer.Option(False, "--final", help="直接出成片(默认只出 720p 预览)")):
    """一键全流程:ingest → detect → analyze-music → auto-cut → preview。"""
    import datetime

    root = directory or Path.cwd() / (
        "project_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    p = Project.init(root)
    typer.echo(f"项目: {p.root}")
    _ingest(p, sources, music)
    detect(project=p.root)
    if music is not None:
        analyze_music(music, project=p.root)
    auto_cut(target=target, anchor_align=anchor_align, project=p.root)
    _render(p.root, preview=True)
    if final:
        _render(p.root, preview=False)
    typer.echo("完成。微调后可在项目目录里执行 vmontage chat / render。")


@app.command()
def calibrate(frame: Path = typer.Argument(..., help="样例帧图片,或直接给录像(配 --at)"),
              at: Optional[float] = typer.Option(
                  None, "--at", help="输入为视频时,抽取该时间点(秒)的画面"),
              kind: Optional[str] = typer.Option(
                  None, help="kill | death | headshot | round-won | round-lost;"
                             "留空则只输出 ROI 标注图"),
              sub: Optional[str] = typer.Option(
                  None, help="ROI 内相对子区域 x,y,w,h,截更紧的模板"),
              project: Optional[Path] = _project_opt):
    """从样例帧(或录像抽帧)校准 ROI / 截取模板资产。"""
    from .pipeline.calibrate import calibrate as run_calibrate
    sub_t = tuple(float(x) for x in sub.split(",")) if sub else None
    proj_dir = project or _try_project_root()
    out = run_calibrate(frame, kind, sub_t, proj_dir, at_s=at)
    typer.echo(f"已输出: {out}")


def _try_project_root() -> Optional[Path]:
    try:
        return Project.find().root
    except FileNotFoundError:
        return None


@app.command()
def detect(project: Optional[Path] = _project_opt):
    """L1 检测 + L2 语义 + 打分 → events.json + scorecards.json。"""
    from .pipeline.detector import Detector
    from .pipeline.ffmpeg_utils import ffprobe_meta
    from .pipeline.scorer import score_clips
    from .pipeline.semantic import build_scorecards

    p = _get_project(project)
    settings = load_settings(p.root)
    videos = sorted(f for f in (p.root / "sources").iterdir()
                    if f.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"})
    if not videos:
        raise typer.BadParameter("sources/ 为空,先运行 vmontage ingest。")

    t0 = time.time()
    events = EventsFile()
    for v in videos:
        meta = ffprobe_meta(v)
        profile = load_roi_profile(meta["width"], meta["height"], settings, p.root)
        det = Detector(settings, profile, project_templates_dir=p.root / "templates")
        _warn(det.warnings)
        typer.echo(f"检测 {v.name} ({meta['width']}x{meta['height']}, "
                   f"{meta['duration_s']:.0f}s)...")
        src_events = det.detect(v, f"sources/{v.name}")
        kills = sum(1 for e in src_events.events if e.type == "kill")
        typer.echo(f"  → {len(src_events.events)} 个事件(kill x{kills})")
        events.sources.append(src_events)
    p.save(proj.EVENTS_JSON, events)

    cards = score_clips(build_scorecards(events, settings), settings)
    p.save(proj.SCORECARDS_JSON, cards)
    sel = sum(1 for c in cards.clips if c.selected)
    typer.echo(f"完成({time.time() - t0:.1f}s):{len(cards.clips)} 个片段,"
               f"{sel} 个入选 → events.json / scorecards.json")


@app.command("analyze-music")
def analyze_music(music: Path, project: Optional[Path] = _project_opt):
    """BGM 节拍分析 → beats.json。"""
    from .pipeline.beat import analyze_music as run_analyze
    from .pipeline.ingest import _link_into

    p = _get_project(project)
    settings = load_settings(p.root)
    music = Path(music)
    if music.resolve().parent != (p.root / "music").resolve():
        music = _link_into(music, p.root / "music")
    beats = run_analyze(p.resolve_media(f"music/{music.name}"),
                        f"music/{music.name}", settings)
    p.save(proj.BEATS_JSON, beats)
    typer.echo(f"BPM {beats.bpm},{len(beats.beats_t)} 个拍点 → beats.json")


@app.command("auto-cut")
def auto_cut(target: float = typer.Option(60, "--target", help="目标总时长(秒)"),
             anchor_align: bool = typer.Option(False, "--anchor-align",
                                               help="高级模式:击杀帧对齐拍点"),
             project: Optional[Path] = _project_opt):
    """按拍点吸附生成剪辑决策表 → edl.json。"""
    from .pipeline.align import build_edl

    p = _get_project(project)
    settings = load_settings(p.root)
    cards = p.load(proj.SCORECARDS_JSON, ScorecardsFile)
    beats = p.load(proj.BEATS_JSON, BeatsFile) if p.exists(proj.BEATS_JSON) else None
    if beats is None:
        typer.secho("无 beats.json → 无音乐粗剪模式(片段直拼,不卡点;"
                    "之后 analyze-music + auto-cut 或 chat 'set_music' 可补歌)",
                    fg=typer.colors.YELLOW)
    events = p.load(proj.EVENTS_JSON, EventsFile)
    edl, warnings = build_edl(cards, beats, events, settings, target,
                              anchor_align=anchor_align)
    _warn(warnings)
    p.save(proj.EDL_JSON, edl)
    total = sum(e.out_t - e.in_t for e in edl.timeline)
    typer.echo(f"{len(edl.timeline)} 个片段,总长 {total:.1f}s → edl.json")


def _render(project: Optional[Path], preview: bool):
    from .pipeline.render import render as run_render

    p = _get_project(project)
    settings = load_settings(p.root)
    edl = p.load(proj.EDL_JSON, EdlFile)
    name = "preview.mp4" if preview else "final.mp4"
    t0 = time.time()
    out, warnings = run_render(p, edl, p.root / "output" / name, settings,
                               preview=preview)
    _warn(warnings)
    typer.echo(f"渲染完成({time.time() - t0:.1f}s): {out}")


@app.command()
def chat(instruction: Optional[str] = typer.Argument(
             None, help="一条中文剪辑指令;留空进入交互模式"),
         project: Optional[Path] = _project_opt):
    """自然语言剪辑:LLM function calling 操作 EDL(需 OPENROUTER_API_KEY)。"""
    from .agent.dsl import AgentSession
    from .agent.llm import LlmError, OpenRouterClient, run_agent

    p = _get_project(project)
    settings = load_settings(p.root)
    try:
        client = OpenRouterClient(settings)
    except LlmError as e:
        raise typer.BadParameter(str(e))

    def once(text: str):
        session = AgentSession(p, settings)  # 每条指令重读 JSON,吃到手工改动
        summary = run_agent(session, text, client,
                            on_event=lambda s: typer.secho(f"  {s}", fg=typer.colors.CYAN))
        typer.echo(summary)

    if instruction:
        once(instruction)
        return
    typer.echo("交互模式(空行或 exit 退出;'undo' 撤销上一步):")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("exit", "quit"):
            break
        once(line)


@app.command()
def direct(style: Optional[str] = typer.Argument(
               None, help='风格提示,如 "爆发开场不要铺垫";留空用默认导演准则'),
           no_frames: bool = typer.Option(False, "--no-frames",
                                          help="不送关键帧(纯文本证据,更便宜)"),
           render_after: bool = typer.Option(True, "--render/--no-render",
                                             help="完成后直接渲染 720p 预览"),
           project: Optional[Path] = _project_opt):
    """MLLM 导演编排:段内间隙裁决(keep/compress/cut)→ storyboard.json + edl.json。"""
    from .agent.director import direct as run_direct
    from .agent.llm import LlmError, OpenRouterClient

    p = _get_project(project)
    settings = load_settings(p.root)
    try:
        client = OpenRouterClient(settings)
        sb, edl, warnings = run_direct(p, settings, client, style_hint=style,
                                       with_frames=not no_frames)
    except LlmError as e:
        raise typer.BadParameter(str(e))
    _warn(warnings)
    typer.secho("分镜表(storyboard.json,思考全文在 reasoning 字段):",
                fg=typer.colors.GREEN)
    for i, shot in enumerate(sb.shots, 1):
        typer.echo(f"  镜头{i} {shot.clip_id} {shot.in_t:.1f}~{shot.out_t:.1f}s"
                   f" — {shot.rationale}")
        for g in shot.gap_treatments:
            mark = (f"压缩x{g.factor:g}" if g.action == "compress"
                    else "剪掉" if g.action == "cut" else "保留")
            typer.echo(f"    · 间隙 {g.from_t:.1f}~{g.to_t:.1f}s [{mark}] {g.rationale}")
    for r in sb.rejected:
        typer.echo(f"  弃用 {r.clip_id} — {r.reason}")
    total = edl.target_duration_s
    typer.echo(f"{len(edl.timeline)} 个时间线条目,预计成片 {total:.1f}s → edl.json")
    if render_after:
        _render(p.root, preview=True)


@app.command()
def undo(project: Optional[Path] = _project_opt):
    """撤销 agent 的上一步操作(从 agent_log.jsonl 回滚)。"""
    from .agent.dsl import AgentSession, DslError

    p = _get_project(project)
    settings = load_settings(p.root)
    try:
        typer.echo(AgentSession(p, settings).undo())
    except DslError as e:
        raise typer.BadParameter(str(e))


@app.command()
def narrate(project: Optional[Path] = _project_opt):
    """L3 叙事:为每个片段生成一句中文描述,写回 scorecards.json。"""
    from .agent.llm import LlmError, OpenRouterClient, narrate as run_narrate

    p = _get_project(project)
    settings = load_settings(p.root)
    try:
        client = OpenRouterClient(settings)
    except LlmError as e:
        raise typer.BadParameter(str(e))
    count = run_narrate(p, settings, client)
    typer.echo(f"已为 {count} 个片段生成叙述。" if count else
               "没有生成任何叙述(可能全部已有,或调用失败——失败不阻塞)。")


@app.command()
def preview(project: Optional[Path] = _project_opt):
    """渲染 720p 预览版 → output/preview.mp4。"""
    _render(project, preview=True)


@app.command()
def render(project: Optional[Path] = _project_opt):
    """渲染成片 → output/final.mp4。"""
    _render(project, preview=False)


@app.command()
def verify(project: Optional[Path] = _project_opt):
    """击杀 node 人工核对表:每个检出事件截图拼图(防漏杀质检)。"""
    from .pipeline.verify import build_verify_sheets

    p = _get_project(project)
    events = p.load(proj.EVENTS_JSON, EventsFile)
    kills = sum(1 for s in events.sources for e in s.events if e.type == "kill")
    paths, warnings = build_verify_sheets(p, events, p.root / "verify")
    _warn(warnings)
    typer.echo(f"共 {kills} 个击杀 node → {len(paths)} 张核对图:")
    for path in paths:
        typer.echo(f"  {path}")
    typer.echo("逐格核对:每格应是一行'我'参与的信息流;发现漏杀可对照录像"
               "手工补 events.json 后重跑 auto-cut(无需重新 detect 其他素材)。")


@app.command("export-vlm-dataset")
def export_vlm_dataset(
        out: Optional[Path] = typer.Option(None, "--out", help="输出目录(默认 项目/dataset)"),
        project: Optional[Path] = _project_opt):
    """P3:规则引擎标注 → VLM 微调数据集(images + labels.jsonl + sft.jsonl)。"""
    from .pipeline.export_dataset import export_dataset

    p = _get_project(project)
    settings = load_settings(p.root)
    events = p.load(proj.EVENTS_JSON, EventsFile)
    out_dir = out or (p.root / "dataset")
    stats = export_dataset(p, events, out_dir, settings)
    typer.echo(f"train {stats['train']} / test {stats['test']} 样本 → {out_dir}"
               + (f"(跳过不可读帧 {stats['skipped_unreadable']})"
                  if stats["skipped_unreadable"] else ""))
    typer.echo("训练方案见 docs/P3.md。")


@app.command()
def serve(port: int = typer.Option(8765, help="监听端口"),
          host: str = typer.Option("127.0.0.1", help="监听地址"),
          project: Optional[Path] = _project_opt):
    """启动 GUI 后端(REST + WebSocket)。"""
    import uvicorn

    from .api.server import create_app

    p = _get_project(project)
    typer.echo(f"vmontage 后端: http://{host}:{port}(项目 {p.root})")
    uvicorn.run(create_app(p.root), host=host, port=port, log_level="warning")


@app.command()
def gui(port: int = typer.Option(8765, help="后端端口"),
        ui_port: int = typer.Option(3000, help="前端端口"),
        project: Optional[Path] = _project_opt):
    """一键启动 GUI:后端 + 前端同时拉起,Ctrl+C 一起退出。"""
    import shutil
    import subprocess

    import uvicorn

    from .api.server import create_app

    p = _get_project(project)
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    pnpm = shutil.which("pnpm")
    if not pnpm:
        raise typer.BadParameter("未找到 pnpm(前端需要):npm install -g pnpm")
    if not (frontend / "node_modules").exists():
        typer.echo("首次运行,安装前端依赖(约 1-2 分钟)...")
        subprocess.run([pnpm, "install"], cwd=frontend, check=True)

    import os
    import signal

    # 独立进程组:退出时整组终止(pnpm 会再派生 next-server 子进程)
    ui = subprocess.Popen([pnpm, "dev", "--port", str(ui_port)], cwd=frontend,
                          start_new_session=True)
    typer.echo(f"项目: {p.root}")
    typer.echo(f"后端: http://127.0.0.1:{port}")
    typer.secho(f"打开 → http://localhost:{ui_port}", fg=typer.colors.GREEN, bold=True)
    from contextlib import suppress
    import sys
    # SIGTERM 默认会直接杀死进程跳过 finally;转成 SystemExit 保证前端被清理
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        uvicorn.run(create_app(p.root), host="127.0.0.1", port=port,
                    log_level="warning")
    finally:
        with suppress(ProcessLookupError):
            os.killpg(ui.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            ui.wait(timeout=4)
        with suppress(ProcessLookupError):
            os.killpg(ui.pid, signal.SIGKILL)  # 兜底:next-server 偶尔无视 SIGTERM


if __name__ == "__main__":
    app()
