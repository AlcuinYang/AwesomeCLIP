# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 项目概述

**vmontage** — Valorant 高光集锦工具。把 ShadowPlay 录像批量加工为卡点音乐集锦:
规则 CV 检测高光(击杀/多杀/残局/神经枪)→ 可解释打分(证据卡)→ 切换点吸附 BGM
拍点 → 自然语言剪辑(LLM function calling)→ GUI 微调 → NVENC 渲染。

开发规格 v1 是唯一需求来源(用户持有);**规格未覆盖的决策一律选最简单可用实现并记入
`DECISIONS.md`**(现有 DX1–DX30,动代码前先读它,别推翻已记录的决策)。

进度:P0 CLI ✅ → P1 NL agent ✅(Kimi K2.6 实测通过附录 A 五条指令;key 在
gitignored `.env`,思考模式必须开启,见 DX31)→ P2 GUI ✅ → P3 数据导出 ✅
(`export-vlm-dataset`,训练需 GPU,方案 docs/P3.md)。
待外部输入:真实 ShadowPlay 素材(校准 + 检测精度验收 ≥95%)。

## 命令

```bash
uv sync                                # 安装(Python 3.12 由 uv 管理,系统只有 3.9)
uv run pytest -q                       # 全部测试(35 个,含合成素材 E2E,~10s)
uv run vmontage --help                 # CLI:run(一键)/init/ingest(多文件)/calibrate(可从视频
                                       #     --at 抽帧)/detect/analyze-music/auto-cut/preview
                                       #     /render/chat/undo/narrate/serve/export-vlm-dataset
uv run vmontage serve -p <项目目录>     # GUI 后端 (REST+WS, 127.0.0.1:8765)
cd frontend && pnpm dev                # GUI 前端 (Next.js, node22+pnpm9)
```

ffmpeg 8.x 已由 Homebrew 安装;本机(macOS)无 NVENC,渲染自动回退 libx264 并告警,
NVENC 代码路径到用户的 Windows/NVIDIA 机器上即生效。

## 架构

数据流(每阶段读写项目目录下的 JSON,渲染是唯一产生视频的步骤):

```
ingest → detector(L1) → semantic(L2) → scorer → beat → align → render
         events.json     scorecards.json        beats.json  edl.json → output/
```

- `backend/schemas/models.py` — 全部 JSON 契约的 pydantic v2 模型,改契约只能改这里
- `backend/pipeline/` — 八个阶段 + project.py(项目目录/JSON IO)+ ffmpeg_utils.py
  - `align.py` 的 `place_clips()` 是拍点吸附核心,DSL 改单后自动重跑它保证切点落拍
- `backend/agent/dsl.py` — NL 剪辑的十个操作原语;每次操作写 agent_log.jsonl
  (内嵌操作前快照,undo=弹行恢复);失败的操作不落盘不写日志
- `backend/agent/llm.py` — OpenAI 兼容 chat completions(默认 OpenRouter,可直连
  Kimi 等;api_key_env/temperature/extra_body 均在 settings.yaml agent 节)
- `backend/api/server.py` — FastAPI:/api/state /api/edl /api/chat /api/undo
  /api/render + WS 进度推送 + /media 静态挂载
- `backend/config/settings.yaml` — **所有魔法数字/阈值/权重必须住在这里**;
  项目目录同名文件深合并覆盖;ROI 按分辨率在 roi_1080p/1440p.yaml
- `frontend/` — designcombo/react-video-editor 的 vendored copy(上游 @9a8c529);
  **集成代码集中在 `src/lib/vmontage.ts`、`src/stores/vmontage-store.ts`、
  `src/components/editor/vmontage/`,对上游文件只有 5 处小补丁(editor/header/
  right-panel/ruler),保持补丁面最小以便合并上游**
- `tests/` — 合成素材测下限:合成帧/合成视频测检测器,ffmpeg lavfi 视频 + numpy
  点击音轨测 E2E;不依赖真实素材

## 硬约束与已踩的坑

- 禁止 moviepy 参与渲染;ffmpeg 一律 subprocess(ffmpeg_utils.py)
- uvicorn 必须 `uvicorn[standard]`,裸 uvicorn 静默拒绝 WS(TestClient 测不出,DX29)
- zustand selector 禁止内联 `?? []` 等新引用兜底,SSR 会无限循环(DX30)
- Kimi K2.6/K2.5:temperature 必须置 null 不发送;多步工具调用必须透传
  reasoning_content(llm.py 已处理,别改回重组字典的写法)
- 检测器缺模板资产时走 HSV-only 降级(全记 kill、confidence=0.6)并每次告警;
  模板由 `vmontage calibrate <样例帧>` 从真实截图生成
- i18n 无;注释与 CLI 输出用中文
