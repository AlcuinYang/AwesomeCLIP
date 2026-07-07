# vmontage — Valorant 高光集锦工具

把一批 ShadowPlay 录像自动加工为卡点音乐集锦:规则 CV 检测高光(击杀/多杀/残局/神经枪)、
可解释打分(证据卡)、切换点吸附 BGM 拍点、NVENC 快速导出。

## 安装

```bash
# 需要:Python 3.11+、ffmpeg >= 6.0(NVENC 支持可选,缺失自动回退 libx264)
uv sync            # 或 pip install -e .
```

## 快速开始(一键流程)

```bash
# 多个视频/目录混着传都行;首次会提示先校准(见下)
uv run vmontage run 素材目录/ 另一段.mp4 --music bgm.mp3 --target 60
# → ./project_<时间戳>/output/preview.mp4;满意后进项目目录 vmontage render 出成片
```

### 首次使用:先校准(约 3 分钟,直接用录像不需要截图)

```bash
# 1. 找一段录像里"正在显示击杀横幅"的时刻(比如第 12.3 秒),先看 ROI 框对不对:
uv run vmontage calibrate 素材.mp4 --at 12.3
#    → 输出 *_roi_overview.png,绿框应框住击杀横幅/记分区/结算横幅;不准就调 roi_*.yaml

# 2. 截取模板(每种横幅各找一个时刻):
uv run vmontage calibrate 素材.mp4 --at 12.3 --kind kill
uv run vmontage calibrate 素材.mp4 --at 45.0 --kind death       # 被击杀的时刻
uv run vmontage calibrate 素材.mp4 --at 12.3 --kind headshot --sub 0.60,0.10,0.12,0.60
uv run vmontage calibrate 素材.mp4 --at 88.0 --kind round-won   # 回合胜利结算
uv run vmontage calibrate 素材.mp4 --at 120.0 --kind round-lost
```

不校准也能跑(HSV 降级模式),但方向区分/爆头/回合事件会缺失,检测精度打折。

## 分步命令

```bash
uv run vmontage init my_montage --sources /path/to/shadowplay_clips --music /path/to/bgm.mp3
cd my_montage
# 或增量导入更多素材(文件/目录混合,可多个):
uv run vmontage ingest 新素材A.mp4 新素材目录/

uv run vmontage detect                    # → events.json + scorecards.json
uv run vmontage analyze-music music/bgm.mp3   # → beats.json
uv run vmontage auto-cut --target 60      # → edl.json
uv run vmontage preview                   # → output/preview.mp4 (720p 快速)
uv run vmontage render                    # → output/final.mp4
```

## 自然语言剪辑(P1)

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run vmontage chat "只要残局和三杀以上的片段"
uv run vmontage chat "第二个片段击杀前多留一秒"
uv run vmontage chat            # 交互模式
uv run vmontage undo            # 撤销上一步(agent_log.jsonl 逐条回滚)
uv run vmontage narrate         # L3:给每个片段生成一句中文叙述
```

LLM 走 OpenAI 兼容协议,默认 OpenRouter;每次操作记录在 `agent_log.jsonl`
(含操作前快照),可回放可撤销。所有 NL 编辑后切换点会自动重新吸附拍点。

**测试期省钱**:在项目目录放一个 `settings.yaml` 覆盖 `agent` 节即可。直连 Kimi K2.6:

```yaml
agent:
  base_url: https://api.moonshot.cn/v1
  api_key_env: MOONSHOT_API_KEY
  model: kimi-k2.6
  narration_model: kimi-k2.6
  temperature: null              # K2.6/K2.5 不接受自定义温度,必须置 null
```

该配置已实测通过附录 A 全部五条指令与 L3 叙事。**保持思考模式开启(默认)**:
实测关思考会做错"其余按时间顺序"这类推理型指令。

或仍走 OpenRouter 只换模型名(`moonshotai/kimi-k2`、`deepseek/deepseek-chat` 等)。
注意所选模型必须支持 function calling(tools),否则 chat 指令无法落到 EDL。

所有中间产物都是项目目录下的 JSON,可手工/程序修改后直接重跑下游
(比如改 `edl.json` 里的 `in_t` 后直接 `vmontage render`,无需重新 detect)。

## 录制建议(影响检测精度)

- **关闭"网络统计/客户端FPS"悬浮窗再录制**(Valorant 设置→视频→统计数据全关)。
  该悬浮窗压在击杀信息流第 4/5 行上,会把超长连杀(同屏 4 行以上)的高亮框切碎导致漏计;
  关掉后信息流完整可见,检测查全率显著提升。
- 检测主信号是右上击杀信息流中"我"行的金色高亮框,**无需任何模板校准即可工作**;
  headshot/round-end 检测才需要 calibrate(可选)。

## 已知边界(V1)

- 假设 Valorant 默认 HUD 与默认准星;自定义 HUD 缩放会破坏检测。
- 开着网络统计悬浮窗时,同屏 4 行以上的连杀会漏计(见上;DX37)。
- 支持 1920x1080 / 2560x1440;其他分辨率需复制 `backend/config/roi_*.yaml`
  校准后在 `settings.yaml` 注册。
- Riot 更新 UI 后需重新 `calibrate`。
- flick 角速度阈值(默认 180°/s)需用真实素材标定,见 `settings.yaml`。
- 成片仅供个人使用,BGM 版权自负。

## GUI(P2)

```bash
# 终端 1:后端(REST + WebSocket,默认 127.0.0.1:8765)
uv run vmontage serve -p <项目目录>

# 终端 2:前端(OpenVideo fork,首次先 pnpm install)
cd frontend && pnpm install && pnpm dev
# 打开 http://localhost:3000
```

- 启动即把 `edl.json` 加载进时间线;画布只播 720p proxy(需先跑过 ingest 生成)。
- 右侧证据卡:点击时间线片段查看 tags/evidence/score/narration。
- 底部聊天框:中文指令直达 agent(需 OPENROUTER_API_KEY 随 serve 进程);"撤销"逐条回滚。
- 时间线标尺下缘的琥珀色三角 = BGM 拍点;切换点应吸附其上。
- 顶栏"保存 EDL"把手工微调写回 edl.json;"导出"调后端 render(浏览器导出已弃用),
  进度经 WebSocket 实时显示。
- 前端接后端地址可用 `frontend/.env.local` 的 `NEXT_PUBLIC_VMONTAGE_API` 覆盖。

## 配置

全局默认在 `backend/config/settings.yaml`;项目目录下放同名 `settings.yaml`
可覆盖任意键(深合并)。所有阈值/权重/魔法数字都在配置里。

## P3:VLM 自蒸馏(数据侧已就绪)

```bash
uv run vmontage export-vlm-dataset   # 规则引擎标注 → dataset/{images, labels.jsonl, sft.jsonl}
```

训练方案(Qwen2.5-VL LoRA、一致性测试集、验收标准)见 [docs/P3.md](docs/P3.md);
训练本身需要 GPU 机器。

## 路线图

P0 CLI ✅ → P1 NL agent ✅(Kimi K2.6 实测通过附录 A)→ P2 GUI ✅ →
P3 VLM 自蒸馏(数据导出 ✅,训练待 GPU+素材)。详见开发规格与 `DECISIONS.md`。
