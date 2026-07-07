# DECISIONS.md — 规格未覆盖决策记录

按规格 v1 约定:本文记录规格未覆盖处的实现决策,原则是选最简单的可用实现。

| # | 决策 | 理由 |
|---|------|------|
| DX1 | 仓库根即 vmontage 根(`/Volumes/Storage/ValClip`),包名 `backend`,console script `vmontage = backend.cli:app` | 与规格 §8 目录结构一字不差,免一层嵌套 |
| DX2 | `events.json` 顶层为 `{"sources": [SourceEvents...]}`(多素材列表) | 规格 §4.1 只给了单素材对象;一次会话有多段素材,列表是最简扩展 |
| DX3 | 新增 `vmontage ingest` 子命令;`init --sources --music` 可顺带导入 | 规格 CLI 表未给素材导入入口,但 §5.1 要求 ingest 阶段 |
| DX4 | multikill 标签命名 `multikill_N`(N=2..5) | §5.3 规范性文字用 `multikill_{2..5}`;§4.2 示例的 `triple_kill` 视为示意 |
| DX5 | 缺模板资产时 detector 降级 HSV-only:横幅全部记为 kill、confidence=0.6、round_end 不产出,并显式告警 | calibrate 前也能先跑通流程;不静默——每次运行都会打警告 |
| DX6 | 多杀在横幅持续期的再触发:HSV 命中面积相对滚动最小值跳升 ≥1.6× 记为新击杀 | 横幅持续 ~4s,纯上升沿会漏掉窗口内连杀;面积跳升=新横幅行叠加,是无模板下最简单的近似 |
| DX7 | 拍点吸附产生的空隙由上一片段延长 out_t 填补(素材富余画面),延长不够时切点不吸附并告警 | 渲染是顺序拼接,时间线不允许空洞;§5.6"出点不做吸附"理解为出点不主动找拍点,填缝延长是被动行为 |
| DX8 | 第一个片段固定从时间线 0 开始,不吸附 | 音乐与时间线同起点,第一个切换点才需要对拍 |
| DX9 | `downbeats_t = beats_t[::4]` 朴素近似,不参与对齐 | §5.5 明确 V1 不做 downbeat 检测,但 §4.3 契约有该字段,用近似填充保持契约完整 |
| DX10 | 主扫描用 ffmpeg rawvideo 管道按 sample_fps 出帧;精定位/光流用 cv2.VideoCapture 随机访问 | 管道顺序读最稳;cv2 seek 只在 ±0.5s 小窗口内用 |
| DX11 | speed_ramp 慢放会增加片段实际时长,增量由 `effects.clip_extra_duration` 显式计算,渲染端音频按增量补齐 | 慢放物理上必然改变时长;默认关,开启后其后切点偏离拍点属已知代价(§9 风险) |
| DX12 | flick 光流角速度取特征点中位位移幅值(抗前景干扰),像素→角度用水平 FOV 103° 线性换算 | 最简单可用;阈值已暴露在配置,留待真实素材标定 |
| DX13 | 存活判定:记分区每侧按 alive_cells=5 均分成列,列平均饱和度 < 阈值视为灰化阵亡 | 头像格灰化检测的最简实现;格数/阈值均在配置 |
| DX14 | 渲染输出分辨率 "source" = 首个时间线片段的源分辨率,其余片段 scale+pad 对齐;帧率同理归一 | 混分辨率素材时必须归一;首片段作基准最简单 |
| DX15 | P1 的 LLM 接入按用户更新用 OpenRouter API(非 Anthropic 直连);P0 不含 agent 代码 | 用户在交付说明中更新了此项;P0 范围不受影响 |
| DX16 | 开发机为 macOS 无 NVENC,渲染自动回退 libx264 并警告;NVENC 路径代码就绪,到用户 Windows/NVIDIA 机器上即生效 | 规格 §5.8 的回退策略本来就要求这样 |
| DX17 | agent_log.jsonl 每行内嵌操作前的 edl+scorecards 完整快照;undo=弹出最后一行并恢复快照 | JSON 很小,快照比逆操作简单可靠;回放(按序重放 op+args)与撤销同时满足 |
| DX18 | DSL 筛选字段(tags_any/min_score/min_multikill/min_clutch_enemies/has_flick)之间为 AND;"残局或三杀以上"这类 OR 语义由 LLM 拆成 select only + select add 两次调用 | 单次调用语义简单无歧义,组合表达力交给 LLM |
| DX19 | 改动切点的 DSL 操作(select/reorder/trim/align/fit_duration/set_music/map_section)自动重跑 place_clips 重吸附 | 保证任何 NL 编辑后切换点仍落拍(验收 #4 不因编辑而破坏) |
| DX20 | map_section 实现:成片均分为 n 段,段均 RMS 排名与片段分数排名对位(最高分对最高能量段) | 能量映射的最简可用实现;策略字段留了扩展位 |
| DX21 | undo 本身不写日志行(物理弹出);失败的操作(DslError)不落盘不写日志 | 日志始终等于"已生效操作序列",回放语义干净 |
| DX22 | LLM 经 OpenRouter(OpenAI 兼容协议 + httpx),API key 走 OPENROUTER_API_KEY 环境变量;模型名在 settings.yaml agent 节 | 用户指定 OpenRouter;裸 httpx 避免多拉一个 SDK 依赖 |
| DX23 | L3 叙事单次批量调用(全部证据卡进一个 prompt,JSON 出),任何失败静默保持 narration=null | 规格 §4.2 要求不阻塞;批量调用省轮次 |
| DX24 | `vmontage chat` 每条指令新建 AgentSession(重读全部 JSON) | 用户可能在两条指令之间手工改 JSON,重读保证一致 |
| DX25 | OpenVideo 以 vendored copy 进 `frontend/`(上游 designcombo/react-video-editor@9a8c529),非 GitHub fork/submodule | 本机无法替用户建 GitHub fork;拷贝 + 记录上游 hash 是最简单可用方案,以后可随时转 fork |
| DX26 | 前端集成层集中在 `src/lib/vmontage.ts` + `src/stores/vmontage-store.ts` + `src/components/editor/vmontage/`,对上游文件只做 5 处小补丁(editor/header/right-panel/ruler + .env.local) | 补丁面越小,以后合并上游更新越容易 |
| DX27 | EDL→工程画布固定 1280x720(proxy 分辨率),浏览器只播 proxy;工程→EDL 只回写 in/out/timeline_start/音量,被手工挪动的切点 snap 置 none | 画布只是预览,成片分辨率由后端 render 决定;snap 字段不再撒谎 |
| DX28 | GUI 微调不自动落盘:显式"保存 EDL"按钮;chat 指令与导出前自动先保存 | 自动同步每次拖动都写盘太吵;两个必须一致的时机(agent 操作前/渲染前)强制保存 |
| DX29 | uvicorn 必须带 [standard](websockets);裸 uvicorn 会静默拒绝 WS 升级 | 实测踩坑:TestClient 不经过 uvicorn,测试全绿但浏览器连不上 WS |
| DX30 | zustand selector 禁止内联 `?? []` 等新引用兜底(返回稳定引用,兜底放组件层) | 实测踩坑:SSR getServerSnapshot 每次新数组 → 无限循环 |
| DX31 | Kimi K2.6 跑 agent 必须保持思考模式开启;关思考实测做错"其余按时间顺序"类推理指令 | 真实 LLM 验收发现;system prompt 同时补了"时间顺序=span 起点升序"规则 |
| DX32 | ingest 接受文件/目录混合多输入(按文件名去重);新增 `vmontage run` 一键流程命令 | 用户实际使用是一次丢多个录像;run 对应"素材到成片 ≤5min"的产品目标 |
| DX33 | calibrate 接受视频 + `--at 秒`直接抽帧(ffmpeg 无损 png,存视频旁) | 用户只有录像没有截图;抽帧产物保留便于反复截模板 |
| DX34 | P3 数据导出:正样本=事件时刻+横幅持续期,负样本距事件 ≥3s,确定性哈希每 5 个留 1 个进一致性测试集;LoRA 训练本身需 GPU,方案见 docs/P3.md | 规格 §7 的"规则可处理片段留出为一致性测试集";本机无 GPU 只能交付数据侧 |
| DX35 | **检测器 v2(真素材驱动的架构修正)**:kill/death 主信号改为右上击杀信息流中"我"参与行的金色高亮空心边框;底部中央横幅方案废弃 | 真素材验证:底部区域是技能图标/手部动画的重灾区,HSV 横幅方案误报严重;金框是官方 UI 的自我标识,空心环形状可与金发头像区分,且无需任何模板 |
| DX36 | 金框去抖三件套:TTL 记账(高亮=行寿命约 6s,脉冲发光,计数瞬时回落不算消失)+ 相邻行粘连按典型行高切分 + 精定位用行内容指纹+计数双条件 | 逐一实测:纯上升沿重复计数(同一杀最多记 4 次);行指纹做主判据在快速画面下不可靠(半透明行透背景);先后踩过的坑都固化成合成测试 |
| DX37 | kill_feed ROI 只取信息流上 3 行(y 0.05-0.16),第 4/5 行放弃 | 用户素材开着"网络统计"悬浮窗,恰好压在第 4/5 行上把金框切碎;超长连杀(4 杀+同屏)会漏计,录制时关掉悬浮窗即可根治(README 已写) |
| DX38 | 存活数改为记分区高饱和度头像连通块计数(替代固定五等分);记分区 ROI 依据真素材重标定 | 阵亡头像灰化/消失,固定分格对不齐;连通块计数对头像数量与位置都稳健 |
| DX39 | multikill_N 改为**簇内总击杀数**(取代规格 §5.3 的 8s 滑动窗口口径);merge_gap 5→12s;两杀之间自己被杀则强制断簇 | 用户明确定义:片段内不连续击杀也算(如 1,2,1 分布的四杀,中间没死就是一段四杀高光);死亡是高光的天然边界 |
