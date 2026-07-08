"""OpenRouter 接入:NL 指令 → function calling → DSL(P1),以及 L3 叙事层。

- API key 从环境变量 OPENROUTER_API_KEY 读取;模型/端点在 settings.yaml 的 agent 节。
- run_agent:多轮工具循环,LLM 每次 tool call 都落到 AgentSession.apply(自动写
  agent_log.jsonl);DslError 作为工具结果回传,LLM 可自行修正参数重试。
- narrate:基于证据卡生成一句中文叙述,写回 scorecards.json 的 narration;
  任何失败都不阻塞(规格 §4.2)。
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import httpx

from ..pipeline import project as proj
from .dsl import TOOLS, AgentSession, DslError


class LlmError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, settings: dict, api_key: Optional[str] = None,
                 transport: Optional[httpx.BaseTransport] = None):
        cfg = settings["agent"]
        self.base_url = str(cfg["base_url"]).rstrip("/")
        self.model = str(cfg["model"])
        self.narration_model = str(cfg.get("narration_model", cfg["model"]))
        # 部分厂商(如 Kimi K2.6/K2.5)不接受自定义 temperature,配 null 则不发送
        self.temperature = cfg.get("temperature")
        # 厂商特有的额外请求体,如 Kimi 的 {"thinking": {"type": "disabled"}}
        self.extra_body: dict = cfg.get("extra_body") or {}
        key_env = str(cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        self.api_key = api_key or os.environ.get(key_env)
        if not self.api_key:
            raise LlmError(f"缺少 {key_env} 环境变量(见 settings.yaml 的 agent.api_key_env)。")
        # 导演 pass(思考模式+多图)可能超过 2 分钟,超时给足
        self._http = httpx.Client(timeout=float(cfg.get("timeout_s", 600)),
                                  transport=transport)

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             model: Optional[str] = None) -> dict:
        """返回 choices[0].message。"""
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            **self.extra_body,
        }
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self._http.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        if resp.status_code != 200:
            raise LlmError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LlmError(f"响应格式异常: {json.dumps(data)[:500]}") from e


SYSTEM_PROMPT = """你是视频剪辑助手,通过调用工具编辑一条 Valorant 高光集锦的时间线。

规则:
- 用户用中文口语下指令;把它翻译成一或多次工具调用,不要凭空编造 clip_id。
- "第 N 个片段"指时间线序号 N(1 起);"最后一个"用 'last'。
- "按时间顺序"指按片段在素材里的发生时间排序,即片段列表中 span 的起点升序,
  与当前时间线顺序无关。
- 工具返回错误时,根据错误信息修正参数重试,不要重复同样的失败调用。
- 所有改动完成后,用一两句中文总结你做了什么。不要输出多余的客套话。

当前项目状态:

{context}"""


def run_agent(session: AgentSession, instruction: str,
              client: OpenRouterClient,
              on_event: Optional[Callable[[str], None]] = None) -> str:
    """执行一条自然语言指令,返回 LLM 的中文总结。"""
    notify = on_event or (lambda s: None)
    max_turns = int(session.settings["agent"]["max_turns"])
    messages: list[dict] = [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(context=session.context_summary())},
        {"role": "user", "content": instruction},
    ]
    for _ in range(max_turns):
        msg = client.chat(messages, tools=TOOLS)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content") or "(无输出)"
        # 原样透传 assistant 消息:思考型模型(如 Kimi K2.6)要求多步工具调用时
        # 上下文里保留 reasoning_content,重组字典会把它丢掉导致报错
        messages.append({**msg, "role": "assistant"})
        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = session.apply(fn, args)
                notify(f"[{fn}] {result}")
            except DslError as e:
                result = f"错误: {e}"
                notify(f"[{fn}] {result}")
            messages.append({"role": "tool",
                             "tool_call_id": tc.get("id", fn),
                             "content": result})
        # 状态已变,刷新上下文让后续调用基于最新时间线
        messages[0]["content"] = SYSTEM_PROMPT.format(
            context=session.context_summary())
    return "达到最大工具调用轮数,操作可能未全部完成;请检查时间线。"


NARRATE_PROMPT = """以下是 Valorant 高光片段的证据卡(JSON)。为每个片段写一句简短的中文
叙述(20 字以内,电竞解说风格,基于证据不要编造)。只输出 JSON 对象,键为 clip_id,
值为叙述字符串,不要输出其他内容。

{cards}"""


def narrate(project, settings: dict, client: OpenRouterClient) -> int:
    """L3 叙事:为缺 narration 的片段生成一句话,写回 scorecards.json。

    返回成功填充的数量;单点失败静默跳过,不阻塞流程。
    """
    from ..schemas.models import ScorecardsFile

    cards = project.load(proj.SCORECARDS_JSON, ScorecardsFile)
    pending = [c for c in cards.clips if c.narration is None]
    if not pending:
        return 0
    payload = [{"clip_id": c.clip_id, "tags": c.tags, "score": c.score.total,
                "evidence": c.evidence.model_dump()} for c in pending]
    try:
        msg = client.chat(
            [{"role": "user",
              "content": NARRATE_PROMPT.format(
                  cards=json.dumps(payload, ensure_ascii=False))}],
            model=client.narration_model,
        )
        text = (msg.get("content") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        mapping = json.loads(text)
    except (LlmError, json.JSONDecodeError, httpx.HTTPError):
        return 0  # 叙事失败不阻塞,narration 保持 null
    count = 0
    for c in pending:
        value = mapping.get(c.clip_id)
        if isinstance(value, str) and value.strip():
            c.narration = value.strip()
            count += 1
    if count:
        project.save(proj.SCORECARDS_JSON, cards)
    return count
