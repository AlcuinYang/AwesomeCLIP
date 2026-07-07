"use client";
/**
 * NL 聊天面板(规格 P2):底部输入中文指令 → 后端 agent → EDL 更新并重灌时间线。
 */
import { useEffect, useRef, useState } from "react";
import { useVmontage } from "@/stores/vmontage-store";

export function ChatPanel() {
  const { chatLog, chatBusy, sendChat, undo, connected } = useVmontage();
  const [text, setText] = useState("");
  const logRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [chatLog]);

  const submit = () => {
    const t = text.trim();
    if (!t || chatBusy) return;
    setText("");
    if (t === "undo" || t === "撤销") void undo();
    else void sendChat(t);
  };

  return (
    <div className="border-t bg-background">
      <div
        className="flex items-center justify-between px-3 py-1 cursor-pointer select-none"
        onClick={() => setOpen(!open)}
      >
        <span className="text-xs font-medium">
          自然语言剪辑 {connected ? "" : "(后端未连接)"}
        </span>
        <span className="text-xs text-muted-foreground">{open ? "收起" : "展开"}</span>
      </div>
      {open && (
        <div className="px-3 pb-2">
          {chatLog.length > 0 && (
            <div
              ref={logRef}
              className="max-h-32 overflow-y-auto text-xs space-y-1 mb-2 rounded bg-muted/40 p-2"
            >
              {chatLog.map((m, i) => (
                <div
                  key={i}
                  className={
                    m.role === "user"
                      ? "font-medium"
                      : m.role === "op"
                        ? "text-muted-foreground pl-3"
                        : "text-primary"
                  }
                >
                  {m.role === "user" ? "你: " : m.role === "op" ? "· " : "助手: "}
                  {m.text}
                </div>
              ))}
              {chatBusy && <div className="text-muted-foreground">思考中…</div>}
            </div>
          )}
          <div className="flex gap-2">
            <input
              className="flex-1 rounded border bg-transparent px-2 py-1.5 text-xs outline-none focus:border-primary"
              placeholder='例:"只要残局和三杀以上的片段"、"总长压到一分钟"、"undo"'
              value={text}
              disabled={chatBusy}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.nativeEvent.isComposing) submit();
              }}
            />
            <button
              className="rounded bg-primary text-primary-foreground px-3 py-1.5 text-xs disabled:opacity-50"
              disabled={chatBusy || !text.trim()}
              onClick={submit}
            >
              执行
            </button>
            <button
              className="rounded border px-3 py-1.5 text-xs disabled:opacity-50"
              disabled={chatBusy}
              onClick={() => void undo()}
              title="撤销上一步 agent 操作"
            >
              撤销
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
