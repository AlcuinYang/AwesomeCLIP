"use client";
/** 把 GUI 时间线微调写回 edl.json(PUT /api/edl)。 */
import { useState } from "react";
import { useVmontage } from "@/stores/vmontage-store";

export function SaveEdlButton() {
  const saveEdl = useVmontage((s) => s.saveEdl);
  const connected = useVmontage((s) => s.connected);
  const [status, setStatus] = useState<"idle" | "saving" | "ok" | "err">("idle");

  const onClick = async () => {
    setStatus("saving");
    try {
      await saveEdl();
      setStatus("ok");
    } catch {
      setStatus("err");
    }
    setTimeout(() => setStatus("idle"), 1500);
  };

  return (
    <button
      className="h-7 rounded-md border px-3 text-xs disabled:opacity-50"
      disabled={!connected || status === "saving"}
      onClick={() => void onClick()}
      title="把时间线微调保存回 edl.json"
    >
      {status === "saving" ? "保存中…" : status === "ok" ? "已保存 ✓"
        : status === "err" ? "保存失败" : "保存 EDL"}
    </button>
  );
}
