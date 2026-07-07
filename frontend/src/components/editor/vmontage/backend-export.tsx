"use client";
/**
 * 后端导出按钮(规格 P2:浏览器导出弃用,改调后端 render)。
 * 先出 720p 预览(秒级),确认后出成片;进度经 WebSocket 推送。
 */
import { useState } from "react";
import { mediaUrl } from "@/lib/vmontage";
import { useVmontage } from "@/stores/vmontage-store";

export function BackendExport() {
  const { renderProgress, lastRenderPath, error, startRender } = useVmontage();
  const [menuOpen, setMenuOpen] = useState(false);
  const busy = renderProgress !== null;

  return (
    <div className="relative">
      <button
        className="rounded bg-primary text-primary-foreground px-3 py-1.5 text-xs disabled:opacity-60"
        disabled={busy}
        onClick={() => setMenuOpen(!menuOpen)}
      >
        {busy ? `渲染中 ${Math.round((renderProgress ?? 0) * 100)}%` : "导出"}
      </button>
      {menuOpen && !busy && (
        <div className="absolute right-0 top-full mt-1 z-50 w-44 rounded border bg-background shadow-lg p-1">
          <button
            className="w-full text-left rounded px-2 py-1.5 text-xs hover:bg-muted"
            onClick={() => {
              setMenuOpen(false);
              void startRender(true);
            }}
          >
            720p 预览(快)
          </button>
          <button
            className="w-full text-left rounded px-2 py-1.5 text-xs hover:bg-muted"
            onClick={() => {
              setMenuOpen(false);
              void startRender(false);
            }}
          >
            成片(NVENC)
          </button>
          {lastRenderPath && (
            <a
              className="block rounded px-2 py-1.5 text-xs text-primary hover:bg-muted"
              href={mediaUrl(lastRenderPath)}
              target="_blank"
              rel="noreferrer"
            >
              打开上次导出 ↗
            </a>
          )}
        </div>
      )}
      {error && (
        <div className="absolute right-0 top-full mt-1 z-50 w-64 rounded border border-red-500/50 bg-background p-2 text-[11px] text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
