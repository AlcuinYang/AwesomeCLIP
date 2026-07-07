"use client";
/**
 * 证据卡侧栏(规格 P2):点击时间线片段 → 显示 tags / evidence / score / narration。
 * 未选中片段时显示全部片段总览。
 */
import { useMemo } from "react";
import { useStudioStore } from "@/stores/studio-store";
import { useVmontage } from "@/stores/vmontage-store";
import type { ScoreCard } from "@/lib/vmontage";

const TAG_LABEL: Record<string, string> = {
  flick: "神经枪",
};

function tagLabel(tag: string): string {
  if (TAG_LABEL[tag]) return TAG_LABEL[tag];
  const mk = tag.match(/^multikill_(\d+)$/);
  if (mk) return `${["", "", "双", "三", "四", "五"][+mk[1]] ?? mk[1]}杀`;
  const cl = tag.match(/^clutch_1v(\d+)$/);
  if (cl) return `残局 1v${cl[1]}`;
  return tag;
}

function Chip({ text }: { text: string }) {
  return (
    <span className="inline-block rounded bg-primary/15 text-primary px-1.5 py-0.5 text-[11px] mr-1 mb-1">
      {text}
    </span>
  );
}

function CardDetail({ card }: { card: ScoreCard }) {
  return (
    <div className="space-y-3 text-xs">
      <div>
        <div className="text-sm font-medium mb-1">{card.clip_id}</div>
        <div className="text-muted-foreground">
          {card.span.start_t.toFixed(1)}s ~ {card.span.end_t.toFixed(1)}s ·{" "}
          {card.source.replace(/^sources\//, "")}
        </div>
      </div>
      {card.narration && (
        <div className="rounded bg-muted p-2 leading-relaxed">{card.narration}</div>
      )}
      <div>{card.tags.map((t) => <Chip key={t} text={tagLabel(t)} />)}</div>
      <div>
        <div className="font-medium mb-1">
          评分 {card.score.total.toFixed(1)}
        </div>
        {Object.entries(card.score.breakdown).map(([k, v]) => (
          <div key={k} className="flex justify-between text-muted-foreground">
            <span>{k}</span>
            <span>+{v.toFixed(1)}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="font-medium mb-1">击杀 × {card.evidence.kills.length}</div>
        {card.evidence.kills.map((k, i) => (
          <div key={i} className="flex justify-between text-muted-foreground">
            <span>
              {k.t.toFixed(2)}s {k.headshot ? "爆头" : ""}
            </span>
            <span>
              {k.pre_kill_angular_velocity_deg_s != null
                ? `${k.pre_kill_angular_velocity_deg_s.toFixed(0)}°/s`
                : ""}
            </span>
          </div>
        ))}
      </div>
      {card.evidence.alive_state && (
        <div className="text-muted-foreground">
          残局状态:{card.evidence.alive_state.ally} v{" "}
          {card.evidence.alive_state.enemy}({card.evidence.alive_state.t.toFixed(1)}s)
        </div>
      )}
      {card.evidence.round_won && (
        <div className="text-muted-foreground">
          回合胜利 @ {card.evidence.round_won.t.toFixed(1)}s
        </div>
      )}
    </div>
  );
}

const NO_CARDS: ScoreCard[] = []; // selector 必须返回稳定引用,内联 ?? [] 会造成快照死循环

export function EvidencePanel() {
  const cards = useVmontage((s) => s.state?.scorecards?.clips) ?? NO_CARDS;
  const selectedClips = useStudioStore((s) => s.selectedClips);

  const selectedCard = useMemo(() => {
    const ids = new Set(
      selectedClips.map((c: any) => c?.id ?? c?.name).filter(Boolean),
    );
    return cards.find((c) => ids.has(c.clip_id)) ?? null;
  }, [cards, selectedClips]);

  if (!cards.length) {
    return (
      <div className="p-3 text-xs text-muted-foreground">
        没有检测数据(先跑 vmontage detect)。
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-3">
      {selectedCard ? (
        <CardDetail card={selectedCard} />
      ) : (
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground mb-2">
            点击时间线片段查看证据卡;当前共 {cards.length} 个片段:
          </div>
          {cards.map((c) => (
            <div
              key={c.clip_id}
              className={`rounded border p-2 text-xs ${c.selected ? "" : "opacity-45"}`}
            >
              <div className="flex justify-between">
                <span className="font-medium">{c.clip_id}</span>
                <span>{c.score.total.toFixed(1)} 分</span>
              </div>
              <div className="mt-1">
                {c.tags.map((t) => <Chip key={t} text={tagLabel(t)} />)}
              </div>
              {c.narration && (
                <div className="text-muted-foreground mt-1">{c.narration}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
