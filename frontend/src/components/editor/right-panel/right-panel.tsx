"use client";

import { PropertiesPanel } from "./properties-panel";
import { EvidencePanel } from "../vmontage/evidence-panel";

export function RightPanel() {
  return (
    <div className="w-full h-full overflow-hidden flex flex-col border-l">
      {/* vmontage: 证据卡(tags/evidence/score/narration) */}
      <div className="h-1/2 min-h-0 overflow-hidden border-b">
        <div className="px-3 pt-2 text-xs font-medium">证据卡</div>
        <div className="h-[calc(100%-1.75rem)]">
          <EvidencePanel />
        </div>
      </div>
      <div className="flex-1 min-h-0 mt-0 overflow-hidden">
        <PropertiesPanel />
      </div>
    </div>
  );
}
