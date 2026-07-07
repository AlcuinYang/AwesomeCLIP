"use client";
import { create } from "zustand";
import {
  api,
  connectWs,
  designToEdl,
  edlToDesign,
  VmontageState,
  WsMessage,
} from "@/lib/vmontage";
import { core } from "@/lib/project";

interface ChatMessage {
  role: "user" | "assistant" | "op";
  text: string;
}

interface VmontageStore {
  state: VmontageState | null;
  connected: boolean;
  chatLog: ChatMessage[];
  chatBusy: boolean;
  renderProgress: number | null; // null=空闲,0..1=进行中
  lastRenderPath: string | null;
  error: string | null;

  init: () => Promise<void>;
  reload: () => Promise<void>;
  loadIntoEditor: () => void;
  saveEdl: () => Promise<void>;
  sendChat: (instruction: string) => Promise<void>;
  undo: () => Promise<void>;
  startRender: (preview: boolean) => Promise<void>;
}

let wsStarted = false;

export const useVmontage = create<VmontageStore>((set, get) => ({
  state: null,
  connected: false,
  chatLog: [],
  chatBusy: false,
  renderProgress: null,
  lastRenderPath: null,
  error: null,

  init: async () => {
    await get().reload();
    if (wsStarted || typeof window === "undefined") return;
    wsStarted = true;
    const open = () => {
      const ws = connectWs((m: WsMessage) => {
        if (m.type === "hello") set({ connected: true });
        else if (m.type === "agent_op")
          set((s) => ({ chatLog: [...s.chatLog, { role: "op", text: m.text }] }));
        else if (m.type === "render_progress") set({ renderProgress: m.fraction });
        else if (m.type === "render_done")
          set({ renderProgress: null, lastRenderPath: m.path });
        else if (m.type === "render_error")
          set({ renderProgress: null, error: m.error });
        else if (m.type === "edl_updated" && m.source !== "gui") {
          // agent/undo 改了 EDL:拉最新状态并重灌时间线
          get()
            .reload()
            .then(() => get().loadIntoEditor());
        }
      });
      ws.onclose = () => {
        set({ connected: false });
        setTimeout(open, 3000); // 简单重连
      };
    };
    open();
  },

  reload: async () => {
    try {
      const state = await api.state();
      set({ state, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  loadIntoEditor: () => {
    const state = get().state;
    if (!state) return;
    const design = edlToDesign(state);
    if (design) core.project.import(design as never);
  },

  saveEdl: async () => {
    const state = get().state;
    if (!state?.edl) return;
    const design = core.project.export();
    const edl = designToEdl(design, state.edl);
    await api.putEdl(edl);
    await get().reload();
  },

  sendChat: async (instruction: string) => {
    set((s) => ({
      chatBusy: true,
      chatLog: [...s.chatLog, { role: "user", text: instruction }],
    }));
    try {
      // 先把 GUI 上未保存的微调落盘,agent 才能在最新状态上操作
      await get().saveEdl();
      const resp = await api.chat(instruction);
      set((s) => ({
        chatLog: [...s.chatLog, { role: "assistant", text: resp.summary }],
      }));
      await get().reload();
      get().loadIntoEditor();
    } catch (e) {
      set((s) => ({
        chatLog: [...s.chatLog, { role: "assistant", text: `出错了: ${e}` }],
      }));
    } finally {
      set({ chatBusy: false });
    }
  },

  undo: async () => {
    try {
      const resp = await api.undo();
      set((s) => ({
        chatLog: [...s.chatLog, { role: "assistant", text: resp.result }],
      }));
      await get().reload();
      get().loadIntoEditor();
    } catch (e) {
      set((s) => ({
        chatLog: [...s.chatLog, { role: "assistant", text: `无法撤销: ${e}` }],
      }));
    }
  },

  startRender: async (preview: boolean) => {
    await get().saveEdl(); // 渲染前自动保存 GUI 微调
    set({ renderProgress: 0, lastRenderPath: null, error: null });
    try {
      await api.render(preview);
    } catch (e) {
      set({ renderProgress: null, error: String(e) });
    }
  },
}));
