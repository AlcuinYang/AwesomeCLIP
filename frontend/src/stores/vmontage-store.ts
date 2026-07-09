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
  pipelineStatus: string | null; // 上传/检测/导演的进行中状态行
  error: string | null;

  init: () => Promise<void>;
  reload: () => Promise<void>;
  loadIntoEditor: () => void;
  saveEdl: () => Promise<void>;
  sendChat: (instruction: string) => Promise<void>;
  undo: () => Promise<void>;
  startRender: (preview: boolean) => Promise<void>;
  uploadAndDetect: (files: File[]) => Promise<void>;
  runDirector: (styleHint?: string) => Promise<void>;
}

let wsStarted = false;

export const useVmontage = create<VmontageStore>((set, get) => ({
  state: null,
  connected: false,
  chatLog: [],
  chatBusy: false,
  renderProgress: null,
  lastRenderPath: null,
  pipelineStatus: null,
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
        else if (m.type === "detect_progress")
          set({ pipelineStatus: `检测中 ${m.done + 1}/${m.total}:${m.file}` });
        else if (m.type === "detect_done")
          set({
            pipelineStatus: null,
            chatLog: [...get().chatLog, {
              role: "assistant",
              text: `检测完成:${m.clips} 个片段,${m.selected} 个入选。时间线已生成,可以直接下指令,或点「导演编排」。`,
            }],
          });
        else if (m.type === "detect_error")
          set({ pipelineStatus: null, error: m.error });
        else if (m.type === "direct_started")
          set({ pipelineStatus: "导演编排中(约 2~3 分钟,思考模式)…" });
        else if (m.type === "direct_done")
          set({
            pipelineStatus: null,
            chatLog: [...get().chatLog, {
              role: "assistant",
              text: `导演编排完成:${m.shots} 个镜头,约 ${m.duration_s.toFixed(0)}s。分镜理由在 storyboard.json。`,
            }],
          });
        else if (m.type === "direct_error")
          set({ pipelineStatus: null, error: m.error });
        else if (m.type === "edl_updated" && m.source !== "gui") {
          // agent/undo/检测/导演改了 EDL:拉最新状态并重灌时间线
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

  uploadAndDetect: async (files: File[]) => {
    if (!files.length) return;
    set({ pipelineStatus: `上传 ${files.length} 个文件中…`, error: null });
    try {
      await api.upload(files);
      set({ pipelineStatus: "上传完成,开始检测…" });
      await api.detect(); // 进度与完成经 WS 推送
    } catch (e) {
      set({ pipelineStatus: null, error: String(e) });
    }
  },

  runDirector: async (styleHint?: string) => {
    set({ error: null });
    try {
      await api.direct(styleHint);
    } catch (e) {
      set({ error: String(e) });
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
