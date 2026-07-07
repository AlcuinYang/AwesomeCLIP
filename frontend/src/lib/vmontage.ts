/**
 * vmontage 后端接入:REST/WS 客户端 + edl.json ↔ OpenVideo 工程的双向转换。
 *
 * 原则(规格 §3):edl.json 是渲染唯一依据;浏览器只播 720p proxy;
 * 导出一律调后端 /api/render。
 */

export const VMONTAGE_API =
  process.env.NEXT_PUBLIC_VMONTAGE_API ?? "http://127.0.0.1:8765";

// ------------------------------------------------------------------ 类型(镜像 pydantic 契约)

export interface EdlEntry {
  clip_id: string;
  source: string;
  in_t: number;
  out_t: number;
  timeline_start_t: number;
  snap: { cut_beat_t: number | null; mode: string };
  effects: {
    frame_drop: boolean;
    frame_drop_strength: number;
    speed_ramp: unknown | null;
  };
  audio: { game_volume: number; duck_music: boolean };
}

export interface Edl {
  version: number;
  music: string | null;
  target_duration_s: number;
  global_effects: { frame_drop: boolean };
  timeline: EdlEntry[];
  render: { resolution: string; codec: string; crf_equivalent: number };
}

export interface ScoreCard {
  clip_id: string;
  source: string;
  span: { start_t: number; end_t: number };
  anchor_ts: number[];
  tags: string[];
  evidence: {
    alive_state: { ally: number; enemy: number; t: number } | null;
    kills: {
      t: number;
      headshot: boolean;
      pre_kill_angular_velocity_deg_s: number | null;
    }[];
    round_won: { t: number } | null;
  };
  score: { total: number; breakdown: Record<string, number> };
  narration: string | null;
  selected: boolean;
}

export interface VmontageState {
  project: string;
  edl: Edl | null;
  scorecards: { clips: ScoreCard[] } | null;
  beats: { music: string; bpm: number; beats_t: number[] } | null;
  events: {
    sources: {
      source: string;
      video_meta: { width: number; height: number; fps: number; duration_s: number };
    }[];
  } | null;
  render: { status: string; path?: string; error?: string };
}

// ------------------------------------------------------------------ REST

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${VMONTAGE_API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.text();
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {}
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

export const api = {
  state: () => req<VmontageState>("/api/state"),
  putEdl: (edl: Edl) =>
    req<{ ok: boolean }>("/api/edl", { method: "PUT", body: JSON.stringify(edl) }),
  chat: (instruction: string) =>
    req<{ summary: string; ops: string[] }>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ instruction }),
    }),
  undo: () => req<{ result: string }>("/api/undo", { method: "POST" }),
  render: (preview: boolean) =>
    req<{ started: boolean; output: string }>("/api/render", {
      method: "POST",
      body: JSON.stringify({ preview }),
    }),
  renderStatus: () =>
    req<{ busy: boolean; last: { status: string; path?: string } }>(
      "/api/render/status",
    ),
};

export type WsMessage =
  | { type: "hello"; project: string }
  | { type: "edl_updated"; source: string }
  | { type: "agent_op"; text: string }
  | { type: "render_progress"; fraction: number }
  | { type: "render_done"; path: string; warnings: string[] }
  | { type: "render_error"; error: string };

export function connectWs(onMessage: (m: WsMessage) => void): WebSocket {
  const ws = new WebSocket(`${VMONTAGE_API.replace(/^http/, "ws")}/ws`);
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data));
    } catch {}
  };
  return ws;
}

// ------------------------------------------------------------------ EDL → OpenVideo 工程

const US = 1_000_000;

/** 浏览器只播 proxy(720p);源文件仅供后端渲染。 */
export function proxyUrl(source: string): string {
  const stem = source.replace(/^sources\//, "").replace(/\.[^.]+$/, "");
  return `${VMONTAGE_API}/media/proxies/${stem}.mp4`;
}

export function mediaUrl(rel: string): string {
  return `${VMONTAGE_API}/media/${rel}`;
}

const CANVAS_W = 1280;
const CANVAS_H = 720;

function videoClip(entry: EdlEntry, srcDurationS: number, fps: number) {
  return {
    type: "Video",
    id: entry.clip_id,
    name: entry.clip_id,
    src: proxyUrl(entry.source),
    timing: {
      display: {
        from: entry.timeline_start_t * US,
        to: (entry.timeline_start_t + entry.out_t - entry.in_t) * US,
      },
      trim: { from: entry.in_t * US, to: entry.out_t * US },
      duration: srcDurationS * US,
      playbackRate: 1,
    },
    transform: {
      x: 0, y: 0, width: CANVAS_W, height: CANVAS_H,
      angle: 0, opacity: 1, zIndex: 10, flip: { x: false, y: false },
    },
    style: {},
    chromaKey: { enabled: false, color: "#00FF00", similarity: 0.1, spill: 0 },
    colorAdjustment: { enabled: false, type: "basic", basic: {}, hsl: {}, curves: {} },
    locked: false,
    metadata: { vmontage: { source: entry.source } },
    effects: [],
    audio: true,
    volume: entry.audio.game_volume,
  };
}

export function edlToDesign(state: VmontageState) {
  const edl = state.edl;
  if (!edl || edl.timeline.length === 0) return null;
  const metas = new Map(
    (state.events?.sources ?? []).map((s) => [s.source, s.video_meta]),
  );
  const fps = metas.get(edl.timeline[0].source)?.fps ?? 60;
  const totalS = edl.timeline.reduce((acc, e) => acc + (e.out_t - e.in_t), 0);

  const clips: Record<string, unknown> = {};
  const videoIds: string[] = [];
  for (const entry of edl.timeline) {
    clips[entry.clip_id] = videoClip(
      entry,
      metas.get(entry.source)?.duration_s ?? entry.out_t,
      fps,
    );
    videoIds.push(entry.clip_id);
  }

  const trackList: unknown[] = [
    { id: "track_vmontage_video", name: "Highlights", type: "video", clipIds: videoIds },
  ];
  if (edl.music) {
    clips["vmontage_music"] = {
      type: "Audio",
      id: "vmontage_music",
      name: edl.music.replace(/^music\//, ""),
      src: mediaUrl(edl.music),
      timing: {
        display: { from: 0, to: totalS * US },
        trim: { from: 0, to: totalS * US },
        duration: totalS * US,
        playbackRate: 1,
      },
      locked: false,
      metadata: { vmontage: { music: true } },
      volume: 1,
    };
    trackList.push({
      id: "track_vmontage_music",
      name: "BGM",
      type: "audio",
      clipIds: ["vmontage_music"],
    });
  }

  return {
    settings: {
      width: CANVAS_W,
      height: CANVAS_H,
      fps,
      backgroundColor: "#111111",
      format: "mp4",
      videoCodec: "avc1.640033",
      bitrate: 8_000_000,
      audio: true,
      audioCodec: "opus",
      audioSampleRate: 48000,
      prioritizeSpeed: true,
      duration: totalS * US,
    },
    tracks: trackList,
    clips,
  };
}

// ------------------------------------------------------------------ OpenVideo 工程 → EDL

/**
 * 把 GUI 时间线微调写回 EDL:按 clip_id 对齐,更新 in/out/timeline_start;
 * GUI 里删掉的片段从 EDL 移除;手工挪动过的切点 snap 置 none(不再声称在拍点上)。
 */
export function designToEdl(design: any, current: Edl): Edl {
  const clipsById: Record<string, any> = design?.clips ?? {};
  const videoTrack = (design?.tracks ?? []).find(
    (t: any) => t.type === "video" && (t.clipIds ?? []).length,
  );
  if (!videoTrack) return current;

  const entriesById = new Map(current.timeline.map((e) => [e.clip_id, e]));
  const nextTimeline: EdlEntry[] = [];
  const ordered = [...videoTrack.clipIds]
    .map((id: string) => clipsById[id])
    .filter((c: any) => c && entriesById.has(c.id))
    .sort((a: any, b: any) => a.timing.display.from - b.timing.display.from);

  for (const clip of ordered) {
    const prev = entriesById.get(clip.id)!;
    const in_t = clip.timing.trim.from / US;
    const out_t = clip.timing.trim.to / US;
    const start = clip.timing.display.from / US;
    const moved =
      Math.abs(in_t - prev.in_t) > 1e-3 ||
      Math.abs(out_t - prev.out_t) > 1e-3 ||
      Math.abs(start - prev.timeline_start_t) > 1e-3;
    nextTimeline.push({
      ...prev,
      in_t: round3(in_t),
      out_t: round3(out_t),
      timeline_start_t: round3(start),
      snap: moved ? { cut_beat_t: null, mode: "none" } : prev.snap,
      audio: { ...prev.audio, game_volume: clip.volume ?? prev.audio.game_volume },
    });
  }
  return { ...current, timeline: nextTimeline };
}

function round3(x: number): number {
  return Math.round(x * 1000) / 1000;
}
