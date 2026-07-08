"""全部 JSON 中间产物的 pydantic 模型(数据契约,见规格 §4)。

events.json / scorecards.json / beats.json / edl.json 的唯一 schema 定义。
任何阶段读写 JSON 都必须经过这里的模型,保证契约稳定。
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- events.json

class VideoMeta(BaseModel):
    width: int
    height: int
    fps: float
    duration_s: float


class KillEvent(BaseModel):
    type: Literal["kill"] = "kill"
    frame: int
    t: float
    headshot: bool = False
    confidence: float = 1.0
    # 击杀前 100–300ms 视角角速度(度/秒),神经枪判定用;检测失败为 None
    pre_kill_angular_velocity_deg_s: Optional[float] = None


class DeathEvent(BaseModel):
    type: Literal["death"] = "death"
    frame: int
    t: float
    confidence: float = 1.0


class AliveStateEvent(BaseModel):
    type: Literal["alive_state"] = "alive_state"
    frame: int
    t: float
    ally_alive: int
    enemy_alive: int


class RoundEndEvent(BaseModel):
    type: Literal["round_end"] = "round_end"
    frame: int
    t: float
    won: bool
    confidence: float = 1.0


class SpikePlantEvent(BaseModel):
    type: Literal["spike_plant"] = "spike_plant"
    frame: int
    t: float


class RoundBoundaryEvent(BaseModel):
    """回合切换(顶部比分数字发生变化):击杀簇在此强制断开。"""
    type: Literal["round_boundary"] = "round_boundary"
    frame: int
    t: float


Event = Union[KillEvent, DeathEvent, AliveStateEvent, RoundEndEvent,
              SpikePlantEvent, RoundBoundaryEvent]


class SourceEvents(BaseModel):
    """单个素材文件的 L1 输出(规格 §4.1)。"""
    source: str
    video_meta: VideoMeta
    events: list[Event] = Field(default_factory=list, discriminator=None)


class EventsFile(BaseModel):
    """events.json 顶层:多素材 → 列表。"""
    sources: list[SourceEvents] = Field(default_factory=list)


# ------------------------------------------------------------ scorecards.json

class ClipSpan(BaseModel):
    start_t: float
    end_t: float

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


class KillEvidence(BaseModel):
    t: float
    headshot: bool = False
    pre_kill_angular_velocity_deg_s: Optional[float] = None


class AliveStateEvidence(BaseModel):
    ally: int
    enemy: int
    t: float


class RoundWonEvidence(BaseModel):
    t: float


class Evidence(BaseModel):
    alive_state: Optional[AliveStateEvidence] = None
    kills: list[KillEvidence] = Field(default_factory=list)
    round_won: Optional[RoundWonEvidence] = None


class Score(BaseModel):
    total: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)


class ScoreCard(BaseModel):
    """片段证据卡(规格 §4.2)。"""
    clip_id: str
    source: str
    span: ClipSpan
    anchor_ts: list[float] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)
    score: Score = Field(default_factory=Score)
    narration: Optional[str] = None  # L3 叙事层填充,失败/未启用为 None
    selected: bool = False


class ScorecardsFile(BaseModel):
    clips: list[ScoreCard] = Field(default_factory=list)


# ----------------------------------------------------------------- beats.json

class EnergyPoint(BaseModel):
    t: float
    rms: float


class BeatsFile(BaseModel):
    music: str
    bpm: float
    beats_t: list[float] = Field(default_factory=list)
    downbeats_t: list[float] = Field(default_factory=list)
    energy_curve: list[EnergyPoint] = Field(default_factory=list)


# ------------------------------------------------------------------- edl.json

class SnapInfo(BaseModel):
    cut_beat_t: Optional[float] = None
    mode: Literal["cut_on_beat", "anchor_align", "none"] = "cut_on_beat"


class SpeedRamp(BaseModel):
    """击杀瞬间变速(锚点式,规格 §5.7)。"""
    anchor_t: float           # 源内时间(击杀帧)
    factor: float = 0.5       # 播放倍速:0.5 = 半速慢放
    pre_s: float = 0.3        # 击杀前进入慢放的时长
    post_s: float = 0.5       # 击杀后退出慢放的时长


class SpeedSpan(BaseModel):
    """任意跨度变速(导演的间隙压缩用;factor>1 快进,<1 慢放)。源内时间。"""
    from_t: float
    to_t: float
    factor: float


class ClipEffects(BaseModel):
    frame_drop: bool = False
    frame_drop_strength: float = 0.5  # 丢帧比例 0..1
    speed_ramp: Optional[SpeedRamp] = None
    speed_spans: list[SpeedSpan] = Field(default_factory=list)


class ClipAudio(BaseModel):
    game_volume: float = 0.35
    duck_music: bool = False


class TimelineEntry(BaseModel):
    clip_id: str
    source: str
    in_t: float
    out_t: float
    timeline_start_t: float
    snap: SnapInfo = Field(default_factory=SnapInfo)
    effects: ClipEffects = Field(default_factory=ClipEffects)
    audio: ClipAudio = Field(default_factory=ClipAudio)

    @property
    def duration(self) -> float:
        return self.out_t - self.in_t


class GlobalEffects(BaseModel):
    frame_drop: bool = False


class RenderSettings(BaseModel):
    resolution: str = "source"     # "source" | "1080p" | "720p"
    codec: str = "h264_nvenc"      # 不可用时渲染阶段自动回退 libx264
    crf_equivalent: int = 20


class GapTreatment(BaseModel):
    """导演对片段内一个击杀间隙的裁决(可解释:必须带 rationale)。"""
    from_t: float             # 源内时间
    to_t: float
    action: Literal["keep", "compress", "cut"]
    factor: Optional[float] = None   # compress 时的倍速(4~8 常用)
    rationale: str = ""


class Shot(BaseModel):
    """分镜表中的一个镜头。"""
    clip_id: str
    in_t: float
    out_t: float
    gap_treatments: list[GapTreatment] = Field(default_factory=list)
    rationale: str = ""


class RejectedClip(BaseModel):
    clip_id: str
    reason: str


class Storyboard(BaseModel):
    """导演分镜表(一等产物):MLLM 以导演思维链产出,EDL 由此生成。

    可解释性:每个镜头/间隙/弃用都带 rationale;reasoning 保存模型完整思考过程,
    供测试阶段优化提示词,也是未来偏好训练的数据。
    """
    style_hint: Optional[str] = None
    shots: list[Shot] = Field(default_factory=list)
    rejected: list[RejectedClip] = Field(default_factory=list)
    model: str = ""
    rubric_version: str = "v2"
    created_at: str = ""
    reasoning: Optional[str] = None


class EdlFile(BaseModel):
    """剪辑决策表 — 渲染的唯一依据(规格 §4.4)。"""
    version: int = 1
    music: Optional[str] = None
    target_duration_s: float = 60
    global_effects: GlobalEffects = Field(default_factory=GlobalEffects)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    render: RenderSettings = Field(default_factory=RenderSettings)
