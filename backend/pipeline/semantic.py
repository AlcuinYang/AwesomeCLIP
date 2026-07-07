"""L2 语义层:原子事件 → 带标签/证据的片段(规格 §5.3)。

- 片段切分:击杀簇(间隔 < merge_gap 合并)外扩 pre_kill/post_kill。
- multikill_N:滑动窗口(默认 8s)内击杀数的最大值 N(N>=2)。
- clutch_1vN:某时刻 ally==1 且 enemy==N(N>=2),此后至 round_end(won) 击杀 >= N-容差。
- flick:headshot 且 pre_kill_angular_velocity >= 阈值。
"""
from __future__ import annotations

from ..schemas.models import (
    AliveStateEvent, AliveStateEvidence, ClipSpan, Evidence, EventsFile, KillEvent,
    KillEvidence, RoundEndEvent, RoundWonEvidence, ScoreCard, ScorecardsFile,
)


def _cluster_kills(kills: list[KillEvent], merge_gap: float) -> list[list[KillEvent]]:
    clusters: list[list[KillEvent]] = []
    for k in sorted(kills, key=lambda e: e.t):
        if clusters and k.t - clusters[-1][-1].t < merge_gap:
            clusters[-1].append(k)
        else:
            clusters.append([k])
    return clusters


def _max_kills_in_window(ts: list[float], window: float) -> int:
    best, j = 0, 0
    for i in range(len(ts)):
        while ts[i] - ts[j] > window:
            j += 1
        best = max(best, i - j + 1)
    return best


def _find_clutches(alive_states: list[AliveStateEvent], kills: list[KillEvent],
                   round_ends: list[RoundEndEvent], tolerance: int
                   ) -> list[tuple[float, float, int, AliveStateEvent, RoundEndEvent]]:
    """返回 [(start_t, end_t, N, 证据状态, 胜利事件)] 的残局区间列表。"""
    results = []
    for st in alive_states:
        if st.ally_alive != 1 or st.enemy_alive < 2:
            continue
        re = next((r for r in round_ends if r.t > st.t), None)
        if re is None or not re.won:
            continue
        n_kills = sum(1 for k in kills if st.t < k.t <= re.t)
        if n_kills >= st.enemy_alive - tolerance:
            results.append((st.t, re.t, st.enemy_alive, st, re))
    # 同一回合可能有多个 1vN 状态(N 递减),保留 N 最大的那个
    dedup: dict[float, tuple] = {}
    for item in results:
        key = item[1]  # round_end 时间作为回合标识
        if key not in dedup or item[2] > dedup[key][2]:
            dedup[key] = item
    return list(dedup.values())


def build_scorecards(events_file: EventsFile, settings: dict) -> ScorecardsFile:
    sem = settings["semantic"]
    clip_cfg = sem["clip"]
    flick_thr = float(sem["flick_angular_velocity_deg_s"])
    window = float(sem["multikill_window_s"])
    clips: list[ScoreCard] = []
    seq = 0

    for src in events_file.sources:
        kills = [e for e in src.events if isinstance(e, KillEvent)]
        alive_states = [e for e in src.events if isinstance(e, AliveStateEvent)]
        round_ends = [e for e in src.events if isinstance(e, RoundEndEvent)]
        clutches = _find_clutches(alive_states, kills, round_ends,
                                  int(sem["clutch_kill_tolerance"]))

        for cluster in _cluster_kills(kills, float(clip_cfg["merge_gap_s"])):
            seq += 1
            ts = [k.t for k in cluster]
            start_t = max(0.0, ts[0] - float(clip_cfg["pre_kill_s"]))
            end_t = min(src.video_meta.duration_s, ts[-1] + float(clip_cfg["post_kill_s"]))
            tags: list[str] = []
            evidence = Evidence(kills=[
                KillEvidence(t=k.t, headshot=k.headshot,
                             pre_kill_angular_velocity_deg_s=k.pre_kill_angular_velocity_deg_s)
                for k in cluster])

            n_multi = _max_kills_in_window(ts, window)
            if n_multi >= 2:
                tags.append(f"multikill_{min(n_multi, 5)}")

            for (c_start, c_end, n, st, re) in clutches:
                if any(c_start < k.t <= c_end for k in cluster):
                    tags.append(f"clutch_1v{n}")
                    evidence.alive_state = AliveStateEvidence(
                        ally=st.ally_alive, enemy=st.enemy_alive, t=st.t)
                    evidence.round_won = RoundWonEvidence(t=re.t)
                    break

            if any(k.headshot and (k.pre_kill_angular_velocity_deg_s or 0) >= flick_thr
                   for k in cluster):
                tags.append("flick")

            clips.append(ScoreCard(
                clip_id=f"clip_{seq:03d}", source=src.source,
                span=ClipSpan(start_t=round(start_t, 3), end_t=round(end_t, 3)),
                anchor_ts=ts, tags=tags, evidence=evidence))
    return ScorecardsFile(clips=clips)
