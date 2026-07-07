"""L2 语义层:原子事件 → 带标签/证据的片段。

- 片段切分:击杀簇(间隔 < merge_gap 合并,**自己被杀则强制断簇**)外扩
  pre_kill/post_kill。
- multikill_N:**簇内总击杀数**(用户定义:不连续击杀也算,如 1,2,1 分布的
  四杀只要中间没死就是一段四杀高光;取代规格 §5.3 的滑动窗口口径)。
- clutch_1vN:某时刻 ally==1 且 enemy==N(N>=2),此后至 round_end(won) 击杀 >= N-容差。
- flick:headshot 且 pre_kill_angular_velocity >= 阈值。
"""
from __future__ import annotations

from ..schemas.models import (
    AliveStateEvent, AliveStateEvidence, ClipSpan, DeathEvent, Evidence, EventsFile,
    KillEvent, KillEvidence, RoundEndEvent, RoundWonEvidence, ScoreCard, ScorecardsFile,
)


def _cluster_kills(kills: list[KillEvent], merge_gap: float,
                   deaths: list[DeathEvent] | None = None) -> list[list[KillEvent]]:
    """击杀聚簇:间隔 < merge_gap 合并;两杀之间若有自己的 death 则断簇。"""
    death_ts = sorted(d.t for d in deaths) if deaths else []

    def died_between(t0: float, t1: float) -> bool:
        return any(t0 < dt <= t1 for dt in death_ts)

    clusters: list[list[KillEvent]] = []
    for k in sorted(kills, key=lambda e: e.t):
        if clusters and k.t - clusters[-1][-1].t < merge_gap \
                and not died_between(clusters[-1][-1].t, k.t):
            clusters[-1].append(k)
        else:
            clusters.append([k])
    return clusters


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
    clips: list[ScoreCard] = []
    seq = 0

    for src in events_file.sources:
        kills = [e for e in src.events if isinstance(e, KillEvent)]
        deaths = [e for e in src.events if isinstance(e, DeathEvent)]
        alive_states = [e for e in src.events if isinstance(e, AliveStateEvent)]
        round_ends = [e for e in src.events if isinstance(e, RoundEndEvent)]
        clutches = _find_clutches(alive_states, kills, round_ends,
                                  int(sem["clutch_kill_tolerance"]))

        for cluster in _cluster_kills(kills, float(clip_cfg["merge_gap_s"]), deaths):
            seq += 1
            ts = [k.t for k in cluster]
            start_t = max(0.0, ts[0] - float(clip_cfg["pre_kill_s"]))
            end_t = min(src.video_meta.duration_s, ts[-1] + float(clip_cfg["post_kill_s"]))
            tags: list[str] = []
            evidence = Evidence(kills=[
                KillEvidence(t=k.t, headshot=k.headshot,
                             pre_kill_angular_velocity_deg_s=k.pre_kill_angular_velocity_deg_s)
                for k in cluster])

            n_multi = len(cluster)  # 簇内总击杀数(不连续也算,用户定义)
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
