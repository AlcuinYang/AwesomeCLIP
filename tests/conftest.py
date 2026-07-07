from __future__ import annotations

import pytest

from backend.config import load_settings
from backend.schemas.models import (
    AliveStateEvent, EventsFile, KillEvent, RoundEndEvent, SourceEvents, VideoMeta,
)


@pytest.fixture(scope="session")
def settings() -> dict:
    return load_settings()


def make_kill(t: float, headshot: bool = False, angvel: float | None = None) -> KillEvent:
    return KillEvent(frame=int(t * 60), t=t, headshot=headshot,
                     pre_kill_angular_velocity_deg_s=angvel)


@pytest.fixture
def clutch_events() -> EventsFile:
    """一段素材:1v3 残局三杀(含一次 flick)后获胜,外加一个孤立单杀。"""
    events = [
        AliveStateEvent(frame=1790, t=29.83, ally_alive=1, enemy_alive=3),
        make_kill(30.57, headshot=True, angvel=236.0),
        make_kill(32.02, headshot=False, angvel=41.2),
        make_kill(33.50, headshot=True, angvel=88.5),
        RoundEndEvent(frame=2095, t=34.92, won=True),
        make_kill(80.00),  # 孤立单杀,不构成 multikill
    ]
    src = SourceEvents(
        source="sources/clip_003.mp4",
        video_meta=VideoMeta(width=2560, height=1440, fps=60, duration_s=312.5),
        events=events,
    )
    return EventsFile(sources=[src])
