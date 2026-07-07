"""音乐分析:librosa beat_track 拍点 + RMS 能量曲线(规格 §5.5)。

downbeat 检测 V1 不做(D3 已放宽):downbeats_t 用 beats_t[::4] 的朴素近似填充,
仅供参考,不参与对齐。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schemas.models import BeatsFile, EnergyPoint


def analyze_music(music_path: Path, rel_music: str, settings: dict) -> BeatsFile:
    import librosa  # 导入耗时,延迟加载

    y, sr = librosa.load(str(music_path), sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats_t = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    hop_s = float(settings["beat"]["energy_hop_s"])
    hop_length = max(1, int(hop_s * sr))
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2,
                              hop_length=hop_length)[0]
    energy = [EnergyPoint(t=round(i * hop_s, 3), rms=round(float(v), 4))
              for i, v in enumerate(rms)]

    return BeatsFile(
        music=rel_music,
        bpm=round(bpm, 2),
        beats_t=[round(float(t), 4) for t in beats_t],
        downbeats_t=[round(float(t), 4) for t in beats_t[::4]],
        energy_curve=energy,
    )
