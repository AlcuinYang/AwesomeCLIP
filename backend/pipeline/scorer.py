"""可解释打分:total = Σ w_i · feature_i,权重在 settings.yaml(规格 §5.4)。"""
from __future__ import annotations

import re

from ..schemas.models import Score, ScorecardsFile


def score_clips(cards: ScorecardsFile, settings: dict) -> ScorecardsFile:
    w = settings["scorer"]["weights"]
    min_score = float(settings["scorer"]["min_score"])
    for clip in cards.clips:
        breakdown: dict[str, float] = {}
        for tag in clip.tags:
            if m := re.fullmatch(r"clutch_1v(\d+)", tag):
                breakdown["clutch"] = int(m.group(1)) * float(w["clutch_per_enemy"])
            elif m := re.fullmatch(r"multikill_(\d+)", tag):
                breakdown["multikill"] = int(m.group(1)) * float(w["multikill_per_kill"])
            elif tag == "flick":
                breakdown["flick"] = float(w["flick"])
        if clip.evidence.round_won is not None:
            breakdown["round_context"] = float(w["round_context"])
        total = round(sum(breakdown.values()), 2)
        clip.score = Score(total=total, breakdown=breakdown)
        clip.selected = total >= min_score
    cards.clips.sort(key=lambda c: c.score.total, reverse=True)
    return cards
