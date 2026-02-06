"""Assesses the current pitcher's fatigue based on in-game trends."""

import json
from typing import Optional

from anthropic import beta_tool


@beta_tool
def get_pitcher_fatigue_assessment(pitcher_id: Optional[str] = None) -> str:
    """Assesses the current pitcher's fatigue based on in-game trends: velocity
    changes, spin rate decline, batted ball quality trend, pitch count, times
    through order, and an overall fatigue rating.

    Args:
        pitcher_id: The unique identifier of the pitcher. Defaults to current pitcher if omitted.
    Returns:
        JSON string with fatigue assessment.
    """
    return json.dumps({
        "status": "ok",
        "pitcher_id": pitcher_id or "current",
        "velocity_change": -1.2,
        "spin_rate_change": -50,
        "batted_ball_quality_trend": [
            {"inning": 1, "avg_exit_velo": 86.0},
            {"inning": 2, "avg_exit_velo": 87.5},
            {"inning": 3, "avg_exit_velo": 89.0},
        ],
        "pitch_count": 75,
        "pitch_count_by_inning": [22, 25, 28],
        "times_through_order": 2,
        "wOBA_per_time_through": {"1st": 0.290, "2nd": 0.320},
        "fatigue_level": "normal",
    })
