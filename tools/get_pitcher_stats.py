"""Retrieves pitching statistics for a pitcher."""

import json
from typing import Optional

from anthropic import beta_tool


@beta_tool
def get_pitcher_stats(
    player_id: str,
    vs_hand: Optional[str] = None,
    home_away: Optional[str] = None,
    recency_window: Optional[str] = None,
) -> str:
    """Retrieves pitching statistics for a pitcher, including ERA/FIP/xFIP,
    strikeout and walk rates, ground ball rate, pitch mix with per-pitch metrics,
    and times-through-order splits.

    Args:
        player_id: The unique identifier of the pitcher.
        vs_hand: Optional split by batter handedness ('L' or 'R').
        home_away: Optional split by venue ('home' or 'away').
        recency_window: Optional recency filter ('last_7', 'last_14', 'last_30', 'season').
    Returns:
        JSON string with pitching statistics.
    """
    return json.dumps({
        "status": "ok",
        "player_id": player_id,
        "splits": {"vs_hand": vs_hand, "home_away": home_away, "recency_window": recency_window},
        "traditional": {"ERA": 3.50, "FIP": 3.40, "xFIP": 3.55, "SIERA": 3.45},
        "rates": {"K_pct": 0.25, "BB_pct": 0.07},
        "batted_ball": {"GB_pct": 0.45, "FB_pct": 0.35, "LD_pct": 0.20},
        "pitch_mix": [
            {"pitch_type": "FF", "usage": 0.55, "velocity": 95.0, "spin_rate": 2300, "whiff_rate": 0.22},
            {"pitch_type": "SL", "usage": 0.25, "velocity": 87.0, "spin_rate": 2500, "whiff_rate": 0.35},
            {"pitch_type": "CH", "usage": 0.20, "velocity": 86.0, "spin_rate": 1800, "whiff_rate": 0.30},
        ],
        "times_through_order": {"1st": 0.300, "2nd": 0.315, "3rd_plus": 0.340},
        "today": {"IP": 0.0, "H": 0, "R": 0, "ER": 0, "BB": 0, "K": 0},
    })
