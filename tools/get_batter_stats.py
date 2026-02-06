"""Retrieves batting statistics for a player."""

import json
from typing import Optional

from anthropic import beta_tool


@beta_tool
def get_batter_stats(
    player_id: str,
    vs_hand: Optional[str] = None,
    home_away: Optional[str] = None,
    recency_window: Optional[str] = None,
) -> str:
    """Retrieves batting statistics for a player, including traditional stats,
    advanced metrics, plate discipline, batted ball profile, and sprint speed.
    Supports splits by handedness, home/away, and recency windows.

    Args:
        player_id: The unique identifier of the batter.
        vs_hand: Optional split by pitcher handedness ('L' or 'R').
        home_away: Optional split by venue ('home' or 'away').
        recency_window: Optional recency filter ('last_7', 'last_14', 'last_30', 'season').
    Returns:
        JSON string with batting statistics.
    """
    return json.dumps({
        "status": "ok",
        "player_id": player_id,
        "splits": {"vs_hand": vs_hand, "home_away": home_away, "recency_window": recency_window},
        "traditional": {"AVG": 0.275, "OBP": 0.350, "SLG": 0.450, "OPS": 0.800},
        "advanced": {"wOBA": 0.340, "wRC_plus": 115, "barrel_rate": 0.08, "xwOBA": 0.335},
        "plate_discipline": {"K_pct": 0.22, "BB_pct": 0.10, "chase_rate": 0.28, "whiff_rate": 0.25},
        "batted_ball": {"GB_pct": 0.42, "pull_pct": 0.40, "exit_velocity": 89.5, "launch_angle": 12.5},
        "sprint_speed": 27.0,
        "situational": {"RISP_avg": 0.280, "high_leverage_ops": 0.810, "late_and_close_ops": 0.790},
        "today": {"AB": 0, "H": 0, "BB": 0, "K": 0, "RBI": 0},
    })
