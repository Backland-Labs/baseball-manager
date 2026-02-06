"""Compares a potential pinch hitter against the current batter."""

import json

from anthropic import beta_tool


@beta_tool
def get_platoon_comparison(
    current_batter_id: str,
    pinch_hitter_id: str,
    pitcher_id: str,
) -> str:
    """Compares a potential pinch hitter against the current batter for the
    active matchup. Returns projected wOBA for each, platoon advantage delta,
    defensive cost, and bench depth impact.

    Args:
        current_batter_id: The unique identifier of the current batter.
        pinch_hitter_id: The unique identifier of the potential pinch hitter.
        pitcher_id: The unique identifier of the current pitcher for matchup context.
    Returns:
        JSON string with platoon comparison.
    """
    return json.dumps({
        "status": "ok",
        "current_batter_id": current_batter_id,
        "pinch_hitter_id": pinch_hitter_id,
        "pitcher_id": pitcher_id,
        "current_batter_projected_wOBA": 0.310,
        "pinch_hitter_projected_wOBA": 0.355,
        "platoon_advantage_delta": 0.045,
        "defensive_cost": -0.010,
        "bench_depth_remaining": 3,
    })
