"""Retrieves head-to-head batter vs pitcher history and projections."""

import json

from anthropic import beta_tool


@beta_tool
def get_matchup_data(batter_id: str, pitcher_id: str) -> str:
    """Retrieves head-to-head batter vs pitcher history and similarity-based
    projections. Returns direct matchup results, sample-size reliability,
    similarity-model projected wOBA, and pitch-type vulnerability breakdown.

    Args:
        batter_id: The unique identifier of the batter.
        pitcher_id: The unique identifier of the pitcher.
    Returns:
        JSON string with matchup data.
    """
    return json.dumps({
        "status": "ok",
        "batter_id": batter_id,
        "pitcher_id": pitcher_id,
        "career_pa": 12,
        "matchup_stats": {"AVG": 0.250, "SLG": 0.417, "K_rate": 0.167},
        "outcome_distribution": {"groundball": 0.40, "flyball": 0.35, "line_drive": 0.25},
        "sample_size_reliability": "small",
        "similarity_projected_wOBA": 0.330,
        "pitch_type_vulnerability": [
            {"pitch_type": "FF", "wOBA_against": 0.350},
            {"pitch_type": "SL", "wOBA_against": 0.280},
            {"pitch_type": "CH", "wOBA_against": 0.310},
        ],
    })
