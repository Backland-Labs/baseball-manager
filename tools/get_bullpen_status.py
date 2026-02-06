"""Returns detailed status of all bullpen pitchers."""

import json

from anthropic import beta_tool


@beta_tool
def get_bullpen_status() -> str:
    """Returns detailed status of all bullpen pitchers for the managed team
    including availability, stats, freshness, rest days, recent pitch counts,
    platoon splits, and warm-up status.

    Returns:
        JSON string with bullpen status.
    """
    return json.dumps({
        "status": "ok",
        "bullpen": [
            {
                "player_id": "bp_001", "name": "Placeholder Closer",
                "throws": "R", "role": "CLOSER",
                "available": True, "unavailable_reason": None,
                "freshness": "FRESH", "days_since_last": 2,
                "pitch_counts_last_3": [15, 0, 20],
                "platoon_splits": {"vs_LHB": {"ERA": 2.80}, "vs_RHB": {"ERA": 3.10}},
                "warmup_state": "cold",
            },
            {
                "player_id": "bp_002", "name": "Placeholder Setup",
                "throws": "L", "role": "SETUP",
                "available": True, "unavailable_reason": None,
                "freshness": "MODERATE", "days_since_last": 1,
                "pitch_counts_last_3": [22, 18, 0],
                "platoon_splits": {"vs_LHB": {"ERA": 3.50}, "vs_RHB": {"ERA": 3.00}},
                "warmup_state": "cold",
            },
        ],
    })
