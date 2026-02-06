"""Evaluates the net value of a defensive substitution."""

import json

from anthropic import beta_tool


@beta_tool
def get_defensive_replacement_value(
    current_fielder_id: str,
    replacement_id: str,
    position: str,
) -> str:
    """Evaluates the net value of a defensive substitution by comparing the
    defensive upgrade against the offensive downgrade, scaled by estimated
    innings remaining.

    Args:
        current_fielder_id: The unique identifier of the current fielder.
        replacement_id: The unique identifier of the potential replacement.
        position: The fielding position for the substitution (e.g., 'LF', 'SS').
    Returns:
        JSON string with defensive replacement evaluation.
    """
    return json.dumps({
        "status": "ok",
        "current_fielder_id": current_fielder_id,
        "replacement_id": replacement_id,
        "position": position,
        "defensive_upgrade_oaa": 3.5,
        "offensive_downgrade_woba": -0.040,
        "estimated_innings_remaining": 3.0,
        "net_expected_value": 0.015,
        "recommendation": "marginal",
    })
