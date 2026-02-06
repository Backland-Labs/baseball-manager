"""Returns recommended defensive positioning for a given matchup and situation."""

import json

from anthropic import beta_tool


@beta_tool
def get_defensive_positioning(
    batter_id: str,
    pitcher_id: str,
    outs: int,
    runner_on_first: bool,
    runner_on_second: bool,
    runner_on_third: bool,
    score_differential: int,
    inning: int,
) -> str:
    """Returns recommended infield and outfield positioning for a given
    batter-pitcher matchup and game situation. Includes spray chart summary,
    infield-in cost/benefit, and shift recommendations.

    Args:
        batter_id: The unique identifier of the batter.
        pitcher_id: The unique identifier of the pitcher.
        outs: Number of outs (0, 1, or 2).
        runner_on_first: Whether there is a runner on first base.
        runner_on_second: Whether there is a runner on second base.
        runner_on_third: Whether there is a runner on third base.
        score_differential: Score difference from managed team perspective.
        inning: Current inning number.
    Returns:
        JSON string with defensive positioning recommendations.
    """
    return json.dumps({
        "status": "ok",
        "batter_id": batter_id,
        "pitcher_id": pitcher_id,
        "spray_chart": {
            "groundball": {"pull_pct": 0.45, "center_pct": 0.30, "oppo_pct": 0.25},
            "flyball": {"pull_pct": 0.40, "center_pct": 0.35, "oppo_pct": 0.25},
        },
        "infield_recommendation": "standard",
        "outfield_recommendation": "standard",
        "infield_in_analysis": {
            "runs_saved_at_home": 0.15,
            "extra_hits_allowed": 0.08,
        },
        "shift_recommendation": "no shift -- comply with 2-and-2 rule",
    })
