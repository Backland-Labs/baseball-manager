"""Evaluates whether a sacrifice bunt is optimal."""

import json

from anthropic import beta_tool


@beta_tool
def evaluate_sacrifice_bunt(
    batter_id: str,
    runner_on_first: bool,
    runner_on_second: bool,
    runner_on_third: bool,
    outs: int,
    score_differential: int,
    inning: int,
) -> str:
    """Evaluates whether a sacrifice bunt is optimal given the batter, base-out
    state, score differential, and inning. Compares bunt vs swing-away expected
    runs and probability of scoring at least one run.

    Args:
        batter_id: The unique identifier of the current batter.
        runner_on_first: Whether there is a runner on first base.
        runner_on_second: Whether there is a runner on second base.
        runner_on_third: Whether there is a runner on third base.
        outs: Number of outs (0 or 1; bunting with 2 outs is rare).
        score_differential: Score difference from managed team perspective.
        inning: Current inning number.
    Returns:
        JSON string with bunt evaluation.
    """
    # Placeholder evaluation
    re_bunt = 0.664
    re_swing = 0.859
    prob_score_bunt = 0.41
    prob_score_swing = 0.43
    bunt_proficiency = 0.60

    advantage = re_bunt - re_swing
    if abs(score_differential) <= 1 and inning >= 7:
        recommendation = "bunt is favorable in a close, late game"
    elif advantage > 0:
        recommendation = "bunt is favorable on expected runs"
    else:
        recommendation = "swing away is preferred on expected runs"

    return json.dumps({
        "status": "ok",
        "batter_id": batter_id,
        "base_out_state": {"first": runner_on_first, "second": runner_on_second, "third": runner_on_third, "outs": outs},
        "expected_runs_bunt": round(re_bunt, 3),
        "expected_runs_swing": round(re_swing, 3),
        "prob_score_bunt": round(prob_score_bunt, 3),
        "prob_score_swing": round(prob_score_swing, 3),
        "bunt_proficiency": round(bunt_proficiency, 3),
        "net_expected_value": round(advantage, 3),
        "recommendation": recommendation,
    })
