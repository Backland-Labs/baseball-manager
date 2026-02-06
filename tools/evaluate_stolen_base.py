"""Evaluates a potential stolen base attempt."""

import json

from anthropic import beta_tool


@beta_tool
def evaluate_stolen_base(
    runner_id: str,
    target_base: int,
    pitcher_id: str,
    catcher_id: str,
) -> str:
    """Evaluates a potential stolen base attempt given runner speed, pitcher hold
    time, catcher arm, and current base-out state. Returns success probability,
    breakeven rate, expected run-expectancy change, and a recommendation.

    Args:
        runner_id: The unique identifier of the baserunner attempting the steal.
        target_base: The base being stolen (2 for 2nd, 3 for 3rd).
        pitcher_id: The unique identifier of the current pitcher.
        catcher_id: The unique identifier of the opposing catcher.
    Returns:
        JSON string with stolen base evaluation.
    """
    # Placeholder evaluation
    success_prob = 0.72
    breakeven = 0.715 if target_base == 2 else 0.80
    re_if_success = 0.241
    re_if_caught = -0.605
    net_re = success_prob * re_if_success + (1 - success_prob) * re_if_caught

    if success_prob >= breakeven + 0.05:
        recommendation = "favorable"
    elif success_prob >= breakeven:
        recommendation = "marginal"
    else:
        recommendation = "unfavorable"

    return json.dumps({
        "status": "ok",
        "runner_id": runner_id,
        "target_base": target_base,
        "pitcher_id": pitcher_id,
        "catcher_id": catcher_id,
        "success_probability": round(success_prob, 3),
        "breakeven_rate": round(breakeven, 3),
        "re_change_if_successful": round(re_if_success, 3),
        "re_change_if_caught": round(re_if_caught, 3),
        "net_expected_re_change": round(net_re, 3),
        "recommendation": recommendation,
    })
