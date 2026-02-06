"""Returns win probability, leverage index, and conditional win probabilities."""

import json
from typing import Optional

from anthropic import beta_tool


@beta_tool
def get_win_probability(
    inning: int,
    half: str,
    outs: int,
    runner_on_first: bool,
    runner_on_second: bool,
    runner_on_third: bool,
    score_differential: int,
    managed_team_home: Optional[bool] = None,
) -> str:
    """Returns win probability, leverage index, and conditional win probabilities
    given the full game state.

    Args:
        inning: Current inning (1-9+).
        half: Half of the inning ('TOP' or 'BOTTOM').
        outs: Number of outs (0, 1, or 2).
        runner_on_first: Whether there is a runner on first base.
        runner_on_second: Whether there is a runner on second base.
        runner_on_third: Whether there is a runner on third base.
        score_differential: Score difference from the managed team's perspective (positive = leading).
        managed_team_home: Whether the managed team is the home team.
    Returns:
        JSON string with win probability data.
    """
    # Placeholder win probability based on score differential and inning
    base_wp = 0.50 + (score_differential * 0.08)
    # Late innings amplify the lead/deficit effect
    inning_factor = 1.0 + (max(0, inning - 5) * 0.03)
    wp = max(0.01, min(0.99, base_wp * inning_factor / (1.0 + (inning_factor - 1.0) * 0.5)))

    # Home team advantage adjustment
    if managed_team_home is True and half == "BOTTOM":
        wp = min(0.99, wp + 0.02)
    elif managed_team_home is False and half == "TOP":
        wp = min(0.99, wp + 0.01)

    # Placeholder leverage index (higher in close, late games)
    is_close = abs(score_differential) <= 2
    is_late = inning >= 7
    li = 1.0
    if is_close and is_late:
        li = 2.5
    elif is_close:
        li = 1.5
    elif is_late:
        li = 1.2

    # Adjust for runners and outs
    runners_on = sum([runner_on_first, runner_on_second, runner_on_third])
    li += runners_on * 0.3 - outs * 0.2

    return json.dumps({
        "status": "ok",
        "game_state": {
            "inning": inning, "half": half, "outs": outs,
            "runners": {"first": runner_on_first, "second": runner_on_second, "third": runner_on_third},
            "score_differential": score_differential,
        },
        "win_probability": round(wp, 3),
        "leverage_index": round(max(0.1, li), 2),
        "wp_if_run_scores": round(min(0.99, wp + 0.05), 3),
        "wp_if_inning_ends_scoreless": round(max(0.01, wp - 0.03), 3),
    })
