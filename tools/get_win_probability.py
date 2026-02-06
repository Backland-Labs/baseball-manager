# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Returns win probability, leverage index, and conditional win probabilities.

Backed by pre-computed win probability and leverage index tables derived from
historical MLB data (Retrosheet play-by-play, 2010-2023 averages). The tables
are loaded from data/win_probability.json and data/leverage_index.json and cover
all combinations of inning (1-12), half (TOP/BOTTOM), outs (0-2), base state
(8 combinations), and score differential (-10 to +10).
"""

import json
from pathlib import Path
from typing import Optional

from anthropic import beta_tool

# ---------------------------------------------------------------------------
# Load pre-computed tables from JSON files at module load time.
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_wp_data = json.loads((_DATA_DIR / "win_probability.json").read_text())
_WP_TABLE: dict = _wp_data["wp_table"]

_li_data = json.loads((_DATA_DIR / "leverage_index.json").read_text())
_LI_TABLE: dict = _li_data["li_table"]


def _runners_key(first: bool, second: bool, third: bool) -> str:
    return f"{'1' if first else '0'}{'1' if second else '0'}{'1' if third else '0'}"


def _lookup_wp(inning: int, half: str, outs: int, runner_key: str, score_diff: int) -> float:
    """Look up win probability for the away team from the pre-computed table.

    Args:
        inning: Current inning (1+).
        half: 'TOP' or 'BOTTOM'.
        outs: Number of outs (0-2).
        runner_key: Base state key like '000', '110', etc.
        score_diff: Score differential from the away team's perspective.

    Returns:
        Win probability for the away team (0.01-0.99).
    """
    # Clamp inning to table range (max 12 in table; beyond 12 use inning 12)
    clamped_inning = min(inning, 12)
    inning_key = f"{clamped_inning}_{half}"

    # Clamp score diff to table range (-10 to +10)
    clamped_diff = max(-10, min(10, score_diff))
    diff_key = str(clamped_diff)

    outs_key = str(outs)

    if inning_key in _WP_TABLE and outs_key in _WP_TABLE[inning_key]:
        outs_data = _WP_TABLE[inning_key][outs_key]
        if runner_key in outs_data and diff_key in outs_data[runner_key]:
            return outs_data[runner_key][diff_key]

    # Fallback: use bases-empty state if runner_key not found
    if inning_key in _WP_TABLE and outs_key in _WP_TABLE[inning_key]:
        outs_data = _WP_TABLE[inning_key][outs_key]
        if "000" in outs_data and diff_key in outs_data["000"]:
            return outs_data["000"][diff_key]

    # Last resort fallback
    return 0.50


def _lookup_li(inning: int, half: str, outs: int, runner_key: str, score_diff: int) -> float:
    """Look up leverage index from the pre-computed table.

    Args:
        inning: Current inning (1+).
        half: 'TOP' or 'BOTTOM'.
        outs: Number of outs (0-2).
        runner_key: Base state key like '000', '110', etc.
        score_diff: Score differential from the away team's perspective.

    Returns:
        Leverage index (0.1 to 10.0, where 1.0 is average).
    """
    clamped_inning = min(inning, 12)
    inning_key = f"{clamped_inning}_{half}"

    clamped_diff = max(-10, min(10, score_diff))
    diff_key = str(clamped_diff)

    outs_key = str(outs)

    if inning_key in _LI_TABLE and outs_key in _LI_TABLE[inning_key]:
        outs_data = _LI_TABLE[inning_key][outs_key]
        if runner_key in outs_data and diff_key in outs_data[runner_key]:
            return outs_data[runner_key][diff_key]

    # Fallback
    return 1.0


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
        managed_team_home: Whether the managed team is the home team. If not provided, the managed team is assumed to be the away team.
    Returns:
        JSON string with win probability data.
    """
    # --- Input validation ---
    if not isinstance(inning, int) or inning < 1:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid inning value: {inning}. Must be a positive integer (1+).",
        })

    half_upper = half.upper() if isinstance(half, str) else ""
    if half_upper not in ("TOP", "BOTTOM"):
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid half value: '{half}'. Must be 'TOP' or 'BOTTOM'.",
        })

    if not isinstance(outs, int) or outs < 0 or outs > 2:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid outs value: {outs}. Must be 0, 1, or 2.",
        })

    if not isinstance(score_differential, int):
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid score_differential: {score_differential}. Must be an integer.",
        })

    # Clamp extreme differentials for table lookup
    clamped_diff = max(-10, min(10, score_differential))

    runner_key = _runners_key(runner_on_first, runner_on_second, runner_on_third)

    # --- Determine perspective ---
    # Tables store WP from the away team's perspective.
    # Convert managed team's score_differential to away-team perspective.
    is_home = managed_team_home if managed_team_home is not None else False

    if is_home:
        away_diff = -clamped_diff
    else:
        away_diff = clamped_diff

    # --- Look up win probability ---
    away_wp = _lookup_wp(inning, half_upper, outs, runner_key, away_diff)

    if is_home:
        managed_wp = 1.0 - away_wp
    else:
        managed_wp = away_wp

    managed_wp = round(max(0.01, min(0.99, managed_wp)), 3)

    # --- Look up leverage index ---
    li = _lookup_li(inning, half_upper, outs, runner_key, away_diff)

    # --- Compute conditional WPs ---
    # WP if a run scores for the managed team
    if is_home:
        # Home team scores -> away diff decreases by 1
        away_wp_run = _lookup_wp(inning, half_upper, outs, "000", away_diff - 1)
        wp_if_run = round(max(0.01, min(0.99, 1.0 - away_wp_run)), 3)
    else:
        # Away team scores -> away diff increases by 1
        away_wp_run = _lookup_wp(inning, half_upper, outs, "000", away_diff + 1)
        wp_if_run = round(max(0.01, min(0.99, away_wp_run)), 3)

    # WP if the current half-inning ends scoreless
    if half_upper == "TOP":
        away_wp_scoreless = _lookup_wp(inning, "BOTTOM", 0, "000", away_diff)
    else:
        next_inn = inning + 1
        away_wp_scoreless = _lookup_wp(next_inn, "TOP", 0, "000", away_diff)

    if is_home:
        wp_if_scoreless = round(max(0.01, min(0.99, 1.0 - away_wp_scoreless)), 3)
    else:
        wp_if_scoreless = round(max(0.01, min(0.99, away_wp_scoreless)), 3)

    return json.dumps({
        "status": "ok",
        "game_state": {
            "inning": inning,
            "half": half_upper,
            "outs": outs,
            "runners": {
                "first": runner_on_first,
                "second": runner_on_second,
                "third": runner_on_third,
            },
            "score_differential": score_differential,
            "managed_team_home": is_home,
        },
        "win_probability": managed_wp,
        "leverage_index": li,
        "wp_if_run_scores": wp_if_run,
        "wp_if_inning_ends_scoreless": wp_if_scoreless,
    })
