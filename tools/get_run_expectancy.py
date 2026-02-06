"""Returns expected runs for a given base-out state.

Backed by the pre-computed 24-state run expectancy matrix with empirically-
grounded scoring probabilities, run distributions, and transition deltas
for common strategic plays (steal, sacrifice bunt, caught stealing, etc.).
"""

import json

from anthropic import beta_tool

# ---------------------------------------------------------------------------
# Pre-computed 24-state run expectancy matrix (2023 MLB averages approximation)
# Source: Retrosheet / Tom Tango's run expectancy tables
# Key: "runners" string (e.g., "000" = bases empty, "100" = runner on 1st)
# Value: [0 outs, 1 out, 2 outs]
# ---------------------------------------------------------------------------

RE_MATRIX: dict[str, list[float]] = {
    "000": [0.481, 0.254, 0.098],
    "100": [0.859, 0.509, 0.224],
    "010": [1.100, 0.664, 0.319],
    "001": [1.343, 0.950, 0.353],
    "110": [1.437, 0.884, 0.429],
    "101": [1.798, 1.124, 0.478],
    "011": [1.920, 1.352, 0.570],
    "111": [2.282, 1.520, 0.736],
}

# ---------------------------------------------------------------------------
# Probability of scoring at least one run from each base-out state
# Source: derived from Retrosheet play-by-play data (2019-2023 averages)
# ---------------------------------------------------------------------------

PROB_AT_LEAST_ONE: dict[str, list[float]] = {
    #           0 outs   1 out    2 outs
    "000": [0.264, 0.153, 0.065],
    "100": [0.421, 0.270, 0.130],
    "010": [0.601, 0.413, 0.218],
    "001": [0.833, 0.650, 0.265],
    "110": [0.609, 0.429, 0.228],
    "101": [0.853, 0.670, 0.290],
    "011": [0.867, 0.688, 0.320],
    "111": [0.873, 0.695, 0.353],
}

# ---------------------------------------------------------------------------
# Run distribution from each base-out state
# [P(0 runs), P(1 run), P(2 runs), P(3+ runs)]
# Source: approximated from Retrosheet transition data
# ---------------------------------------------------------------------------

RUN_DISTRIBUTION: dict[str, list[list[float]]] = {
    # 0 outs, 1 out, 2 outs
    "000": [
        [0.736, 0.142, 0.068, 0.054],
        [0.847, 0.093, 0.035, 0.025],
        [0.935, 0.044, 0.014, 0.007],
    ],
    "100": [
        [0.579, 0.194, 0.118, 0.109],
        [0.730, 0.141, 0.070, 0.059],
        [0.870, 0.079, 0.031, 0.020],
    ],
    "010": [
        [0.399, 0.320, 0.142, 0.139],
        [0.587, 0.228, 0.098, 0.087],
        [0.782, 0.139, 0.048, 0.031],
    ],
    "001": [
        [0.167, 0.486, 0.181, 0.166],
        [0.350, 0.378, 0.146, 0.126],
        [0.735, 0.179, 0.052, 0.034],
    ],
    "110": [
        [0.391, 0.254, 0.177, 0.178],
        [0.571, 0.196, 0.116, 0.117],
        [0.772, 0.119, 0.060, 0.049],
    ],
    "101": [
        [0.147, 0.380, 0.227, 0.246],
        [0.330, 0.317, 0.176, 0.177],
        [0.710, 0.157, 0.072, 0.061],
    ],
    "011": [
        [0.133, 0.365, 0.247, 0.255],
        [0.312, 0.300, 0.194, 0.194],
        [0.680, 0.171, 0.082, 0.067],
    ],
    "111": [
        [0.127, 0.285, 0.249, 0.339],
        [0.305, 0.264, 0.195, 0.236],
        [0.647, 0.172, 0.094, 0.087],
    ],
}


def _runners_key(first: bool, second: bool, third: bool) -> str:
    return f"{'1' if first else '0'}{'1' if second else '0'}{'1' if third else '0'}"


def _get_re(key: str, outs: int) -> float:
    """Get run expectancy, returning 0.0 for 3-out (inning over) states."""
    if outs > 2:
        return 0.0
    return RE_MATRIX[key][outs]


def _compute_transitions(first: bool, second: bool, third: bool, outs: int) -> dict:
    """Compute run-expectancy change for common strategic transitions.

    Returns a dict of transition_name -> {re_change, description}.
    Only includes transitions that are applicable to the given state.
    """
    key = _runners_key(first, second, third)
    current_re = _get_re(key, outs)
    transitions = {}

    # --- Steal of 2nd base (runner on 1st, 2nd base open) ---
    if first and not second:
        # Success: runner moves 1st -> 2nd
        success_key = _runners_key(False, True, third)
        success_re = _get_re(success_key, outs)
        # Caught stealing: runner removed, +1 out
        cs_key = _runners_key(False, False, third)
        cs_re = _get_re(cs_key, outs + 1)
        transitions["steal_2b_success"] = {
            "re_change": round(success_re - current_re, 3),
            "description": "Runner steals 2nd base successfully",
        }
        transitions["steal_2b_caught"] = {
            "re_change": round(cs_re - current_re, 3),
            "description": "Runner caught stealing 2nd base",
        }
        # Breakeven rate for this steal attempt
        re_gain = success_re - current_re
        re_loss = current_re - cs_re
        if re_gain + re_loss > 0:
            breakeven = round(re_loss / (re_gain + re_loss), 3)
        else:
            breakeven = 1.0
        transitions["steal_2b_breakeven"] = {
            "rate": breakeven,
            "description": f"Runner needs >{breakeven:.1%} success rate for positive EV",
        }

    # --- Steal of 3rd base (runner on 2nd, 3rd base open) ---
    if second and not third and outs < 2:
        # Success: runner moves 2nd -> 3rd
        success_key = _runners_key(first, False, True)
        success_re = _get_re(success_key, outs)
        # Caught stealing: runner removed, +1 out
        cs_key = _runners_key(first, False, False)
        cs_re = _get_re(cs_key, outs + 1)
        transitions["steal_3b_success"] = {
            "re_change": round(success_re - current_re, 3),
            "description": "Runner steals 3rd base successfully",
        }
        transitions["steal_3b_caught"] = {
            "re_change": round(cs_re - current_re, 3),
            "description": "Runner caught stealing 3rd base",
        }
        re_gain = success_re - current_re
        re_loss = current_re - cs_re
        if re_gain + re_loss > 0:
            breakeven = round(re_loss / (re_gain + re_loss), 3)
        else:
            breakeven = 1.0
        transitions["steal_3b_breakeven"] = {
            "rate": breakeven,
            "description": f"Runner needs >{breakeven:.1%} success rate for positive EV",
        }

    # --- Sacrifice bunt transitions ---
    if outs < 2:
        # Runner on 1st only -> bunt advances to 2nd, batter out
        if first and not second and not third:
            bunt_key = _runners_key(False, True, False)
            bunt_re = _get_re(bunt_key, outs + 1)
            transitions["sac_bunt_1st_to_2nd"] = {
                "re_change": round(bunt_re - current_re, 3),
                "description": "Sacrifice bunt: runner 1st->2nd, batter out",
            }

        # Runners on 1st and 2nd -> bunt advances both, batter out
        if first and second and not third:
            bunt_key = _runners_key(False, True, True)
            bunt_re = _get_re(bunt_key, outs + 1)
            transitions["sac_bunt_1st2nd_to_2nd3rd"] = {
                "re_change": round(bunt_re - current_re, 3),
                "description": "Sacrifice bunt: runners advance to 2nd & 3rd, batter out",
            }

        # Runner on 2nd only -> bunt advances to 3rd, batter out
        if second and not first and not third:
            bunt_key = _runners_key(False, False, True)
            bunt_re = _get_re(bunt_key, outs + 1)
            transitions["sac_bunt_2nd_to_3rd"] = {
                "re_change": round(bunt_re - current_re, 3),
                "description": "Sacrifice bunt: runner 2nd->3rd, batter out",
            }

    # --- Double play transition ---
    if first and outs < 2:
        # GIDP: runner on 1st removed, batter out, +2 outs
        dp_key = _runners_key(False, second, third)
        dp_outs = outs + 2
        dp_re = _get_re(dp_key, dp_outs)
        transitions["double_play"] = {
            "re_change": round(dp_re - current_re, 3),
            "description": "Ground into double play (runner on 1st and batter out)",
        }

    # --- Wild pitch / passed ball with runner on 3rd ---
    if third and outs < 2:
        # Runner scores from 3rd, RE resets (approximate as 1 run scored + new state)
        wp_key = _runners_key(first, second, False)
        wp_re = _get_re(wp_key, outs)
        transitions["wild_pitch_run_scores"] = {
            "re_change": round((1.0 + wp_re) - current_re, 3),
            "description": "Wild pitch/passed ball: runner scores from 3rd",
        }

    return transitions


@beta_tool
def get_run_expectancy(
    runner_on_first: bool,
    runner_on_second: bool,
    runner_on_third: bool,
    outs: int,
) -> str:
    """Returns expected runs for a given base-out state, probability of scoring
    at least one run, and the run distribution. Backed by the 24-state run
    expectancy matrix.

    Args:
        runner_on_first: Whether there is a runner on first base.
        runner_on_second: Whether there is a runner on second base.
        runner_on_third: Whether there is a runner on third base.
        outs: Number of outs (0, 1, or 2).
    Returns:
        JSON string with run expectancy data.
    """
    if not isinstance(outs, int) or outs < 0 or outs > 2:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid outs value: {outs}. Must be 0, 1, or 2.",
        })

    key = _runners_key(runner_on_first, runner_on_second, runner_on_third)
    expected_runs = RE_MATRIX[key][outs]

    # Empirically-grounded probability of scoring at least one run
    prob_score = PROB_AT_LEAST_ONE[key][outs]

    # Empirically-grounded run distribution
    dist = RUN_DISTRIBUTION[key][outs]

    # Common transition deltas
    transitions = _compute_transitions(runner_on_first, runner_on_second, runner_on_third, outs)

    return json.dumps({
        "status": "ok",
        "base_out_state": {
            "first": runner_on_first,
            "second": runner_on_second,
            "third": runner_on_third,
            "outs": outs,
        },
        "expected_runs": round(expected_runs, 3),
        "prob_scoring_at_least_one": round(prob_score, 3),
        "run_distribution": {
            "prob_0_runs": round(dist[0], 3),
            "prob_1_run": round(dist[1], 3),
            "prob_2_runs": round(dist[2], 3),
            "prob_3_plus_runs": round(dist[3], 3),
        },
        "transitions": transitions,
    })
