"""Returns expected runs for a given base-out state."""

import json

from anthropic import beta_tool

# Pre-computed 24-state run expectancy matrix (2023 MLB averages approximation)
RE_MATRIX: dict[str, list[float]] = {
    # Key: "runners" string (e.g., "000" = bases empty, "100" = runner on 1st)
    # Value: [0 outs, 1 out, 2 outs]
    "000": [0.481, 0.254, 0.098],
    "100": [0.859, 0.509, 0.224],
    "010": [1.100, 0.664, 0.319],
    "001": [1.343, 0.950, 0.353],
    "110": [1.437, 0.884, 0.429],
    "101": [1.798, 1.124, 0.478],
    "011": [1.920, 1.352, 0.570],
    "111": [2.282, 1.520, 0.736],
}


def _runners_key(first: bool, second: bool, third: bool) -> str:
    return f"{'1' if first else '0'}{'1' if second else '0'}{'1' if third else '0'}"


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
    if outs < 0 or outs > 2:
        return json.dumps({"status": "error", "message": f"Invalid outs value: {outs}. Must be 0, 1, or 2."})

    key = _runners_key(runner_on_first, runner_on_second, runner_on_third)
    expected_runs = RE_MATRIX[key][outs]

    # Approximate probability of scoring at least one run (rough heuristic)
    prob_score = min(0.95, expected_runs * 0.55) if expected_runs > 0 else 0.0

    # Approximate run distribution
    prob_0 = 1.0 - prob_score
    prob_1 = prob_score * 0.50
    prob_2 = prob_score * 0.25
    prob_3_plus = prob_score * 0.25

    # Common transition deltas
    transitions = {}
    if runner_on_first and not runner_on_second and outs < 2:
        steal_key = _runners_key(False, True, runner_on_third)
        cs_key = _runners_key(False, False, runner_on_third)
        re_success = RE_MATRIX[steal_key][outs]
        re_caught = RE_MATRIX[cs_key][min(outs + 1, 2)] if outs + 1 <= 2 else 0.0
        transitions["steal_2b_success"] = round(re_success - expected_runs, 3)
        transitions["steal_2b_caught"] = round(re_caught - expected_runs, 3)

    return json.dumps({
        "status": "ok",
        "base_out_state": {"first": runner_on_first, "second": runner_on_second, "third": runner_on_third, "outs": outs},
        "expected_runs": round(expected_runs, 3),
        "prob_scoring_at_least_one": round(prob_score, 3),
        "run_distribution": {
            "prob_0_runs": round(prob_0, 3),
            "prob_1_run": round(prob_1, 3),
            "prob_2_runs": round(prob_2, 3),
            "prob_3_plus_runs": round(prob_3_plus, 3),
        },
        "transitions": transitions,
    })
