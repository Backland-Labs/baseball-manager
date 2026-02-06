# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_run_expectancy tool.

Verifies all feature requirements from features.json:
1. Accepts base state (runners on 1st, 2nd, 3rd in any combination) and out count (0, 1, 2)
2. Returns expected runs from this state to end of inning
3. Returns probability of scoring at least one run
4. Returns run distribution (probability of scoring exactly 0, 1, 2, 3+ runs)
5. Returns the run expectancy change for common transitions (steal, sacrifice bunt, caught stealing)
6. Values are derived from the pre-computed 24-state run expectancy matrix
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_run_expectancy import (
    get_run_expectancy,
    RE_MATRIX,
    PROB_AT_LEAST_ONE,
    RUN_DISTRIBUTION,
    _runners_key,
)


def parse(result: str) -> dict:
    return json.loads(result)


# -----------------------------------------------------------------------
# Step 1: Accepts base state and out count
# -----------------------------------------------------------------------

def test_step1_bases_empty_0_outs():
    """Step 1: Accepts bases empty with 0 outs."""
    result = parse(get_run_expectancy(False, False, False, 0))
    assert result["status"] == "ok"
    assert result["base_out_state"] == {"first": False, "second": False, "third": False, "outs": 0}


def test_step1_all_base_states():
    """Step 1: Accepts all 8 base states with all 3 out counts (24 total states)."""
    base_combos = [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]
    for first, second, third in base_combos:
        for outs in (0, 1, 2):
            result = parse(get_run_expectancy(first, second, third, outs))
            assert result["status"] == "ok", f"Failed for ({first},{second},{third},{outs})"
            assert result["base_out_state"]["first"] == first
            assert result["base_out_state"]["second"] == second
            assert result["base_out_state"]["third"] == third
            assert result["base_out_state"]["outs"] == outs


def test_step1_invalid_outs():
    """Step 1: Rejects invalid out counts."""
    for bad_outs in (-1, 3, 4, 10):
        result = parse(get_run_expectancy(False, False, False, bad_outs))
        assert result["status"] == "error"
        assert result["error_code"] == "INVALID_PARAMETER"
        assert "outs" in result["message"].lower()


# -----------------------------------------------------------------------
# Step 2: Returns expected runs from this state to end of inning
# -----------------------------------------------------------------------

def test_step2_expected_runs_present():
    """Step 2: Response includes expected_runs field."""
    result = parse(get_run_expectancy(False, False, False, 0))
    assert "expected_runs" in result
    assert isinstance(result["expected_runs"], float)


def test_step2_expected_runs_values_match_matrix():
    """Step 2: Expected runs values match the pre-computed RE matrix."""
    base_combos = [
        (False, False, False, "000"),
        (True, False, False, "100"),
        (False, True, False, "010"),
        (False, False, True, "001"),
        (True, True, False, "110"),
        (True, False, True, "101"),
        (False, True, True, "011"),
        (True, True, True, "111"),
    ]
    for first, second, third, key in base_combos:
        for outs in (0, 1, 2):
            result = parse(get_run_expectancy(first, second, third, outs))
            expected = RE_MATRIX[key][outs]
            assert result["expected_runs"] == round(expected, 3), \
                f"Mismatch for {key}/{outs}: {result['expected_runs']} != {expected}"


def test_step2_expected_runs_monotonic_with_outs():
    """Step 2: Expected runs decrease as outs increase (for any base state)."""
    for first in (True, False):
        for second in (True, False):
            for third in (True, False):
                re = [parse(get_run_expectancy(first, second, third, o))["expected_runs"] for o in (0, 1, 2)]
                assert re[0] >= re[1] >= re[2], \
                    f"RE not monotonic for ({first},{second},{third}): {re}"


def test_step2_more_runners_means_more_runs():
    """Step 2: Bases loaded should have higher RE than bases empty (same outs)."""
    for outs in (0, 1, 2):
        empty = parse(get_run_expectancy(False, False, False, outs))["expected_runs"]
        loaded = parse(get_run_expectancy(True, True, True, outs))["expected_runs"]
        assert loaded > empty, f"Loaded RE ({loaded}) not > empty RE ({empty}) at {outs} outs"


# -----------------------------------------------------------------------
# Step 3: Returns probability of scoring at least one run
# -----------------------------------------------------------------------

def test_step3_prob_scoring_present():
    """Step 3: Response includes prob_scoring_at_least_one field."""
    result = parse(get_run_expectancy(True, False, False, 0))
    assert "prob_scoring_at_least_one" in result
    assert isinstance(result["prob_scoring_at_least_one"], float)


def test_step3_prob_scoring_range():
    """Step 3: Probability of scoring is between 0 and 1 for all states."""
    for first in (True, False):
        for second in (True, False):
            for third in (True, False):
                for outs in (0, 1, 2):
                    result = parse(get_run_expectancy(first, second, third, outs))
                    prob = result["prob_scoring_at_least_one"]
                    assert 0.0 <= prob <= 1.0, \
                        f"Prob out of range for ({first},{second},{third},{outs}): {prob}"


def test_step3_prob_scoring_monotonic_with_outs():
    """Step 3: Probability of scoring decreases as outs increase."""
    for first in (True, False):
        for second in (True, False):
            for third in (True, False):
                probs = [
                    parse(get_run_expectancy(first, second, third, o))["prob_scoring_at_least_one"]
                    for o in (0, 1, 2)
                ]
                assert probs[0] >= probs[1] >= probs[2], \
                    f"Prob scoring not monotonic for ({first},{second},{third}): {probs}"


def test_step3_runner_on_third_high_prob():
    """Step 3: Runner on 3rd with 0 outs should have very high probability of scoring."""
    result = parse(get_run_expectancy(False, False, True, 0))
    assert result["prob_scoring_at_least_one"] > 0.75


def test_step3_bases_empty_2_outs_low_prob():
    """Step 3: Bases empty with 2 outs should have low probability."""
    result = parse(get_run_expectancy(False, False, False, 2))
    assert result["prob_scoring_at_least_one"] < 0.15


# -----------------------------------------------------------------------
# Step 4: Returns run distribution
# -----------------------------------------------------------------------

def test_step4_run_distribution_present():
    """Step 4: Response includes run_distribution with all 4 probabilities."""
    result = parse(get_run_expectancy(False, True, False, 1))
    dist = result["run_distribution"]
    assert "prob_0_runs" in dist
    assert "prob_1_run" in dist
    assert "prob_2_runs" in dist
    assert "prob_3_plus_runs" in dist


def test_step4_run_distribution_sums_to_one():
    """Step 4: Run distribution probabilities sum to approximately 1.0."""
    for first in (True, False):
        for second in (True, False):
            for third in (True, False):
                for outs in (0, 1, 2):
                    result = parse(get_run_expectancy(first, second, third, outs))
                    dist = result["run_distribution"]
                    total = dist["prob_0_runs"] + dist["prob_1_run"] + dist["prob_2_runs"] + dist["prob_3_plus_runs"]
                    assert abs(total - 1.0) < 0.01, \
                        f"Distribution doesn't sum to 1.0 for ({first},{second},{third},{outs}): {total}"


def test_step4_run_distribution_consistent_with_prob_scoring():
    """Step 4: P(0 runs) should be consistent with P(scoring at least one)."""
    for first in (True, False):
        for second in (True, False):
            for third in (True, False):
                for outs in (0, 1, 2):
                    result = parse(get_run_expectancy(first, second, third, outs))
                    prob_score = result["prob_scoring_at_least_one"]
                    prob_0 = result["run_distribution"]["prob_0_runs"]
                    # P(scoring) = 1 - P(0 runs)
                    assert abs(prob_score - (1.0 - prob_0)) < 0.01, \
                        f"Inconsistency for ({first},{second},{third},{outs}): " \
                        f"prob_score={prob_score}, 1-prob_0={1.0-prob_0}"


def test_step4_run_distribution_all_non_negative():
    """Step 4: All distribution probabilities are non-negative."""
    for first in (True, False):
        for second in (True, False):
            for third in (True, False):
                for outs in (0, 1, 2):
                    result = parse(get_run_expectancy(first, second, third, outs))
                    dist = result["run_distribution"]
                    for k, v in dist.items():
                        assert v >= 0.0, f"Negative prob {k}={v} for ({first},{second},{third},{outs})"


def test_step4_bases_loaded_0_outs_high_multi_run_prob():
    """Step 4: Bases loaded, 0 outs should have significant probability of 3+ runs."""
    result = parse(get_run_expectancy(True, True, True, 0))
    dist = result["run_distribution"]
    assert dist["prob_3_plus_runs"] > 0.20


# -----------------------------------------------------------------------
# Step 5: Returns RE change for common transitions
# -----------------------------------------------------------------------

def test_step5_transitions_present():
    """Step 5: Response includes transitions dict."""
    result = parse(get_run_expectancy(True, False, False, 0))
    assert "transitions" in result
    assert isinstance(result["transitions"], dict)


def test_step5_steal_2b_transitions():
    """Step 5: Runner on 1st (2nd open) should have steal_2b transitions."""
    result = parse(get_run_expectancy(True, False, False, 0))
    trans = result["transitions"]
    assert "steal_2b_success" in trans
    assert "steal_2b_caught" in trans
    assert "steal_2b_breakeven" in trans
    # Success should have positive RE change (moving from 1st to 2nd gains value)
    assert trans["steal_2b_success"]["re_change"] > 0
    # Caught stealing should have negative RE change
    assert trans["steal_2b_caught"]["re_change"] < 0
    # Breakeven rate should be between 0 and 1
    assert 0.0 < trans["steal_2b_breakeven"]["rate"] < 1.0


def test_step5_steal_3b_transitions():
    """Step 5: Runner on 2nd (3rd open) should have steal_3b transitions."""
    result = parse(get_run_expectancy(False, True, False, 0))
    trans = result["transitions"]
    assert "steal_3b_success" in trans
    assert "steal_3b_caught" in trans
    assert "steal_3b_breakeven" in trans
    # Stealing 3rd is harder to justify -- higher breakeven
    assert trans["steal_3b_breakeven"]["rate"] > 0.5


def test_step5_no_steal_2b_when_2nd_occupied():
    """Step 5: No steal_2b transition when 2nd base is already occupied."""
    result = parse(get_run_expectancy(True, True, False, 0))
    trans = result["transitions"]
    assert "steal_2b_success" not in trans


def test_step5_no_steal_3b_when_3rd_occupied():
    """Step 5: No steal_3b transition when 3rd base is already occupied."""
    result = parse(get_run_expectancy(False, True, True, 1))
    trans = result["transitions"]
    assert "steal_3b_success" not in trans


def test_step5_no_steal_3b_with_2_outs():
    """Step 5: No steal_3b transition with 2 outs (too risky to compute)."""
    result = parse(get_run_expectancy(False, True, False, 2))
    trans = result["transitions"]
    assert "steal_3b_success" not in trans


def test_step5_sacrifice_bunt_runner_on_1st():
    """Step 5: Sac bunt transition for runner on 1st, 0 outs."""
    result = parse(get_run_expectancy(True, False, False, 0))
    trans = result["transitions"]
    assert "sac_bunt_1st_to_2nd" in trans
    # Sac bunt from 1st with 0 outs typically has negative RE change
    # (trading an out for a base is usually bad for RE)
    bunt_change = trans["sac_bunt_1st_to_2nd"]["re_change"]
    assert isinstance(bunt_change, float)


def test_step5_sacrifice_bunt_runners_1st_2nd():
    """Step 5: Sac bunt transition for runners on 1st and 2nd."""
    result = parse(get_run_expectancy(True, True, False, 0))
    trans = result["transitions"]
    assert "sac_bunt_1st2nd_to_2nd3rd" in trans


def test_step5_sacrifice_bunt_runner_on_2nd():
    """Step 5: Sac bunt transition for runner on 2nd only."""
    result = parse(get_run_expectancy(False, True, False, 0))
    trans = result["transitions"]
    assert "sac_bunt_2nd_to_3rd" in trans


def test_step5_no_bunt_with_2_outs():
    """Step 5: No sacrifice bunt transitions with 2 outs."""
    result = parse(get_run_expectancy(True, False, False, 2))
    trans = result["transitions"]
    assert "sac_bunt_1st_to_2nd" not in trans


def test_step5_double_play_transition():
    """Step 5: Double play transition when runner on 1st with < 2 outs."""
    result = parse(get_run_expectancy(True, False, False, 0))
    trans = result["transitions"]
    assert "double_play" in trans
    # GIDP should have large negative RE change
    assert trans["double_play"]["re_change"] < -0.5


def test_step5_no_double_play_with_2_outs():
    """Step 5: No DP transition with 2 outs (3rd out ends inning regardless)."""
    result = parse(get_run_expectancy(True, False, False, 2))
    trans = result["transitions"]
    assert "double_play" not in trans


def test_step5_wild_pitch_with_runner_on_third():
    """Step 5: Wild pitch transition when runner on 3rd with < 2 outs."""
    result = parse(get_run_expectancy(False, False, True, 0))
    trans = result["transitions"]
    assert "wild_pitch_run_scores" in trans
    # Should be positive (run scores)
    assert trans["wild_pitch_run_scores"]["re_change"] > 0


def test_step5_no_transitions_bases_empty():
    """Step 5: No steal/bunt/DP transitions when bases are empty."""
    result = parse(get_run_expectancy(False, False, False, 0))
    trans = result["transitions"]
    assert len(trans) == 0


def test_step5_steal_2b_breakeven_known_value():
    """Step 5: Steal 2B breakeven from runner on 1st, 0 outs should be ~70-75%."""
    result = parse(get_run_expectancy(True, False, False, 0))
    be = result["transitions"]["steal_2b_breakeven"]["rate"]
    assert 0.60 < be < 0.85, f"Breakeven {be} outside expected range for 1st/0out"


# -----------------------------------------------------------------------
# Step 6: Values derived from pre-computed RE matrix
# -----------------------------------------------------------------------

def test_step6_re_matrix_has_all_24_states():
    """Step 6: The RE matrix has all 8 base states."""
    assert len(RE_MATRIX) == 8
    for key in ("000", "100", "010", "001", "110", "101", "011", "111"):
        assert key in RE_MATRIX
        assert len(RE_MATRIX[key]) == 3  # 0, 1, 2 outs


def test_step6_re_matrix_values_realistic():
    """Step 6: RE values are within realistic MLB ranges."""
    for key, values in RE_MATRIX.items():
        for outs, re in enumerate(values):
            assert 0.0 < re < 3.0, f"RE {re} for {key}/{outs} out of range"


def test_step6_prob_scoring_matrix_complete():
    """Step 6: Probability-of-scoring matrix covers all 24 states."""
    assert len(PROB_AT_LEAST_ONE) == 8
    for key in ("000", "100", "010", "001", "110", "101", "011", "111"):
        assert key in PROB_AT_LEAST_ONE
        assert len(PROB_AT_LEAST_ONE[key]) == 3


def test_step6_run_distribution_matrix_complete():
    """Step 6: Run distribution matrix covers all 24 states."""
    assert len(RUN_DISTRIBUTION) == 8
    for key in ("000", "100", "010", "001", "110", "101", "011", "111"):
        assert key in RUN_DISTRIBUTION
        assert len(RUN_DISTRIBUTION[key]) == 3
        for outs in range(3):
            assert len(RUN_DISTRIBUTION[key][outs]) == 4  # 0, 1, 2, 3+ runs


# -----------------------------------------------------------------------
# Additional integration and edge case tests
# -----------------------------------------------------------------------

def test_response_structure():
    """Full response has all required top-level fields."""
    result = parse(get_run_expectancy(True, True, True, 0))
    assert "status" in result
    assert "base_out_state" in result
    assert "expected_runs" in result
    assert "prob_scoring_at_least_one" in result
    assert "run_distribution" in result
    assert "transitions" in result


def test_runners_key_helper():
    """Internal helper _runners_key produces correct strings."""
    assert _runners_key(False, False, False) == "000"
    assert _runners_key(True, False, False) == "100"
    assert _runners_key(False, True, False) == "010"
    assert _runners_key(False, False, True) == "001"
    assert _runners_key(True, True, True) == "111"


def test_multiple_transitions_runner_on_first_0_outs():
    """Runner on 1st with 0 outs should have steal, bunt, and DP transitions."""
    result = parse(get_run_expectancy(True, False, False, 0))
    trans = result["transitions"]
    assert "steal_2b_success" in trans
    assert "steal_2b_caught" in trans
    assert "steal_2b_breakeven" in trans
    assert "sac_bunt_1st_to_2nd" in trans
    assert "double_play" in trans


def test_steal_2b_with_runner_on_third():
    """Steal 2B with runner on 1st and 3rd (1st & 3rd play)."""
    result = parse(get_run_expectancy(True, False, True, 0))
    trans = result["transitions"]
    # Should still have steal_2b (2nd is open)
    assert "steal_2b_success" in trans
    # Success moves 1st to 2nd, runner stays on 3rd
    assert trans["steal_2b_success"]["re_change"] > 0


def test_json_parseable():
    """All 24 states return valid JSON."""
    for first in (True, False):
        for second in (True, False):
            for third in (True, False):
                for outs in (0, 1, 2):
                    raw = get_run_expectancy(first, second, third, outs)
                    data = json.loads(raw)
                    assert isinstance(data, dict)


if __name__ == "__main__":
    tests = [
        test_step1_bases_empty_0_outs,
        test_step1_all_base_states,
        test_step1_invalid_outs,
        test_step2_expected_runs_present,
        test_step2_expected_runs_values_match_matrix,
        test_step2_expected_runs_monotonic_with_outs,
        test_step2_more_runners_means_more_runs,
        test_step3_prob_scoring_present,
        test_step3_prob_scoring_range,
        test_step3_prob_scoring_monotonic_with_outs,
        test_step3_runner_on_third_high_prob,
        test_step3_bases_empty_2_outs_low_prob,
        test_step4_run_distribution_present,
        test_step4_run_distribution_sums_to_one,
        test_step4_run_distribution_consistent_with_prob_scoring,
        test_step4_run_distribution_all_non_negative,
        test_step4_bases_loaded_0_outs_high_multi_run_prob,
        test_step5_transitions_present,
        test_step5_steal_2b_transitions,
        test_step5_steal_3b_transitions,
        test_step5_no_steal_2b_when_2nd_occupied,
        test_step5_no_steal_3b_when_3rd_occupied,
        test_step5_no_steal_3b_with_2_outs,
        test_step5_sacrifice_bunt_runner_on_1st,
        test_step5_sacrifice_bunt_runners_1st_2nd,
        test_step5_sacrifice_bunt_runner_on_2nd,
        test_step5_no_bunt_with_2_outs,
        test_step5_double_play_transition,
        test_step5_no_double_play_with_2_outs,
        test_step5_wild_pitch_with_runner_on_third,
        test_step5_no_transitions_bases_empty,
        test_step5_steal_2b_breakeven_known_value,
        test_step6_re_matrix_has_all_24_states,
        test_step6_re_matrix_values_realistic,
        test_step6_prob_scoring_matrix_complete,
        test_step6_run_distribution_matrix_complete,
        test_response_structure,
        test_runners_key_helper,
        test_multiple_transitions_runner_on_first_0_outs,
        test_steal_2b_with_runner_on_third,
        test_json_parseable,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
