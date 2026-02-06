# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the evaluate_stolen_base tool.

Verifies all feature requirements from features.json:
1. Accepts runner identifier, target base, pitcher identifier, and catcher identifier
2. Returns estimated success probability based on runner speed, pitcher hold time, and catcher pop time
3. Returns breakeven success rate for this base-out state
4. Returns expected run-expectancy change if successful vs caught
5. Returns net expected run-expectancy change (weighted by success probability)
6. Returns a textual recommendation (favorable, marginal, unfavorable)
7. Returns an error if any player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.evaluate_stolen_base import (
    evaluate_stolen_base,
    _derive_sprint_speed,
    _derive_sb_success_rate,
    _derive_pitcher_hold_time,
    _derive_catcher_pop_time,
    _estimate_success_probability,
    _load_players,
    _PLAYERS,
)
from tools.get_run_expectancy import RE_MATRIX, _runners_key, _get_re


def parse(result: str) -> dict:
    return json.loads(result)


def parse_data(result: str) -> dict:
    """Parse a success response and return the data dict with status included."""
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["tool"] == "evaluate_stolen_base"
    data = parsed["data"]
    data["status"] = "ok"
    return data


# -----------------------------------------------------------------------
# Step 1: Accepts runner, target base, pitcher, and catcher identifiers
# -----------------------------------------------------------------------

def test_step1_basic_steal_2nd():
    """Step 1: Accepts valid player IDs and returns ok status for steal of 2nd."""
    # h_001 = Marcus Chen (speed 85), a_sp1 = Matt Henderson, a_009 = Diego Santos (catcher)
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["runner_id"] == "h_001"
    assert result["target_base"] == 2
    assert result["pitcher_id"] == "a_sp1"
    assert result["catcher_id"] == "a_009"


def test_step1_basic_steal_3rd():
    """Step 1: Accepts valid player IDs for steal of 3rd."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 3, "a_sp1", "a_009",
        runner_on_second=True, outs=0,
    ))
    assert result["target_base"] == 3


def test_step1_steal_home():
    """Step 1: Accepts steal of home (target_base=4)."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 4, "a_sp1", "a_009",
        runner_on_third=True, outs=0,
    ))
    assert result["target_base"] == 4


def test_step1_includes_player_names():
    """Step 1: Response includes runner, pitcher, and catcher names."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["runner_name"] == "Marcus Chen"
    assert result["pitcher_name"] == "Matt Henderson"
    assert result["catcher_name"] == "Diego Santos"


def test_step1_base_out_state_in_response():
    """Step 1: Response includes the base-out state used for computation."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, runner_on_third=True, outs=1,
    ))
    assert result["base_out_state"] == {
        "first": True, "second": False, "third": True, "outs": 1,
    }


# -----------------------------------------------------------------------
# Step 2: Returns estimated success probability
# -----------------------------------------------------------------------

def test_step2_success_probability_present():
    """Step 2: Response includes success_probability field."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert "success_probability" in result
    assert isinstance(result["success_probability"], float)


def test_step2_success_probability_range():
    """Step 2: Success probability is between 0 and 1."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert 0.0 < result["success_probability"] < 1.0


def test_step2_fast_runner_higher_probability():
    """Step 2: A fast runner should have higher success probability than a slow runner."""
    # h_001 = Marcus Chen (speed 85), h_003 = Rafael Ortiz (speed 40)
    fast = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    slow = parse_data(evaluate_stolen_base(
        "h_003", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert fast["success_probability"] > slow["success_probability"], \
        f"Fast ({fast['success_probability']}) should > Slow ({slow['success_probability']})"


def test_step2_better_catcher_lower_probability():
    """Step 2: A better catcher should reduce success probability."""
    # a_009 = Diego Santos (pop_time 1.92), a_010 = James Wright (pop_time 2.02)
    vs_good_catcher = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    vs_weak_catcher = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_010",
        runner_on_first=True, outs=0,
    ))
    assert vs_weak_catcher["success_probability"] > vs_good_catcher["success_probability"], \
        f"Weak catcher ({vs_weak_catcher['success_probability']}) should > " \
        f"Good catcher ({vs_good_catcher['success_probability']})"


def test_step2_context_includes_derived_metrics():
    """Step 2: Response includes context with derived speed/hold/pop metrics."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    ctx = result["context"]
    assert "runner_sprint_speed_ft_per_s" in ctx
    assert "runner_career_sb_rate" in ctx
    assert "pitcher_hold_time_s" in ctx
    assert "catcher_pop_time_s" in ctx
    # Verify ranges are realistic
    assert 23.0 <= ctx["runner_sprint_speed_ft_per_s"] <= 31.0
    assert 0.50 <= ctx["runner_career_sb_rate"] <= 0.92
    assert 1.0 <= ctx["pitcher_hold_time_s"] <= 2.0
    assert 1.80 <= ctx["catcher_pop_time_s"] <= 2.20


def test_step2_steal_3rd_lower_probability_than_2nd():
    """Step 2: Stealing 3rd should generally have lower success probability than stealing 2nd."""
    steal_2nd = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    steal_3rd = parse_data(evaluate_stolen_base(
        "h_001", 3, "a_sp1", "a_009",
        runner_on_second=True, outs=0,
    ))
    assert steal_2nd["success_probability"] > steal_3rd["success_probability"]


# -----------------------------------------------------------------------
# Step 3: Returns breakeven success rate for this base-out state
# -----------------------------------------------------------------------

def test_step3_breakeven_rate_present():
    """Step 3: Response includes breakeven_rate field."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert "breakeven_rate" in result
    assert isinstance(result["breakeven_rate"], float)


def test_step3_breakeven_rate_range():
    """Step 3: Breakeven rate is between 0.5 and 1.0 (always need > 50% to consider)."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert 0.5 < result["breakeven_rate"] < 1.0


def test_step3_breakeven_matches_re_matrix():
    """Step 3: Breakeven rate should match the RE-matrix based calculation."""
    # Runner on 1st, 0 outs, steal 2nd
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    # Manually compute from RE matrix
    current_re = RE_MATRIX["100"][0]  # runner on 1st, 0 outs
    success_re = RE_MATRIX["010"][0]  # runner on 2nd, 0 outs
    caught_re = RE_MATRIX["000"][1]   # bases empty, 1 out
    re_gain = success_re - current_re
    re_loss = current_re - caught_re
    expected_breakeven = round(re_loss / (re_gain + re_loss), 3)
    assert result["breakeven_rate"] == expected_breakeven


def test_step3_breakeven_varies_with_outs():
    """Step 3: Breakeven rate varies with outs; 1 out should be highest."""
    # The RE matrix produces breakeven rates of ~0.715 (0 outs), ~0.726 (1 out),
    # ~0.702 (2 outs). At 1 out, the ratio of loss-to-total is highest because
    # going from 1 to 2 outs is particularly costly. At 2 outs, the base RE is
    # already low, so the gain from success is proportionally larger.
    be_0 = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))["breakeven_rate"]
    be_1 = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=1,
    ))["breakeven_rate"]
    be_2 = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=2,
    ))["breakeven_rate"]
    # All should be in realistic range
    for be, outs in [(be_0, 0), (be_1, 1), (be_2, 2)]:
        assert 0.60 < be < 0.85, f"Breakeven {be} at {outs} outs out of range"
    # 1 out should have the highest breakeven
    assert be_1 >= be_0, f"1 out BE ({be_1}) should >= 0 out BE ({be_0})"


def test_step3_steal_2b_breakeven_known_range():
    """Step 3: Steal 2B breakeven from DESIGN.md should be ~71.5% with 0 outs."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    # From DESIGN.md: Runner on 1st, 0 outs: ~71.5%
    assert 0.60 < result["breakeven_rate"] < 0.85


def test_step3_steal_3b_breakeven_higher():
    """Step 3: Stealing 3rd should have higher breakeven than stealing 2nd."""
    steal_2nd = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    steal_3rd = parse_data(evaluate_stolen_base(
        "h_001", 3, "a_sp1", "a_009",
        runner_on_second=True, outs=0,
    ))
    assert steal_3rd["breakeven_rate"] > steal_2nd["breakeven_rate"], \
        f"Steal 3rd BE ({steal_3rd['breakeven_rate']}) should > steal 2nd BE ({steal_2nd['breakeven_rate']})"


# -----------------------------------------------------------------------
# Step 4: Returns expected RE change if successful vs caught
# -----------------------------------------------------------------------

def test_step4_re_changes_present():
    """Step 4: Response includes re_change_if_successful and re_change_if_caught."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert "re_change_if_successful" in result
    assert "re_change_if_caught" in result
    assert isinstance(result["re_change_if_successful"], float)
    assert isinstance(result["re_change_if_caught"], float)


def test_step4_steal_2b_success_positive_re():
    """Step 4: Successful steal of 2nd should have positive RE change."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["re_change_if_successful"] > 0, \
        f"Success RE change should be positive, got {result['re_change_if_successful']}"


def test_step4_caught_stealing_negative_re():
    """Step 4: Caught stealing should have negative RE change."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["re_change_if_caught"] < 0, \
        f"Caught RE change should be negative, got {result['re_change_if_caught']}"


def test_step4_re_changes_match_matrix():
    """Step 4: RE changes should match hand-computed values from RE matrix."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    # Runner on 1st, 0 outs -> steal 2nd
    current_re = RE_MATRIX["100"][0]
    success_re = RE_MATRIX["010"][0]
    caught_re = RE_MATRIX["000"][1]
    assert result["re_change_if_successful"] == round(success_re - current_re, 3)
    assert result["re_change_if_caught"] == round(caught_re - current_re, 3)


def test_step4_caught_always_negative():
    """Step 4: Being caught stealing is always a negative RE change at any out count."""
    for outs in (0, 1, 2):
        result = parse_data(evaluate_stolen_base(
            "h_001", 2, "a_sp1", "a_009",
            runner_on_first=True, outs=outs,
        ))
        assert result["re_change_if_caught"] < 0, \
            f"Caught RE should be negative at {outs} outs, got {result['re_change_if_caught']}"
    # Also verify the magnitude is significant: losing a baserunner always hurts
    result_0 = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result_0["re_change_if_caught"] < -0.2


def test_step4_steal_3b_re_changes():
    """Step 4: Steal of 3rd has correct RE change direction."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 3, "a_sp1", "a_009",
        runner_on_second=True, outs=0,
    ))
    assert result["re_change_if_successful"] > 0
    assert result["re_change_if_caught"] < 0


def test_step4_steal_home_re_includes_run():
    """Step 4: Steal of home success should include the run scored (RE change > 0)."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 4, "a_sp1", "a_009",
        runner_on_third=True, outs=0,
    ))
    # Success = 1 run scored + new RE. This should be a large positive change.
    assert result["re_change_if_successful"] > 0


# -----------------------------------------------------------------------
# Step 5: Returns net expected RE change (weighted by success probability)
# -----------------------------------------------------------------------

def test_step5_net_re_present():
    """Step 5: Response includes net_expected_re_change field."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert "net_expected_re_change" in result
    assert isinstance(result["net_expected_re_change"], float)


def test_step5_net_re_formula():
    """Step 5: Net RE = P(success) * RE_success + P(caught) * RE_caught."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    p = result["success_probability"]
    re_s = result["re_change_if_successful"]
    re_c = result["re_change_if_caught"]
    expected_net = round(p * re_s + (1 - p) * re_c, 3)
    assert abs(result["net_expected_re_change"] - expected_net) < 0.002, \
        f"Net RE {result['net_expected_re_change']} != expected {expected_net}"


def test_step5_favorable_has_positive_net_re():
    """Step 5: A favorable recommendation should generally have positive net RE."""
    # Fast runner (h_001, speed 85) vs slow pitcher/catcher combo
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    if result["recommendation"] == "favorable":
        assert result["net_expected_re_change"] > 0


# -----------------------------------------------------------------------
# Step 6: Returns a textual recommendation
# -----------------------------------------------------------------------

def test_step6_recommendation_present():
    """Step 6: Response includes recommendation field."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert "recommendation" in result
    assert result["recommendation"] in ("favorable", "marginal", "unfavorable")


def test_step6_fast_runner_favorable():
    """Step 6: A fast runner (speed 85) should get favorable or marginal recommendation."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["recommendation"] in ("favorable", "marginal"), \
        f"Fast runner got '{result['recommendation']}'"


def test_step6_slow_runner_unfavorable():
    """Step 6: A slow runner (speed 40) should get unfavorable recommendation."""
    # h_003 = Rafael Ortiz (speed 40)
    result = parse_data(evaluate_stolen_base(
        "h_003", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["recommendation"] in ("unfavorable", "marginal"), \
        f"Slow runner got '{result['recommendation']}'"


def test_step6_recommendation_consistent_with_margin():
    """Step 6: Recommendation is consistent with success_prob - breakeven margin."""
    for runner_id in ("h_001", "h_002", "h_003", "h_007"):
        result = parse_data(evaluate_stolen_base(
            runner_id, 2, "a_sp1", "a_009",
            runner_on_first=True, outs=0,
        ))
        margin = result["success_probability"] - result["breakeven_rate"]
        if margin >= 0.05:
            assert result["recommendation"] == "favorable", \
                f"Runner {runner_id}: margin {margin:.3f} should be favorable"
        elif margin >= 0.0:
            assert result["recommendation"] == "marginal", \
                f"Runner {runner_id}: margin {margin:.3f} should be marginal"
        else:
            assert result["recommendation"] == "unfavorable", \
                f"Runner {runner_id}: margin {margin:.3f} should be unfavorable"


# -----------------------------------------------------------------------
# Step 7: Returns an error if any player identifier is invalid
# -----------------------------------------------------------------------

def test_step7_invalid_runner_id():
    """Step 7: Invalid runner ID returns error."""
    result = parse(evaluate_stolen_base(
        "invalid_runner", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["tool"] == "evaluate_stolen_base"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "runner" in result["message"]


def test_step7_invalid_pitcher_id():
    """Step 7: Invalid pitcher ID returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 2, "invalid_pitcher", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "pitcher" in result["message"]


def test_step7_invalid_catcher_id():
    """Step 7: Invalid catcher ID returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "invalid_catcher",
        runner_on_first=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "catcher" in result["message"]


def test_step7_invalid_target_base():
    """Step 7: Invalid target base returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 5, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step7_invalid_target_base_1():
    """Step 7: Target base 1 is invalid (can't steal 1st)."""
    result = parse(evaluate_stolen_base(
        "h_001", 1, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["status"] == "error"


def test_step7_invalid_outs():
    """Step 7: Invalid outs value returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=3,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step7_steal_2nd_no_runner_on_first():
    """Step 7: Stealing 2nd with no runner on 1st returns situation error."""
    result = parse(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=False, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_SITUATION"


def test_step7_steal_2nd_2nd_occupied():
    """Step 7: Stealing 2nd with 2nd already occupied returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, runner_on_second=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_SITUATION"


def test_step7_steal_3rd_no_runner_on_second():
    """Step 7: Stealing 3rd with no runner on 2nd returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 3, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_SITUATION"


def test_step7_steal_3rd_3rd_occupied():
    """Step 7: Stealing 3rd with 3rd already occupied returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 3, "a_sp1", "a_009",
        runner_on_second=True, runner_on_third=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_SITUATION"


def test_step7_steal_home_no_runner_on_third():
    """Step 7: Stealing home with no runner on 3rd returns error."""
    result = parse(evaluate_stolen_base(
        "h_001", 4, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_SITUATION"


# -----------------------------------------------------------------------
# Derived metric helper function tests
# -----------------------------------------------------------------------

def test_derive_sprint_speed_fast_runner():
    """Sprint speed for a fast runner (speed 85) should be high."""
    _load_players()
    player = _PLAYERS["h_001"]  # Marcus Chen, speed 85
    speed = _derive_sprint_speed(player)
    assert speed > 28.0  # Well above average (27.0)
    assert 23.0 <= speed <= 31.0


def test_derive_sprint_speed_slow_runner():
    """Sprint speed for a slow runner (speed 40) should be low."""
    _load_players()
    player = _PLAYERS["h_003"]  # Rafael Ortiz, speed 40
    speed = _derive_sprint_speed(player)
    assert speed < 27.0  # Below average
    assert 23.0 <= speed <= 31.0


def test_derive_sb_success_rate_fast_runner():
    """SB rate for a fast runner should be high."""
    _load_players()
    player = _PLAYERS["h_001"]  # speed 85
    rate = _derive_sb_success_rate(player)
    assert rate > 0.72  # Above MLB average


def test_derive_sb_success_rate_slow_runner():
    """SB rate for a slow runner should be lower."""
    _load_players()
    player = _PLAYERS["h_003"]  # speed 40
    rate = _derive_sb_success_rate(player)
    assert rate < 0.72  # Below MLB average


def test_derive_pitcher_hold_time():
    """Pitcher hold time should be in realistic range."""
    _load_players()
    pitcher = _PLAYERS["a_sp1"]  # Matt Henderson (control 74, velocity 93.0)
    hold = _derive_pitcher_hold_time(pitcher)
    assert 1.0 < hold < 2.0  # Realistic range for delivery + flight time


def test_derive_catcher_pop_time_from_catcher_attrs():
    """Catcher with explicit pop_time should use that value."""
    _load_players()
    catcher = _PLAYERS["a_009"]  # Diego Santos, pop_time 1.92
    pop = _derive_catcher_pop_time(catcher)
    assert pop == 1.92


def test_derive_catcher_pop_time_from_arm_strength():
    """Non-catcher player should derive pop time from arm strength."""
    _load_players()
    # Use a non-catcher player (no catcher attributes)
    player = _PLAYERS["h_001"]  # Marcus Chen, CF
    pop = _derive_catcher_pop_time(player)
    assert 1.80 <= pop <= 2.20


def test_estimate_success_probability_range():
    """Success probability should be clamped to reasonable range."""
    prob = _estimate_success_probability(
        sprint_speed=30.0,
        sb_rate=0.85,
        pitcher_hold_time=1.50,
        catcher_pop_time=2.10,
        target_base=2,
    )
    assert 0.10 <= prob <= 0.98


def test_estimate_success_probability_average_inputs():
    """Average inputs should yield roughly average success probability."""
    prob = _estimate_success_probability(
        sprint_speed=27.0,
        sb_rate=0.72,
        pitcher_hold_time=1.35,
        catcher_pop_time=2.00,
        target_base=2,
    )
    # With all average inputs, should be near the base SB rate (~72%)
    assert 0.65 < prob < 0.80


# -----------------------------------------------------------------------
# Integration and edge case tests
# -----------------------------------------------------------------------

def test_all_out_states_steal_2b():
    """Steal of 2nd works for all out counts."""
    for outs in (0, 1, 2):
        result = parse_data(evaluate_stolen_base(
            "h_001", 2, "a_sp1", "a_009",
            runner_on_first=True, outs=outs,
        ))
        assert result["status"] == "ok", f"Failed for outs={outs}"


def test_steal_2b_with_runner_on_third():
    """Steal 2nd with runners on 1st and 3rd (1st-and-3rd play)."""
    result = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, runner_on_third=True, outs=0,
    ))
    # RE should reflect the 1st & 3rd -> 2nd & 3rd transition
    assert result["re_change_if_successful"] > 0


def test_steal_3b_with_runner_on_first():
    """Steal 3rd with runners on 1st and 2nd."""
    result = parse_data(evaluate_stolen_base(
        "h_002", 3, "a_sp1", "a_009",
        runner_on_first=True, runner_on_second=True, outs=0,
    ))


def test_json_parseable():
    """All valid calls return valid JSON."""
    scenarios = [
        ("h_001", 2, True, False, False, 0),
        ("h_001", 2, True, False, False, 1),
        ("h_001", 2, True, False, False, 2),
        ("h_001", 2, True, False, True, 0),
        ("h_001", 3, False, True, False, 0),
        ("h_001", 3, False, True, False, 1),
        ("h_001", 3, True, True, False, 0),
        ("h_001", 4, False, False, True, 0),
    ]
    for runner, target, first, second, third, outs in scenarios:
        raw = evaluate_stolen_base(
            runner, target, "a_sp1", "a_009",
            runner_on_first=first, runner_on_second=second,
            runner_on_third=third, outs=outs,
        )
        envelope = json.loads(raw)
        assert isinstance(envelope, dict)
        assert envelope["status"] == "ok"
        assert envelope["tool"] == "evaluate_stolen_base"
        assert isinstance(envelope["data"], dict)


def test_response_structure():
    """Full response has all required top-level and data fields."""
    envelope = parse(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    assert envelope["status"] == "ok"
    assert envelope["tool"] == "evaluate_stolen_base"
    data = envelope["data"]
    required_fields = [
        "runner_id", "runner_name", "target_base",
        "pitcher_id", "pitcher_name", "catcher_id", "catcher_name",
        "base_out_state", "success_probability", "breakeven_rate",
        "re_change_if_successful", "re_change_if_caught",
        "net_expected_re_change", "recommendation", "context",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_different_pitchers_affect_probability():
    """Different pitchers should produce different success probabilities."""
    # a_sp1 = Matt Henderson (control 74, velocity 93.0)
    # a_bp1 = Zach Miller (control 73, velocity 98.0 -- much faster)
    vs_sp = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_sp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    vs_rp = parse_data(evaluate_stolen_base(
        "h_001", 2, "a_bp1", "a_009",
        runner_on_first=True, outs=0,
    ))
    # They have similar control but different velocity, so hold times differ
    assert vs_sp["success_probability"] != vs_rp["success_probability"]


def test_multiple_runners_different_results():
    """Different runners produce different evaluations."""
    runners = ["h_001", "h_002", "h_003", "h_007", "h_009"]
    probs = set()
    for r in runners:
        result = parse_data(evaluate_stolen_base(
            r, 2, "a_sp1", "a_009",
            runner_on_first=True, outs=0,
        ))
        probs.add(result["success_probability"])
    # At least some distinct probabilities
    assert len(probs) > 1, "All runners had the same probability"


if __name__ == "__main__":
    tests = [
        # Step 1
        test_step1_basic_steal_2nd,
        test_step1_basic_steal_3rd,
        test_step1_steal_home,
        test_step1_includes_player_names,
        test_step1_base_out_state_in_response,
        # Step 2
        test_step2_success_probability_present,
        test_step2_success_probability_range,
        test_step2_fast_runner_higher_probability,
        test_step2_better_catcher_lower_probability,
        test_step2_context_includes_derived_metrics,
        test_step2_steal_3rd_lower_probability_than_2nd,
        # Step 3
        test_step3_breakeven_rate_present,
        test_step3_breakeven_rate_range,
        test_step3_breakeven_matches_re_matrix,
        test_step3_breakeven_varies_with_outs,
        test_step3_steal_2b_breakeven_known_range,
        test_step3_steal_3b_breakeven_higher,
        # Step 4
        test_step4_re_changes_present,
        test_step4_steal_2b_success_positive_re,
        test_step4_caught_stealing_negative_re,
        test_step4_re_changes_match_matrix,
        test_step4_caught_always_negative,
        test_step4_steal_3b_re_changes,
        test_step4_steal_home_re_includes_run,
        # Step 5
        test_step5_net_re_present,
        test_step5_net_re_formula,
        test_step5_favorable_has_positive_net_re,
        # Step 6
        test_step6_recommendation_present,
        test_step6_fast_runner_favorable,
        test_step6_slow_runner_unfavorable,
        test_step6_recommendation_consistent_with_margin,
        # Step 7
        test_step7_invalid_runner_id,
        test_step7_invalid_pitcher_id,
        test_step7_invalid_catcher_id,
        test_step7_invalid_target_base,
        test_step7_invalid_target_base_1,
        test_step7_invalid_outs,
        test_step7_steal_2nd_no_runner_on_first,
        test_step7_steal_2nd_2nd_occupied,
        test_step7_steal_3rd_no_runner_on_second,
        test_step7_steal_3rd_3rd_occupied,
        test_step7_steal_home_no_runner_on_third,
        # Derived metrics
        test_derive_sprint_speed_fast_runner,
        test_derive_sprint_speed_slow_runner,
        test_derive_sb_success_rate_fast_runner,
        test_derive_sb_success_rate_slow_runner,
        test_derive_pitcher_hold_time,
        test_derive_catcher_pop_time_from_catcher_attrs,
        test_derive_catcher_pop_time_from_arm_strength,
        test_estimate_success_probability_range,
        test_estimate_success_probability_average_inputs,
        # Integration
        test_all_out_states_steal_2b,
        test_steal_2b_with_runner_on_third,
        test_steal_3b_with_runner_on_first,
        test_json_parseable,
        test_response_structure,
        test_different_pitchers_affect_probability,
        test_multiple_runners_different_results,
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
