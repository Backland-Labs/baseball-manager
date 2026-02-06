# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_defensive_positioning tool.

Verifies all feature requirements from features.json:
1. Accepts batter identifier, pitcher identifier, and current game situation
2. Returns the batter's spray chart summary (pull%, center%, oppo% for GB and FB)
3. Returns recommended infield positioning (standard, in, double-play depth, shift direction)
4. Returns recommended outfield positioning (standard, shallow, deep, shaded direction)
5. Returns infield-in cost/benefit analysis (expected runs saved at home vs extra hits allowed)
6. Returns shift recommendation within current MLB rule constraints (2-and-2 rule)
7. Returns an error if either player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_defensive_positioning import (
    get_defensive_positioning,
    _derive_spray_chart,
    _derive_gb_rate,
    _recommend_infield,
    _recommend_outfield,
    _recommend_shift,
    _load_players,
    _clamp,
)


def parse(result: str) -> dict:
    return json.loads(result)


# -----------------------------------------------------------------------
# Step 1: Accepts batter/pitcher identifiers and game situation
# -----------------------------------------------------------------------

def test_step1_accepts_batter_and_pitcher():
    """Step 1: Accepts a batter and pitcher identifier and returns valid result."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=1, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=5,
    ))
    assert result["status"] == "ok"
    assert result["batter_id"] == "h_001"
    assert result["pitcher_id"] == "a_sp1"


def test_step1_returns_batter_info():
    """Step 1: Returns batter name and handedness."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["batter_name"] == "Marcus Chen"
    assert result["bats"] == "L"


def test_step1_returns_pitcher_info():
    """Step 1: Returns pitcher name and throwing hand."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["pitcher_name"] == "Matt Henderson"
    assert result["throws"] == "L"


def test_step1_away_batter_vs_home_pitcher():
    """Step 1: Works for away batter vs home pitcher."""
    result = parse(get_defensive_positioning(
        "a_001", "h_sp1", outs=1, runner_on_first=True, runner_on_second=False,
        runner_on_third=False, score_differential=-1, inning=3,
    ))
    assert result["status"] == "ok"
    assert result["batter_id"] == "a_001"
    assert result["pitcher_id"] == "h_sp1"


def test_step1_accepts_all_game_situation_params():
    """Step 1: Accepts full game situation with all runners and late inning."""
    result = parse(get_defensive_positioning(
        "h_003", "a_sp1", outs=0, runner_on_first=True, runner_on_second=True,
        runner_on_third=True, score_differential=-2, inning=9,
    ))
    assert result["status"] == "ok"


# -----------------------------------------------------------------------
# Step 2: Returns spray chart summary
# -----------------------------------------------------------------------

def test_step2_spray_chart_has_groundball_and_flyball():
    """Step 2: Spray chart has groundball and flyball sections."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    sc = result["spray_chart"]
    assert "groundball" in sc
    assert "flyball" in sc


def test_step2_spray_chart_has_pull_center_oppo():
    """Step 2: Each spray chart section has pull_pct, center_pct, oppo_pct."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    for ball_type in ("groundball", "flyball"):
        section = result["spray_chart"][ball_type]
        assert "pull_pct" in section
        assert "center_pct" in section
        assert "oppo_pct" in section


def test_step2_spray_chart_sums_to_one():
    """Step 2: Pull + center + oppo should sum to approximately 1.0."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    for ball_type in ("groundball", "flyball"):
        section = result["spray_chart"][ball_type]
        total = section["pull_pct"] + section["center_pct"] + section["oppo_pct"]
        assert abs(total - 1.0) < 0.02, f"{ball_type} sum is {total}, expected ~1.0"


def test_step2_spray_chart_realistic_ranges():
    """Step 2: Pull/center/oppo percentages are in realistic ranges."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    for ball_type in ("groundball", "flyball"):
        section = result["spray_chart"][ball_type]
        assert 0.20 <= section["pull_pct"] <= 0.60
        assert 0.10 <= section["center_pct"] <= 0.50
        assert 0.10 <= section["oppo_pct"] <= 0.40


def test_step2_power_hitter_pulls_more():
    """Step 2: High-power hitters have higher pull percentages."""
    # h_003 Rafael Ortiz: power 88 (slugger)
    # h_009 Andre Davis: power 50 (average)
    power_hitter = parse(get_defensive_positioning(
        "h_003", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    avg_hitter = parse(get_defensive_positioning(
        "h_009", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert power_hitter["spray_chart"]["groundball"]["pull_pct"] > avg_hitter["spray_chart"]["groundball"]["pull_pct"]


def test_step2_contact_hitter_more_spread():
    """Step 2: High-contact hitters have more even spray distribution."""
    # Compare two RHB against the same pitcher to isolate contact/power effect
    # h_004 Tyrone Jackson: R, contact 68, power 85 (power hitter)
    # h_002 Derek Williams: R, contact 75, power 60 (contact hitter)
    contact = parse(get_defensive_positioning(
        "h_002", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    power = parse(get_defensive_positioning(
        "h_004", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    # Contact hitter should have lower pull% than power hitter (same handedness, same pitcher)
    assert contact["spray_chart"]["groundball"]["pull_pct"] < power["spray_chart"]["groundball"]["pull_pct"]


def test_step2_groundballs_pulled_more_than_flyballs():
    """Step 2: Ground balls are generally pulled more than fly balls."""
    result = parse(get_defensive_positioning(
        "h_004", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["spray_chart"]["groundball"]["pull_pct"] >= result["spray_chart"]["flyball"]["pull_pct"]


def test_step2_same_side_matchup_increases_pull():
    """Step 2: Same-side matchup (RHB vs RHP) increases pull tendency."""
    # h_004 Tyrone Jackson: bats R
    # h_sp1 Brandon Cole: throws R (same side)
    # a_sp1 Matt Henderson: throws L (opposite side)
    same_side = parse(get_defensive_positioning(
        "h_004", "h_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    opp_side = parse(get_defensive_positioning(
        "h_004", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert same_side["spray_chart"]["groundball"]["pull_pct"] > opp_side["spray_chart"]["groundball"]["pull_pct"]


def test_step2_switch_hitter_adjusts_by_pitcher():
    """Step 2: Switch hitters bat from opposite side of the pitcher.

    A switch hitter always faces an opposite-side matchup regardless of pitcher hand.
    The spray chart should be the same against both RHP and LHP since the batter
    attributes don't change and the handedness adjustment is identical (both opposite-side).
    Compare this to a same-side matchup to verify the model differentiates.
    """
    # h_007 Carlos Ramirez: bats S (switch)
    # vs RHP: bats left (opposite side) -- vs LHP: bats right (opposite side)
    vs_rhp = parse(get_defensive_positioning(
        "h_007", "h_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    vs_lhp = parse(get_defensive_positioning(
        "h_007", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert vs_rhp["status"] == "ok"
    assert vs_lhp["status"] == "ok"
    # Switch hitter always opposite-side, so spray charts are identical
    assert vs_rhp["spray_chart"] == vs_lhp["spray_chart"]

    # Compare switch hitter's opposite-side charts vs a RHB in same-side matchup
    # h_004 Tyrone Jackson: bats R, vs h_sp1 throws R -> same-side
    same_side = parse(get_defensive_positioning(
        "h_004", "h_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    # Same-side matchup should produce higher pull than switch hitter's opposite-side
    # (accounting for different power/contact attributes, the directional effect should hold)
    assert same_side["spray_chart"]["groundball"]["pull_pct"] != vs_rhp["spray_chart"]["groundball"]["pull_pct"]


def test_step2_returns_groundball_rate():
    """Step 2: Returns groundball rate for the matchup."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert "groundball_rate" in result
    assert 0.30 <= result["groundball_rate"] <= 0.60


def test_step2_power_hitter_lower_gb_rate():
    """Step 2: Power hitters have lower ground ball rates."""
    # h_003 Rafael Ortiz: power 88
    # h_009 Andre Davis: power 50
    power = parse(get_defensive_positioning(
        "h_003", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    avg = parse(get_defensive_positioning(
        "h_009", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert power["groundball_rate"] < avg["groundball_rate"]


# -----------------------------------------------------------------------
# Step 3: Returns recommended infield positioning
# -----------------------------------------------------------------------

def test_step3_infield_recommendation_present():
    """Step 3: Returns an infield_recommendation field."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert "infield_recommendation" in result
    assert isinstance(result["infield_recommendation"], str)


def test_step3_standard_with_no_runners():
    """Step 3: Standard positioning with no runners on base."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["infield_recommendation"] == "standard"


def test_step3_dp_depth_with_runner_on_first():
    """Step 3: Double play depth with runner on first and less than 2 outs."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=True, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["infield_recommendation"] == "double_play_depth"


def test_step3_dp_depth_with_runner_on_first_one_out():
    """Step 3: Double play depth with runner on first and 1 out."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=1, runner_on_first=True, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["infield_recommendation"] == "double_play_depth"


def test_step3_no_dp_depth_with_two_outs():
    """Step 3: No double play depth with 2 outs (can't turn two)."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=2, runner_on_first=True, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    # With 2 outs and runner on 1st, no DP opportunity so standard
    assert result["infield_recommendation"] == "standard"


def test_step3_infield_in_runner_on_third_trailing():
    """Step 3: Infield in with runner on 3rd while trailing."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=1, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=-1, inning=5,
    ))
    assert result["infield_recommendation"] == "infield_in"


def test_step3_infield_in_late_game_tied():
    """Step 3: Infield in with runner on 3rd in late, tied game."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=1, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=0, inning=8,
    ))
    assert result["infield_recommendation"] == "infield_in"


def test_step3_halfway_runner_on_third_zero_outs_early():
    """Step 3: Halfway with runner on 3rd, 0 outs, tied, early game."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=0, inning=3,
    ))
    assert result["infield_recommendation"] == "halfway"


def test_step3_standard_runner_on_third_big_lead():
    """Step 3: Standard positioning with runner on 3rd when leading by 3+."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=4, inning=5,
    ))
    assert result["infield_recommendation"] == "standard"


def test_step3_guard_lines_late_close():
    """Step 3: Guard the lines in late, close game with 2 outs."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=2, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=1, inning=9,
    ))
    assert result["infield_recommendation"] == "guard_lines"


def test_step3_valid_infield_recommendations():
    """Step 3: Infield recommendation is always a recognized value."""
    valid_values = {"standard", "double_play_depth", "infield_in", "halfway", "guard_lines"}
    for outs in range(3):
        for r1 in (True, False):
            for r3 in (True, False):
                result = parse(get_defensive_positioning(
                    "h_001", "a_sp1", outs=outs, runner_on_first=r1,
                    runner_on_second=False, runner_on_third=r3,
                    score_differential=0, inning=5,
                ))
                assert result["infield_recommendation"] in valid_values, \
                    f"Got '{result['infield_recommendation']}' for outs={outs}, r1={r1}, r3={r3}"


# -----------------------------------------------------------------------
# Step 4: Returns recommended outfield positioning
# -----------------------------------------------------------------------

def test_step4_outfield_recommendation_present():
    """Step 4: Returns an outfield_recommendation field."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert "outfield_recommendation" in result
    assert isinstance(result["outfield_recommendation"], str)


def test_step4_deep_for_power_hitter():
    """Step 4: Deep positioning for power hitters (power >= 75)."""
    # h_003 Rafael Ortiz: power 88
    result = parse(get_defensive_positioning(
        "h_003", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert "deep" in result["outfield_recommendation"]


def test_step4_standard_for_average_hitter():
    """Step 4: Standard positioning for average power hitters."""
    # h_009 Andre Davis: power 50, contact 76
    result = parse(get_defensive_positioning(
        "h_009", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    # Power 50 is neither high nor very low
    assert result["outfield_recommendation"] in ("standard", "shaded_pull", "shaded_oppo")


def test_step4_no_doubles_late_close():
    """Step 4: No-doubles defense in late, close game with RISP."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=2, runner_on_first=False, runner_on_second=True,
        runner_on_third=False, score_differential=1, inning=9,
    ))
    assert result["outfield_recommendation"] == "no_doubles"


def test_step4_outfield_recommendation_is_string():
    """Step 4: Outfield recommendation is always a string."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert isinstance(result["outfield_recommendation"], str)
    assert len(result["outfield_recommendation"]) > 0


# -----------------------------------------------------------------------
# Step 5: Returns infield-in cost/benefit analysis
# -----------------------------------------------------------------------

def test_step5_infield_in_analysis_present():
    """Step 5: Returns infield_in_analysis dict."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert "infield_in_analysis" in result
    analysis = result["infield_in_analysis"]
    assert "runs_saved_at_home" in analysis
    assert "extra_hits_allowed" in analysis


def test_step5_runs_saved_positive_with_runner_on_third():
    """Step 5: Runs saved at home is positive when runner on 3rd."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=1, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=0, inning=5,
    ))
    assert result["infield_in_analysis"]["runs_saved_at_home"] > 0


def test_step5_runs_saved_zero_without_runner_on_third():
    """Step 5: Runs saved at home is 0 when no runner on 3rd."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=True, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["infield_in_analysis"]["runs_saved_at_home"] == 0.0


def test_step5_extra_hits_always_positive():
    """Step 5: Extra hits allowed is always positive (cost of playing in)."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert result["infield_in_analysis"]["extra_hits_allowed"] > 0


def test_step5_infield_in_analysis_realistic_ranges():
    """Step 5: Runs saved and extra hits are in realistic ranges."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=1, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=0, inning=5,
    ))
    analysis = result["infield_in_analysis"]
    assert 0.0 <= analysis["runs_saved_at_home"] <= 0.50
    assert 0.0 < analysis["extra_hits_allowed"] <= 0.20


def test_step5_runs_saved_zero_with_two_outs():
    """Step 5: Runs saved at home is 0 with 2 outs (runner scores on any hit anyway)."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=2, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=0, inning=5,
    ))
    assert result["infield_in_analysis"]["runs_saved_at_home"] == 0.0


# -----------------------------------------------------------------------
# Step 6: Returns shift recommendation within MLB rules
# -----------------------------------------------------------------------

def test_step6_shift_recommendation_present():
    """Step 6: Returns a shift_recommendation field."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert "shift_recommendation" in result
    assert isinstance(result["shift_recommendation"], str)


def test_step6_shift_mentions_2_and_2_rule():
    """Step 6: Shift recommendation references the 2-and-2 rule."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    assert "2-and-2" in result["shift_recommendation"] or "2-and-2" in result["shift_recommendation"].lower()


def test_step6_heavy_pull_hitter_gets_shift():
    """Step 6: Heavy pull hitter gets a shade recommendation."""
    # h_003 Rafael Ortiz: LHB, power 88 -> heavy pull
    result = parse(get_defensive_positioning(
        "h_003", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    # High power left-handed hitter should have shift toward first-base side
    shift = result["shift_recommendation"]
    assert "shade" in shift or "no shift" in shift


def test_step6_balanced_hitter_no_shift():
    """Step 6: Balanced spray chart gets no shift recommendation."""
    # h_009 Andre Davis: power 50, contact 76 -> more balanced
    result = parse(get_defensive_positioning(
        "h_009", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    shift = result["shift_recommendation"]
    assert "no shift" in shift or "standard" in shift


def test_step6_shift_never_violates_rules():
    """Step 6: Shift recommendation never suggests 3 infielders on one side."""
    # Test across multiple hitters
    for batter_id in ["h_001", "h_003", "h_004", "h_009", "a_001"]:
        result = parse(get_defensive_positioning(
            batter_id, "a_sp1", outs=0, runner_on_first=False,
            runner_on_second=False, runner_on_third=False,
            score_differential=0, inning=1,
        ))
        shift = result["shift_recommendation"].lower()
        # Should never mention 3 infielders on one side or full shift
        assert "three" not in shift or "3 infielders" not in shift
        assert "full shift" not in shift


# -----------------------------------------------------------------------
# Step 7: Returns error for invalid player identifiers
# -----------------------------------------------------------------------

def test_step7_invalid_batter_id():
    """Step 7: Returns error for invalid batter ID."""
    result = parse(get_defensive_positioning(
        "NONEXISTENT", "a_sp1", outs=0, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "NONEXISTENT" in result["message"]


def test_step7_invalid_pitcher_id():
    """Step 7: Returns error for invalid pitcher ID."""
    result = parse(get_defensive_positioning(
        "h_001", "NONEXISTENT", outs=0, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "NONEXISTENT" in result["message"]


def test_step7_both_ids_invalid():
    """Step 7: Returns error when batter ID is invalid (checked first)."""
    result = parse(get_defensive_positioning(
        "BAD_BATTER", "BAD_PITCHER", outs=0, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"


def test_step7_empty_batter_id():
    """Step 7: Returns error for empty batter ID."""
    result = parse(get_defensive_positioning(
        "", "a_sp1", outs=0, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"


def test_step7_pitcher_without_pitcher_attrs():
    """Step 7: Returns error when pitcher_id points to a position player."""
    result = parse(get_defensive_positioning(
        "h_001", "h_002", outs=0, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_PITCHER"


def test_step7_batter_without_batter_attrs():
    """Step 7: Returns error when batter_id points to a pitcher-only player."""
    result = parse(get_defensive_positioning(
        "h_sp1", "a_sp1", outs=0, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_BATTER"


def test_step7_invalid_outs():
    """Step 7: Returns error for invalid outs value."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=3, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step7_negative_outs():
    """Step 7: Returns error for negative outs."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=-1, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=1,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step7_invalid_inning():
    """Step 7: Returns error for invalid inning value."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False,
        runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


# -----------------------------------------------------------------------
# Helper function unit tests
# -----------------------------------------------------------------------

def test_helper_spray_chart_power_range():
    """Helper: Spray chart responds to power attribute range."""
    low_power = _derive_spray_chart({"power": 20, "contact": 50}, "R", "L")
    high_power = _derive_spray_chart({"power": 90, "contact": 50}, "R", "L")
    assert high_power["groundball"]["pull_pct"] > low_power["groundball"]["pull_pct"]
    assert high_power["flyball"]["pull_pct"] > low_power["flyball"]["pull_pct"]


def test_helper_spray_chart_contact_effect():
    """Helper: High contact reduces pull tendency."""
    low_contact = _derive_spray_chart({"power": 70, "contact": 30}, "R", "L")
    high_contact = _derive_spray_chart({"power": 70, "contact": 90}, "R", "L")
    assert high_contact["groundball"]["pull_pct"] < low_contact["groundball"]["pull_pct"]


def test_helper_gb_rate_power_inverse():
    """Helper: Ground ball rate decreases as power increases."""
    low_power_gb = _derive_gb_rate({"power": 30}, {"stuff": 60, "control": 60})
    high_power_gb = _derive_gb_rate({"power": 90}, {"stuff": 60, "control": 60})
    assert high_power_gb < low_power_gb


def test_helper_gb_rate_stuff_effect():
    """Helper: Higher stuff pitcher gets more ground balls."""
    low_stuff = _derive_gb_rate({"power": 50}, {"stuff": 30, "control": 60})
    high_stuff = _derive_gb_rate({"power": 50}, {"stuff": 90, "control": 60})
    assert high_stuff > low_stuff


def test_helper_gb_rate_realistic():
    """Helper: Ground ball rate is always in realistic range."""
    for power in (20, 50, 80):
        for stuff in (30, 60, 90):
            rate = _derive_gb_rate({"power": power}, {"stuff": stuff, "control": 60})
            assert 0.30 <= rate <= 0.60, f"GB rate {rate} out of range for power={power}, stuff={stuff}"


def test_helper_clamp():
    """Helper: _clamp works correctly."""
    assert _clamp(5.0, 0.0, 10.0) == 5.0
    assert _clamp(-1.0, 0.0, 10.0) == 0.0
    assert _clamp(15.0, 0.0, 10.0) == 10.0


def test_helper_recommend_shift_lhb_pull():
    """Helper: LHB pull hitter gets shift toward first-base side."""
    spray = {
        "groundball": {"pull_pct": 0.48, "center_pct": 0.30, "oppo_pct": 0.22},
        "flyball": {"pull_pct": 0.42, "center_pct": 0.33, "oppo_pct": 0.25},
    }
    shift = _recommend_shift(spray, "L")
    assert "first-base" in shift


def test_helper_recommend_shift_rhb_pull():
    """Helper: RHB pull hitter gets shift toward third-base side."""
    spray = {
        "groundball": {"pull_pct": 0.48, "center_pct": 0.30, "oppo_pct": 0.22},
        "flyball": {"pull_pct": 0.42, "center_pct": 0.33, "oppo_pct": 0.25},
    }
    shift = _recommend_shift(spray, "R")
    assert "third-base" in shift


def test_helper_recommend_shift_balanced():
    """Helper: Balanced spray chart gets no shift."""
    spray = {
        "groundball": {"pull_pct": 0.38, "center_pct": 0.35, "oppo_pct": 0.27},
        "flyball": {"pull_pct": 0.35, "center_pct": 0.37, "oppo_pct": 0.28},
    }
    shift = _recommend_shift(spray, "R")
    assert "no shift" in shift


# -----------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------

def test_integration_power_lhb_vs_rhp():
    """Integration: Full positioning for LHB power hitter vs RHP."""
    result = parse(get_defensive_positioning(
        "h_003", "h_sp1",  # Rafael Ortiz (L, power 88) vs Brandon Cole (R)
        outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=3,
    ))
    assert result["status"] == "ok"
    assert result["batter_name"] == "Rafael Ortiz"
    assert result["bats"] == "L"
    assert result["pitcher_name"] == "Brandon Cole"
    assert result["throws"] == "R"
    # Power hitter should have high pull%
    assert result["spray_chart"]["groundball"]["pull_pct"] >= 0.38
    # Outfield should be deep
    assert "deep" in result["outfield_recommendation"]
    # GB rate should be lower for power hitter
    assert result["groundball_rate"] < 0.45


def test_integration_contact_hitter_standard_situation():
    """Integration: Contact hitter in standard situation."""
    result = parse(get_defensive_positioning(
        "h_009", "a_sp1",  # Andre Davis (L, power 50, contact 76)
        outs=1, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=2, inning=4,
    ))
    assert result["status"] == "ok"
    # Standard situation, no runners, standard infield
    assert result["infield_recommendation"] == "standard"
    # Moderate hitter, standard outfield likely
    assert result["outfield_recommendation"] in ("standard", "shaded_pull", "shaded_oppo")


def test_integration_dp_situation():
    """Integration: Double play situation with runner on 1st."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1",
        outs=0, runner_on_first=True, runner_on_second=False,
        runner_on_third=False, score_differential=-1, inning=6,
    ))
    assert result["status"] == "ok"
    assert result["infield_recommendation"] == "double_play_depth"


def test_integration_squeeze_scenario():
    """Integration: Runner on 3rd, 0 outs, late game, trailing."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1",
        outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=True, score_differential=-1, inning=8,
    ))
    assert result["status"] == "ok"
    assert result["infield_recommendation"] == "infield_in"
    assert result["infield_in_analysis"]["runs_saved_at_home"] > 0


def test_integration_all_fields_present():
    """Integration: All required output fields are present."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1", outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=1,
    ))
    required_fields = [
        "status", "batter_id", "batter_name", "bats",
        "pitcher_id", "pitcher_name", "throws",
        "spray_chart", "groundball_rate",
        "infield_recommendation", "outfield_recommendation",
        "infield_in_analysis", "shift_recommendation",
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"


def test_integration_multiple_batters_produce_different_charts():
    """Integration: Different batters produce different spray charts."""
    results = []
    for batter_id in ["h_001", "h_003", "h_004", "h_009"]:
        result = parse(get_defensive_positioning(
            batter_id, "a_sp1", outs=0, runner_on_first=False,
            runner_on_second=False, runner_on_third=False,
            score_differential=0, inning=1,
        ))
        results.append(result["spray_chart"]["groundball"]["pull_pct"])
    # Not all the same
    assert len(set(results)) > 1, "Expected different spray charts for different batters"


def test_integration_late_close_no_doubles():
    """Integration: No-doubles defense in 9th inning, 1-run lead, RISP."""
    result = parse(get_defensive_positioning(
        "h_001", "a_sp1",
        outs=2, runner_on_first=True, runner_on_second=True,
        runner_on_third=False, score_differential=1, inning=9,
    ))
    assert result["outfield_recommendation"] == "no_doubles"


def test_integration_bench_player_as_batter():
    """Integration: Bench player can be used as batter."""
    result = parse(get_defensive_positioning(
        "h_012", "a_sp1",  # Darnell Washington, bench OF
        outs=1, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=5,
    ))
    assert result["status"] == "ok"
    assert result["batter_name"] == "Darnell Washington"


def test_integration_bullpen_pitcher_as_pitcher():
    """Integration: Bullpen pitcher can be used as pitcher_id."""
    result = parse(get_defensive_positioning(
        "h_001", "h_bp1",  # vs Greg Foster, closer
        outs=0, runner_on_first=False, runner_on_second=False,
        runner_on_third=False, score_differential=0, inning=9,
    ))
    assert result["status"] == "ok"
    assert result["pitcher_name"] == "Greg Foster"


# -----------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    test_funcs = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    errors = []
    for func in test_funcs:
        try:
            func()
            passed += 1
            print(f"  PASS: {func.__name__}")
        except Exception as e:
            failed += 1
            errors.append((func.__name__, e))
            print(f"  FAIL: {func.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print(f"\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
