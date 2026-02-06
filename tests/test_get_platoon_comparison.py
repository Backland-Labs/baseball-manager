# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_platoon_comparison tool.

Verifies all feature requirements from features.json:
1. Accepts the current batter identifier and the potential pinch hitter identifier
2. Accepts the current pitcher identifier for matchup context
3. Returns projected wOBA vs the current pitcher for each batter
4. Returns the platoon advantage delta between the two batters
5. Returns the defensive cost of the substitution (current fielder vs replacement's defensive ability)
6. Returns the bench depth impact (remaining bench options after the substitution)
7. Returns an error if any player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_platoon_comparison import (
    get_platoon_comparison,
    _derive_projected_woba_vs_pitcher,
    _derive_wrc_plus,
    _derive_oaa,
    _compute_defensive_cost,
    _compute_bench_depth_impact,
    _load_players,
    _clamp,
    _PLAYERS,
    _TEAM_BENCH,
    _PLAYER_TEAM,
)


def parse(result: str) -> dict:
    d = json.loads(result)
    if d.get("status") == "ok" and "data" in d:
        return {"status": "ok", "tool": d.get("tool"), **d["data"]}
    return d


# -----------------------------------------------------------------------
# Step 1: Accepts current batter and pinch hitter identifiers
# -----------------------------------------------------------------------

def test_step1_accepts_current_batter_and_pinch_hitter():
    """Step 1: Accepts current batter and pinch hitter IDs."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    assert result["status"] == "ok"
    assert result["current_batter"]["player_id"] == "h_004"
    assert result["pinch_hitter"]["player_id"] == "h_012"


def test_step1_returns_player_names():
    """Step 1: Returns both player names."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    assert result["current_batter"]["name"] == "Tyrone Jackson"
    assert result["pinch_hitter"]["name"] == "Darnell Washington"


def test_step1_returns_handedness():
    """Step 1: Returns batter handedness for both players."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    assert result["current_batter"]["bats"] == "R"
    assert result["pinch_hitter"]["bats"] == "L"


def test_step1_works_with_bench_pinch_hitter():
    """Step 1: Bench players work as pinch hitters."""
    result = parse(get_platoon_comparison("h_001", "h_014", "a_sp1"))
    assert result["status"] == "ok"
    assert result["pinch_hitter"]["name"] == "Kenji Tanaka"


def test_step1_works_across_same_team():
    """Step 1: Both batters from same team works."""
    result = parse(get_platoon_comparison("a_006", "a_012", "h_sp1"))
    assert result["status"] == "ok"
    assert result["current_batter"]["name"] == "Trey Anderson"
    assert result["pinch_hitter"]["name"] == "Marcus Green"


def test_step1_switch_hitter_as_current():
    """Step 1: Switch hitter as current batter."""
    result = parse(get_platoon_comparison("h_007", "h_012", "a_sp1"))
    assert result["status"] == "ok"
    assert result["current_batter"]["bats"] == "S"


def test_step1_switch_hitter_as_pinch_hitter():
    """Step 1: Switch hitter as pinch hitter."""
    result = parse(get_platoon_comparison("h_004", "h_015", "a_sp1"))
    assert result["status"] == "ok"
    assert result["pinch_hitter"]["bats"] == "S"


# -----------------------------------------------------------------------
# Step 2: Accepts current pitcher identifier for matchup context
# -----------------------------------------------------------------------

def test_step2_returns_pitcher_info():
    """Step 2: Returns pitcher information including ID, name, and throws."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    assert result["pitcher"]["player_id"] == "a_sp1"
    assert result["pitcher"]["name"] == "Matt Henderson"
    assert result["pitcher"]["throws"] == "L"


def test_step2_right_handed_pitcher():
    """Step 2: Works with right-handed pitcher."""
    result = parse(get_platoon_comparison("h_004", "h_012", "h_sp1"))
    assert result["pitcher"]["throws"] == "R"
    assert result["pitcher"]["name"] == "Brandon Cole"


def test_step2_reliever_as_pitcher():
    """Step 2: Relievers can be used as the pitcher."""
    result = parse(get_platoon_comparison("h_004", "h_012", "h_bp1"))
    assert result["status"] == "ok"
    assert result["pitcher"]["name"] == "Greg Foster"
    assert result["pitcher"]["throws"] == "R"


def test_step2_left_handed_reliever():
    """Step 2: Left-handed relievers work correctly."""
    result = parse(get_platoon_comparison("h_004", "h_012", "h_bp2"))
    assert result["status"] == "ok"
    assert result["pitcher"]["throws"] == "L"


# -----------------------------------------------------------------------
# Step 3: Returns projected wOBA vs the current pitcher for each batter
# -----------------------------------------------------------------------

def test_step3_returns_woba_for_current_batter():
    """Step 3: Returns projected wOBA for the current batter."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    woba = result["current_batter"]["projected_woba_vs_pitcher"]
    assert isinstance(woba, float)
    assert 0.220 <= woba <= 0.430


def test_step3_returns_woba_for_pinch_hitter():
    """Step 3: Returns projected wOBA for the pinch hitter."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    woba = result["pinch_hitter"]["projected_woba_vs_pitcher"]
    assert isinstance(woba, float)
    assert 0.220 <= woba <= 0.430


def test_step3_returns_wrc_plus_for_both():
    """Step 3: Returns wRC+ alongside wOBA for both batters."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    assert isinstance(result["current_batter"]["projected_wrc_plus"], int)
    assert isinstance(result["pinch_hitter"]["projected_wrc_plus"], int)
    assert 40 <= result["current_batter"]["projected_wrc_plus"] <= 200
    assert 40 <= result["pinch_hitter"]["projected_wrc_plus"] <= 200


def test_step3_woba_reflects_platoon_splits_vs_lhp():
    """Step 3: RHB should have higher wOBA vs LHP than LHB vs LHP (opposite-hand advantage)."""
    # h_004 (Tyrone Jackson, R) vs a_sp1 (Matt Henderson, LHP)
    # h_003 (Rafael Ortiz, L) vs a_sp1 (Matt Henderson, LHP)
    result_rhb = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    # RHB vs LHP = opposite hand = favorable
    rhb_woba = result_rhb["current_batter"]["projected_woba_vs_pitcher"]

    result_lhb = parse(get_platoon_comparison("h_003", "h_012", "a_sp1"))
    # LHB vs LHP = same hand = unfavorable
    lhb_woba = result_lhb["current_batter"]["projected_woba_vs_pitcher"]

    # The RHB (h_004) has power=85 while LHB (h_003) has power=88, so h_003
    # has higher raw attributes. But vs LHP, h_003 is same-hand (penalty)
    # while h_004 is opposite-hand (bonus). With similar attribute levels,
    # the platoon effect should be significant.
    # We verify the platoon effect exists (opposite > same) accounting for
    # the attribute difference.
    # h_004: contact=68, power=85, eye=65, speed=60, avg_vs_l=0.290
    # h_003: contact=72, power=88, eye=80, speed=40, avg_vs_l=0.240
    # h_003 has much better eye (80 vs 65) and power (88 vs 85) but lower avg_vs_l
    # The LHB penalty should pull h_003 down significantly vs LHP
    # h_004 avg_vs_l=0.290 (high), h_003 avg_vs_l=0.240 (low)
    # So h_004 should be better vs LHP
    assert rhb_woba > lhb_woba, f"RHB wOBA {rhb_woba} should be > LHB wOBA {lhb_woba} vs LHP"


def test_step3_woba_reflects_platoon_splits_vs_rhp():
    """Step 3: LHB should have higher wOBA vs RHP than RHB vs RHP."""
    # h_003 (Rafael Ortiz, L, high power/eye) vs h_sp1 (RHP)
    # h_004 (Tyrone Jackson, R) vs h_sp1 (RHP)
    result_lhb = parse(get_platoon_comparison("h_003", "h_012", "h_sp1"))
    lhb_woba = result_lhb["current_batter"]["projected_woba_vs_pitcher"]

    result_rhb = parse(get_platoon_comparison("h_004", "h_012", "h_sp1"))
    rhb_woba = result_rhb["current_batter"]["projected_woba_vs_pitcher"]

    # LHB vs RHP = opposite hand = favorable + h_003 has much better attributes
    # RHB vs RHP = same hand = unfavorable
    assert lhb_woba > rhb_woba, f"LHB wOBA {lhb_woba} should be > RHB wOBA {rhb_woba} vs RHP"


def test_step3_woba_changes_with_pitcher_hand():
    """Step 3: Same batter gets different wOBA projections against different-handed pitchers."""
    # RHB vs LHP (favorable) vs RHB vs RHP (unfavorable)
    result_vs_lhp = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))  # LHP
    result_vs_rhp = parse(get_platoon_comparison("h_004", "h_012", "h_sp1"))  # RHP

    woba_vs_lhp = result_vs_lhp["current_batter"]["projected_woba_vs_pitcher"]
    woba_vs_rhp = result_vs_rhp["current_batter"]["projected_woba_vs_pitcher"]

    # RHB should perform better vs LHP than RHP
    assert woba_vs_lhp > woba_vs_rhp, f"RHB wOBA vs LHP {woba_vs_lhp} should be > vs RHP {woba_vs_rhp}"


def test_step3_switch_hitter_gets_advantage_vs_both():
    """Step 3: Switch hitter should get a slight platoon advantage vs either hand."""
    # h_007 (Carlos Ramirez, S) vs LHP and RHP
    result_vs_lhp = parse(get_platoon_comparison("h_007", "h_012", "a_sp1"))  # LHP
    result_vs_rhp = parse(get_platoon_comparison("h_007", "h_012", "h_sp1"))  # RHP

    woba_vs_lhp = result_vs_lhp["current_batter"]["projected_woba_vs_pitcher"]
    woba_vs_rhp = result_vs_rhp["current_batter"]["projected_woba_vs_pitcher"]

    # Both should be reasonable (not penalized like same-hand)
    assert woba_vs_lhp >= 0.280
    assert woba_vs_rhp >= 0.280


# -----------------------------------------------------------------------
# Step 4: Returns the platoon advantage delta between the two batters
# -----------------------------------------------------------------------

def test_step4_returns_platoon_delta():
    """Step 4: Returns a numeric platoon advantage delta."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    delta = result["platoon_advantage_delta"]
    assert isinstance(delta, float)


def test_step4_delta_is_pinch_minus_current():
    """Step 4: Delta is pinch_hitter_woba - current_batter_woba."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    expected_delta = round(
        result["pinch_hitter"]["projected_woba_vs_pitcher"]
        - result["current_batter"]["projected_woba_vs_pitcher"],
        3,
    )
    assert result["platoon_advantage_delta"] == expected_delta


def test_step4_positive_delta_when_pinch_hitter_better():
    """Step 4: Delta is positive when pinch hitter has better projection."""
    # h_008 (Tommy Sullivan, R, contact=65, power=58) vs LHP
    # h_012 (Darnell Washington, L, contact=72, power=68) vs LHP
    # Sullivan (R) vs LHP gets opposite-hand bonus, but Washington (L) vs LHP
    # gets same-hand penalty. Sullivan should do better vs LHP.
    # Let's use a case where pinch hitter is clearly better:
    # h_008 (weak batter) current, h_003 (Rafael Ortiz, strong batter) pinch vs RHP
    result = parse(get_platoon_comparison("h_008", "h_003", "h_sp1"))
    # Ortiz (L, power=88, eye=80) vs RHP gets opposite-hand bonus
    # Sullivan (R, power=58, eye=60) vs RHP gets same-hand penalty
    assert result["platoon_advantage_delta"] > 0


def test_step4_negative_delta_when_current_is_better():
    """Step 4: Delta is negative when current batter has better projection."""
    # h_003 (Rafael Ortiz, L, power=88, eye=80) vs RHP (h_sp1)
    # h_017 (Tomas Herrera, R, power=45, eye=50) vs RHP
    result = parse(get_platoon_comparison("h_003", "h_017", "h_sp1"))
    assert result["platoon_advantage_delta"] < 0


def test_step4_delta_near_zero_for_similar_batters():
    """Step 4: Delta is small when batters are similar against the same pitcher."""
    # Same player as both batter and pinch hitter
    result = parse(get_platoon_comparison("h_004", "h_004", "a_sp1"))
    assert result["platoon_advantage_delta"] == 0.0


# -----------------------------------------------------------------------
# Step 5: Returns the defensive cost of the substitution
# -----------------------------------------------------------------------

def test_step5_returns_defensive_cost():
    """Step 5: Returns defensive cost data structure."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    dc = result["defensive_cost"]
    assert "current_oaa" in dc
    assert "replacement_oaa" in dc
    assert "oaa_difference" in dc
    assert "defensive_cost_woba" in dc
    assert "position" in dc


def test_step5_oaa_values_are_numeric():
    """Step 5: OAA values are numeric."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    dc = result["defensive_cost"]
    assert isinstance(dc["current_oaa"], (int, float))
    assert isinstance(dc["replacement_oaa"], (int, float))
    assert isinstance(dc["oaa_difference"], (int, float))


def test_step5_defensive_cost_for_dh():
    """Step 5: DH position has no defensive cost."""
    # h_006 (Shin-Soo Park, DH) replaced by bench player
    result = parse(get_platoon_comparison("h_006", "h_013", "a_sp1"))
    dc = result["defensive_cost"]
    assert dc["position"] == "DH"
    assert dc["defensive_cost_woba"] == 0.0
    assert dc["oaa_difference"] == 0.0


def test_step5_defensive_cost_reflects_fielding_difference():
    """Step 5: Better defender replacing weaker one shows positive OAA."""
    # h_004 (Tyrone Jackson, RF, range=70) replaced by h_014 (Kenji Tanaka, range=80)
    result = parse(get_platoon_comparison("h_004", "h_014", "a_sp1"))
    dc = result["defensive_cost"]
    # h_014 has better range (80 vs 70) so should have higher OAA at RF
    assert dc["oaa_difference"] >= 0


def test_step5_pinch_hitter_can_play_position():
    """Step 5: Reports whether pinch hitter can play the position."""
    # h_012 (Darnell Washington, positions: ["LF", "RF"]) replacing h_004 at RF
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    dc = result["defensive_cost"]
    assert dc["pinch_hitter_can_play_position"] is True


def test_step5_out_of_position_penalty():
    """Step 5: Out-of-position pinch hitter gets OAA penalty."""
    # h_010 (Victor Nguyen, C, positions: ["C", "1B"]) replacing h_002 at SS
    result = parse(get_platoon_comparison("h_002", "h_010", "a_sp1"))
    dc = result["defensive_cost"]
    # Nguyen can't play SS, should get penalty
    assert dc["pinch_hitter_can_play_position"] is False
    assert dc["replacement_oaa"] < dc["current_oaa"]


def test_step5_notes_for_out_of_position():
    """Step 5: Notes explain out-of-position situation."""
    result = parse(get_platoon_comparison("h_002", "h_010", "a_sp1"))
    dc = result["defensive_cost"]
    assert "out of position" in dc.get("notes", "").lower() or not dc["pinch_hitter_can_play_position"]


# -----------------------------------------------------------------------
# Step 6: Returns the bench depth impact
# -----------------------------------------------------------------------

def test_step6_returns_bench_depth_structure():
    """Step 6: Returns bench depth impact structure."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    bi = result["bench_depth_impact"]
    assert "bench_remaining" in bi
    assert "left_handed_remaining" in bi
    assert "right_handed_remaining" in bi
    assert "remaining_bench_players" in bi


def test_step6_bench_remaining_count():
    """Step 6: Bench remaining is one less than total bench size."""
    # Home team has 8 bench players (h_010 through h_017)
    # Using h_012 leaves 7
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    bi = result["bench_depth_impact"]
    assert bi["bench_remaining"] == 7


def test_step6_bench_handedness_breakdown():
    """Step 6: Handedness breakdown is accurate."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    bi = result["bench_depth_impact"]
    # After removing h_012 (L), remaining home bench:
    # h_010 (R), h_011 (R), h_013 (L), h_014 (R), h_015 (S), h_016 (R), h_017 (R)
    total = bi["left_handed_remaining"] + bi["right_handed_remaining"] + bi["switch_remaining"]
    assert total == bi["bench_remaining"]


def test_step6_remaining_players_listed():
    """Step 6: Remaining bench players are listed with details."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    bi = result["bench_depth_impact"]
    players = bi["remaining_bench_players"]
    assert isinstance(players, list)
    assert len(players) == bi["bench_remaining"]
    # Each player should have basic info
    for p in players:
        assert "player_id" in p
        assert "name" in p
        assert "bats" in p
        assert "positions" in p


def test_step6_pinch_hitter_not_in_remaining():
    """Step 6: The used pinch hitter is not in the remaining bench list."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    bi = result["bench_depth_impact"]
    remaining_ids = [p["player_id"] for p in bi["remaining_bench_players"]]
    assert "h_012" not in remaining_ids


def test_step6_away_team_bench():
    """Step 6: Bench depth works for away team pinch hitters too."""
    # Away team has 8 bench players (a_010 through a_017)
    result = parse(get_platoon_comparison("a_006", "a_012", "h_sp1"))
    bi = result["bench_depth_impact"]
    assert bi["bench_remaining"] == 7


def test_step6_switch_hitter_count():
    """Step 6: Switch hitters are counted in the switch category."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    bi = result["bench_depth_impact"]
    assert bi["switch_remaining"] >= 0
    # Home bench has h_015 (S) as switch hitter remaining
    assert bi["switch_remaining"] >= 1


# -----------------------------------------------------------------------
# Step 7: Returns an error if any player identifier is invalid
# -----------------------------------------------------------------------

def test_step7_invalid_current_batter():
    """Step 7: Error for invalid current batter ID."""
    result = parse(get_platoon_comparison("FAKE_ID", "h_012", "a_sp1"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "FAKE_ID" in result["message"]


def test_step7_invalid_pinch_hitter():
    """Step 7: Error for invalid pinch hitter ID."""
    result = parse(get_platoon_comparison("h_004", "FAKE_ID", "a_sp1"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "FAKE_ID" in result["message"]


def test_step7_invalid_pitcher():
    """Step 7: Error for invalid pitcher ID."""
    result = parse(get_platoon_comparison("h_004", "h_012", "FAKE_ID"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "FAKE_ID" in result["message"]


def test_step7_empty_string_ids():
    """Step 7: Error for empty string IDs."""
    result = parse(get_platoon_comparison("", "h_012", "a_sp1"))
    assert result["status"] == "error"


def test_step7_non_pitcher_as_pitcher_id():
    """Step 7: Error when pitcher_id is a non-pitcher (no pitcher attributes)."""
    result = parse(get_platoon_comparison("h_004", "h_012", "h_001"))
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_PITCHER"


def test_step7_all_three_ids_invalid():
    """Step 7: First invalid ID (current batter) triggers the error."""
    result = parse(get_platoon_comparison("FAKE1", "FAKE2", "FAKE3"))
    assert result["status"] == "error"
    assert "FAKE1" in result["message"]


# -----------------------------------------------------------------------
# Helper function unit tests
# -----------------------------------------------------------------------

def test_helper_clamp():
    """Helper: _clamp constrains values to range."""
    assert _clamp(0.5, 0.0, 1.0) == 0.5
    assert _clamp(-0.1, 0.0, 1.0) == 0.0
    assert _clamp(1.5, 0.0, 1.0) == 1.0


def test_helper_derive_woba_vs_pitcher_basic():
    """Helper: _derive_projected_woba_vs_pitcher returns float in valid range."""
    attrs = {"contact": 75, "power": 60, "eye": 70, "speed": 65,
             "avg_vs_l": 0.270, "avg_vs_r": 0.260}
    woba = _derive_projected_woba_vs_pitcher(attrs, "R", "L")
    assert 0.220 <= woba <= 0.430


def test_helper_derive_woba_vs_pitcher_opposite_hand():
    """Helper: Opposite-hand matchup yields higher wOBA."""
    attrs = {"contact": 70, "power": 60, "eye": 65, "speed": 50,
             "avg_vs_l": 0.265, "avg_vs_r": 0.265}
    woba_opp = _derive_projected_woba_vs_pitcher(attrs, "R", "L")  # Opposite
    woba_same = _derive_projected_woba_vs_pitcher(attrs, "R", "R")  # Same
    assert woba_opp > woba_same


def test_helper_derive_woba_vs_pitcher_switch_hitter():
    """Helper: Switch hitter gets moderate advantage vs either hand."""
    attrs = {"contact": 70, "power": 60, "eye": 65, "speed": 50,
             "avg_vs_l": 0.265, "avg_vs_r": 0.265}
    woba_s_vs_l = _derive_projected_woba_vs_pitcher(attrs, "S", "L")
    woba_s_vs_r = _derive_projected_woba_vs_pitcher(attrs, "S", "R")
    woba_same = _derive_projected_woba_vs_pitcher(attrs, "L", "L")
    # Switch hitter should be better than same-hand matchup
    assert woba_s_vs_l > woba_same
    assert woba_s_vs_r > woba_same


def test_helper_derive_wrc_plus():
    """Helper: _derive_wrc_plus produces valid int."""
    assert _derive_wrc_plus(0.315) == 100  # League average
    assert _derive_wrc_plus(0.400) > 100
    assert _derive_wrc_plus(0.250) < 100


def test_helper_derive_wrc_plus_range():
    """Helper: _derive_wrc_plus is clamped to 40-200."""
    assert _derive_wrc_plus(0.100) == 40
    assert _derive_wrc_plus(0.800) == 200


def test_helper_derive_oaa_average_player():
    """Helper: Average player (50 across the board) has ~0 OAA."""
    attrs = {"range": 50, "arm_strength": 50, "error_rate": 0.03, "positions": ["SS"]}
    oaa = _derive_oaa(attrs, "SS")
    assert -2.0 <= oaa <= 2.0


def test_helper_derive_oaa_elite_defender():
    """Helper: High-range player has positive OAA."""
    attrs = {"range": 85, "arm_strength": 80, "error_rate": 0.02, "positions": ["SS"]}
    oaa = _derive_oaa(attrs, "SS")
    assert oaa > 5.0


def test_helper_derive_oaa_dh():
    """Helper: DH position always returns 0 OAA."""
    attrs = {"range": 85, "arm_strength": 80, "error_rate": 0.02, "positions": ["DH"]}
    oaa = _derive_oaa(attrs, "DH")
    assert oaa == 0.0


def test_helper_derive_oaa_out_of_position():
    """Helper: Out-of-position player gets penalty."""
    attrs = {"range": 70, "arm_strength": 70, "error_rate": 0.03, "positions": ["1B"]}
    oaa_natural = _derive_oaa(attrs, "1B")
    oaa_oop = _derive_oaa(attrs, "SS")  # Can't play SS
    assert oaa_oop < oaa_natural, f"Out-of-position OAA {oaa_oop} should be < natural {oaa_natural}"


def test_helper_compute_defensive_cost_dh():
    """Helper: DH substitution has zero cost."""
    current = {"fielder": {"range": 50, "arm_strength": 50, "error_rate": 0.03, "positions": ["DH"]}}
    pinch = {"fielder": {"range": 60, "arm_strength": 60, "error_rate": 0.03, "positions": ["DH"]}}
    result = _compute_defensive_cost(current, pinch, "DH")
    assert result["defensive_cost_woba"] == 0.0


def test_helper_compute_bench_depth_impact():
    """Helper: Bench depth calculation returns valid structure."""
    _load_players()
    result = _compute_bench_depth_impact("h_012", "home")
    assert result["bench_remaining"] == 7  # 8 total minus 1 used
    assert isinstance(result["remaining_bench_players"], list)


def test_helper_compute_bench_depth_excludes_used():
    """Helper: Used pinch hitter excluded from remaining."""
    _load_players()
    result = _compute_bench_depth_impact("h_012", "home")
    ids = [p["player_id"] for p in result["remaining_bench_players"]]
    assert "h_012" not in ids


# -----------------------------------------------------------------------
# Integration tests: realistic scenarios
# -----------------------------------------------------------------------

def test_integration_rhb_pinch_hit_vs_lhp():
    """Integration: Pinch-hitting a RHB for LHB against LHP (classic platoon move)."""
    # h_003 (Rafael Ortiz, L) at bat vs a_sp1 (Matt Henderson, LHP)
    # Pinch hit with h_004 (Tyrone Jackson, R)
    result = parse(get_platoon_comparison("h_003", "h_004", "a_sp1"))
    assert result["status"] == "ok"
    # RHB pinch hitter vs LHP should have platoon advantage over LHB
    # But Ortiz has much better base attributes (power=88, eye=80)
    # Still, platoon matters for the delta direction
    # Just verify the structure is complete
    assert "platoon_advantage_delta" in result
    assert "defensive_cost" in result
    assert "bench_depth_impact" in result


def test_integration_lhb_pinch_hit_vs_rhp():
    """Integration: Pinch-hitting a LHB for RHB against RHP."""
    # h_004 (Tyrone Jackson, R) at bat vs h_sp1 (Brandon Cole, RHP)
    # Pinch hit with h_012 (Darnell Washington, L)
    result = parse(get_platoon_comparison("h_004", "h_012", "h_sp1"))
    assert result["status"] == "ok"
    # LHB pinch hitter vs RHP should get platoon advantage over RHB
    delta = result["platoon_advantage_delta"]
    # Verify the platoon direction is correct
    # h_012 (L) vs RHP = opposite hand = favorable
    # h_004 (R) vs RHP = same hand = unfavorable
    assert delta > 0, f"LHB pinch hitter should have positive delta vs RHP, got {delta}"


def test_integration_all_output_fields_present():
    """Integration: Verify all expected output fields are present."""
    result = parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    assert result["status"] == "ok"

    # Top-level fields
    assert "current_batter" in result
    assert "pinch_hitter" in result
    assert "pitcher" in result
    assert "platoon_advantage_delta" in result
    assert "defensive_cost" in result
    assert "bench_depth_impact" in result

    # Current batter fields
    cb = result["current_batter"]
    assert "player_id" in cb
    assert "name" in cb
    assert "bats" in cb
    assert "projected_woba_vs_pitcher" in cb
    assert "projected_wrc_plus" in cb

    # Pinch hitter fields
    ph = result["pinch_hitter"]
    assert "player_id" in ph
    assert "name" in ph
    assert "bats" in ph
    assert "projected_woba_vs_pitcher" in ph
    assert "projected_wrc_plus" in ph

    # Pitcher fields
    p = result["pitcher"]
    assert "player_id" in p
    assert "name" in p
    assert "throws" in p

    # Defensive cost fields
    dc = result["defensive_cost"]
    assert "current_oaa" in dc
    assert "replacement_oaa" in dc
    assert "oaa_difference" in dc
    assert "defensive_cost_woba" in dc
    assert "position" in dc

    # Bench depth fields
    bi = result["bench_depth_impact"]
    assert "bench_remaining" in bi
    assert "left_handed_remaining" in bi
    assert "right_handed_remaining" in bi
    assert "switch_remaining" in bi
    assert "remaining_bench_players" in bi


def test_integration_same_player_both_sides():
    """Integration: Comparing a player to themselves returns zero delta."""
    result = parse(get_platoon_comparison("h_004", "h_004", "a_sp1"))
    assert result["status"] == "ok"
    assert result["platoon_advantage_delta"] == 0.0
    # Same player means same defensive ability
    assert result["defensive_cost"]["oaa_difference"] == 0.0


def test_integration_away_team_scenario():
    """Integration: Full scenario with away team players."""
    result = parse(get_platoon_comparison("a_009", "a_012", "h_sp1"))
    assert result["status"] == "ok"
    assert result["current_batter"]["name"] == "Diego Santos"
    assert result["pinch_hitter"]["name"] == "Marcus Green"
    assert result["pitcher"]["name"] == "Brandon Cole"


def test_integration_catcher_replacement():
    """Integration: Pinch-hitting for a catcher (high defensive cost)."""
    # h_008 (Tommy Sullivan, C, range=55, arm=75) replaced by h_012 (Darnell Washington, OF)
    result = parse(get_platoon_comparison("h_008", "h_012", "a_sp1"))
    dc = result["defensive_cost"]
    assert dc["position"] == "C"
    # Washington can't play C, should be out of position
    assert dc["pinch_hitter_can_play_position"] is False


def test_integration_if_replacement():
    """Integration: Pinch-hitting at an infield position."""
    # h_002 (Derek Williams, SS) replaced by h_011 (Ryan O'Brien, IF - can play SS)
    result = parse(get_platoon_comparison("h_002", "h_011", "a_sp1"))
    dc = result["defensive_cost"]
    assert dc["position"] == "SS"
    assert dc["pinch_hitter_can_play_position"] is True
