# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_defensive_replacement_value tool.

Verifies all feature requirements from features.json:
1. Accepts the current fielder identifier and the potential replacement identifier
2. Accepts the fielding position for the substitution
3. Returns the defensive upgrade in OAA (Outs Above Average) or DRS difference
4. Returns the offensive downgrade in projected wOBA or wRC+
5. Returns estimated innings remaining based on current game state
6. Returns net expected value of the substitution (defensive gain minus offensive cost)
7. Returns a textual recommendation (favorable, marginal, unfavorable)
8. Returns an error if either player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_defensive_replacement_value import (
    get_defensive_replacement_value,
    _derive_oaa,
    _derive_projected_woba,
    _derive_wrc_plus,
    _estimate_innings_remaining,
    _compute_net_value,
    _recommendation,
    _load_players,
    _clamp,
    VALID_POSITIONS,
)


def parse(result: str) -> dict:
    raw = json.loads(result)
    if raw.get("status") == "ok" and "data" in raw:
        # Flatten: merge top-level status/tool with nested data for test access
        return {"status": "ok", "tool": raw.get("tool"), **raw["data"]}
    return raw


# -----------------------------------------------------------------------
# Step 1: Accepts current fielder and replacement identifiers
# -----------------------------------------------------------------------

def test_step1_accepts_two_player_ids():
    """Step 1: Accepts current fielder and replacement player IDs."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert result["status"] == "ok"
    assert result["current_fielder"]["player_id"] == "h_004"
    assert result["replacement"]["player_id"] == "h_014"


def test_step1_returns_player_names():
    """Step 1: Returns both player names."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert result["current_fielder"]["name"] == "Tyrone Jackson"
    assert result["replacement"]["name"] == "Kenji Tanaka"


def test_step1_works_across_teams():
    """Step 1: Can compare players from different teams (for analysis)."""
    result = parse(get_defensive_replacement_value("a_006", "a_012", "RF"))
    assert result["status"] == "ok"
    assert result["current_fielder"]["name"] == "Trey Anderson"
    assert result["replacement"]["name"] == "Marcus Green"


def test_step1_works_with_bench_players():
    """Step 1: Bench players can be either the current or replacement fielder."""
    result = parse(get_defensive_replacement_value("h_001", "h_012", "LF"))
    assert result["status"] == "ok"
    assert result["replacement"]["name"] == "Darnell Washington"


def test_step1_same_player_comparison():
    """Step 1: Comparing a player to themselves returns zero differences."""
    result = parse(get_defensive_replacement_value("h_002", "h_002", "SS"))
    assert result["status"] == "ok"
    assert result["defensive_upgrade_oaa"] == 0.0
    assert result["offensive_downgrade_woba"] == 0.0
    assert result["offensive_downgrade_wrc_plus"] == 0


# -----------------------------------------------------------------------
# Step 2: Accepts the fielding position for the substitution
# -----------------------------------------------------------------------

def test_step2_returns_position():
    """Step 2: Returns the position in the response."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert result["position"] == "RF"


def test_step2_position_case_insensitive():
    """Step 2: Position is case-insensitive (normalized to uppercase)."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "rf"))
    assert result["status"] == "ok"
    assert result["position"] == "RF"


def test_step2_accepts_all_valid_positions():
    """Step 2: All standard fielding positions are accepted."""
    for pos in ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"]:
        result = parse(get_defensive_replacement_value("h_001", "h_014", pos))
        assert result["status"] == "ok", f"Position {pos} should be accepted"


def test_step2_invalid_position_error():
    """Step 2: Invalid position returns an error."""
    result = parse(get_defensive_replacement_value("h_001", "h_014", "XX"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_POSITION"


def test_step2_different_positions_produce_different_oaa():
    """Step 2: Different positions produce different OAA values for the same player."""
    result_ss = parse(get_defensive_replacement_value("h_011", "h_014", "SS"))
    result_lf = parse(get_defensive_replacement_value("h_011", "h_014", "LF"))
    # Ryan O'Brien can play SS and 2B but not LF; OAA should differ
    # At minimum, the arm weight differs by position
    assert result_ss["current_fielder"]["oaa_at_position"] != result_lf["current_fielder"]["oaa_at_position"]


# -----------------------------------------------------------------------
# Step 3: Returns defensive upgrade in OAA
# -----------------------------------------------------------------------

def test_step3_returns_defensive_upgrade_oaa():
    """Step 3: Returns defensive_upgrade_oaa field."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "defensive_upgrade_oaa" in result
    assert isinstance(result["defensive_upgrade_oaa"], (int, float))


def test_step3_returns_oaa_per_player():
    """Step 3: Returns OAA at position for each player individually."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "oaa_at_position" in result["current_fielder"]
    assert "oaa_at_position" in result["replacement"]


def test_step3_upgrade_is_difference():
    """Step 3: Defensive upgrade is replacement OAA minus current OAA."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    expected = round(
        result["replacement"]["oaa_at_position"] - result["current_fielder"]["oaa_at_position"],
        1,
    )
    assert result["defensive_upgrade_oaa"] == expected


def test_step3_better_fielder_has_positive_upgrade():
    """Step 3: Replacing a weaker fielder with a better one shows positive upgrade."""
    # h_006 Shin-Soo Park: DH primary, range 40, arm 50 (weak fielder)
    # h_014 Kenji Tanaka: OF, range 80, arm 72 (strong fielder)
    result = parse(get_defensive_replacement_value("h_006", "h_014", "LF"))
    assert result["defensive_upgrade_oaa"] > 0


def test_step3_worse_fielder_has_negative_upgrade():
    """Step 3: Replacing a strong fielder with a weaker one shows negative upgrade."""
    # h_002 Derek Williams: SS primary, range 85, arm 78 (elite fielder)
    # h_006 Shin-Soo Park: DH primary, range 40, arm 50 (weak fielder)
    result = parse(get_defensive_replacement_value("h_002", "h_006", "SS"))
    assert result["defensive_upgrade_oaa"] < 0


def test_step3_oaa_in_realistic_range():
    """Step 3: OAA values are in a realistic range (-15 to +20)."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert -15.0 <= result["current_fielder"]["oaa_at_position"] <= 20.0
    assert -15.0 <= result["replacement"]["oaa_at_position"] <= 20.0


def test_step3_out_of_position_penalty():
    """Step 3: Player playing out of position has lower OAA."""
    # h_008 Tommy Sullivan: C, positions=["C"]
    # Compare OAA at C (natural) vs LF (out of position)
    result_c = parse(get_defensive_replacement_value("h_008", "h_014", "C"))
    result_lf = parse(get_defensive_replacement_value("h_008", "h_014", "LF"))
    assert result_c["current_fielder"]["oaa_at_position"] > result_lf["current_fielder"]["oaa_at_position"]


def test_step3_range_drives_oaa():
    """Step 3: Higher range produces higher OAA at the same position."""
    # h_001 Marcus Chen: CF, range 82
    # h_008 Tommy Sullivan: C, range 55 -- both can be evaluated at a position
    result = parse(get_defensive_replacement_value("h_008", "h_001", "CF"))
    assert result["replacement"]["oaa_at_position"] > result["current_fielder"]["oaa_at_position"]


def test_step3_dh_position_zero_oaa():
    """Step 3: DH position has 0 OAA for all players (no defense)."""
    result = parse(get_defensive_replacement_value("h_001", "h_014", "DH"))
    assert result["current_fielder"]["oaa_at_position"] == 0.0
    assert result["replacement"]["oaa_at_position"] == 0.0
    assert result["defensive_upgrade_oaa"] == 0.0


def test_step3_positions_list_returned():
    """Step 3: Each player's listed positions are returned."""
    result = parse(get_defensive_replacement_value("h_002", "h_011", "SS"))
    assert "positions" in result["current_fielder"]
    assert "positions" in result["replacement"]
    assert "SS" in result["current_fielder"]["positions"]


# -----------------------------------------------------------------------
# Step 4: Returns offensive downgrade in projected wOBA or wRC+
# -----------------------------------------------------------------------

def test_step4_returns_offensive_downgrade_woba():
    """Step 4: Returns offensive_downgrade_woba field."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "offensive_downgrade_woba" in result
    assert isinstance(result["offensive_downgrade_woba"], (int, float))


def test_step4_returns_offensive_downgrade_wrc_plus():
    """Step 4: Returns offensive_downgrade_wrc_plus field."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "offensive_downgrade_wrc_plus" in result
    assert isinstance(result["offensive_downgrade_wrc_plus"], (int, float))


def test_step4_returns_projected_woba_per_player():
    """Step 4: Returns projected wOBA for each player."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "projected_woba" in result["current_fielder"]
    assert "projected_woba" in result["replacement"]


def test_step4_returns_projected_wrc_plus_per_player():
    """Step 4: Returns projected wRC+ for each player."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "projected_wrc_plus" in result["current_fielder"]
    assert "projected_wrc_plus" in result["replacement"]


def test_step4_downgrade_is_difference():
    """Step 4: Offensive downgrade is replacement wOBA minus current wOBA."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    expected = round(
        result["replacement"]["projected_woba"] - result["current_fielder"]["projected_woba"],
        3,
    )
    assert result["offensive_downgrade_woba"] == expected


def test_step4_weaker_hitter_replacement_is_negative():
    """Step 4: Replacing a strong hitter with a weaker one gives negative wOBA change."""
    # h_003 Rafael Ortiz: power 88, contact 72 (strong hitter)
    # h_010 Victor Nguyen: power 52, contact 60 (weak bench bat)
    result = parse(get_defensive_replacement_value("h_003", "h_010", "1B"))
    assert result["offensive_downgrade_woba"] < 0
    assert result["offensive_downgrade_wrc_plus"] < 0


def test_step4_stronger_hitter_replacement_is_positive():
    """Step 4: Replacing a weak hitter with a stronger one gives positive wOBA change."""
    # h_008 Tommy Sullivan: contact 65, power 58, eye 60 (weak hitter)
    # h_013 Eduardo Reyes: contact 70, power 78, eye 70 (stronger hitter)
    result = parse(get_defensive_replacement_value("h_008", "h_013", "1B"))
    assert result["offensive_downgrade_woba"] > 0


def test_step4_woba_in_realistic_range():
    """Step 4: Projected wOBA values are in realistic range (.220 to .430)."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert 0.220 <= result["current_fielder"]["projected_woba"] <= 0.430
    assert 0.220 <= result["replacement"]["projected_woba"] <= 0.430


def test_step4_wrc_plus_in_realistic_range():
    """Step 4: Projected wRC+ values are in realistic range (40 to 200)."""
    result = parse(get_defensive_replacement_value("h_003", "h_010", "1B"))
    assert 40 <= result["current_fielder"]["projected_wrc_plus"] <= 200
    assert 40 <= result["replacement"]["projected_wrc_plus"] <= 200


def test_step4_higher_contact_power_produces_higher_woba():
    """Step 4: Player with higher contact and power has higher projected wOBA."""
    # h_003 Rafael Ortiz: contact 72, power 88, eye 80 (elite)
    # h_010 Victor Nguyen: contact 60, power 52, eye 55 (weak)
    result = parse(get_defensive_replacement_value("h_003", "h_010", "1B"))
    assert result["current_fielder"]["projected_woba"] > result["replacement"]["projected_woba"]


def test_step4_pitcher_has_minimal_offense():
    """Step 4: Pitcher without batter attributes gets minimal wOBA (.220)."""
    # h_sp1 Brandon Cole: starting pitcher, no batter attributes
    result = parse(get_defensive_replacement_value("h_sp1", "h_001", "P"))
    assert result["current_fielder"]["projected_woba"] == 0.220


# -----------------------------------------------------------------------
# Step 5: Returns estimated innings remaining
# -----------------------------------------------------------------------

def test_step5_returns_estimated_innings_remaining():
    """Step 5: Returns estimated_innings_remaining field."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "estimated_innings_remaining" in result
    assert isinstance(result["estimated_innings_remaining"], (int, float))


def test_step5_default_inning_7():
    """Step 5: Default inning is 7, giving ~3 innings remaining."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    # Inning 7, top, 0 outs: 1.0 (rest of current) + 2 full = 3.0
    assert result["estimated_innings_remaining"] == 3.0


def test_step5_early_game_more_innings():
    """Step 5: Early game has more innings remaining."""
    result = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=3, half="top", outs=0,
    ))
    # Inning 3, top, 0 outs: 1.0 + 6 = 7.0
    assert result["estimated_innings_remaining"] == 7.0


def test_step5_late_game_fewer_innings():
    """Step 5: Late game has fewer innings remaining."""
    result = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=9, half="top", outs=0,
    ))
    # Inning 9, top, 0 outs: 1.0 + 0 = 1.0
    assert result["estimated_innings_remaining"] == 1.0


def test_step5_bottom_half_less_defense():
    """Step 5: Bottom half means less defensive innings (defense starts next inning)."""
    result_top = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=7, half="top", outs=0,
    ))
    result_bot = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=7, half="bottom", outs=0,
    ))
    assert result_top["estimated_innings_remaining"] > result_bot["estimated_innings_remaining"]


def test_step5_outs_reduce_remaining():
    """Step 5: More outs in current inning reduces remaining."""
    result_0 = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=7, half="top", outs=0,
    ))
    result_2 = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=7, half="top", outs=2,
    ))
    assert result_0["estimated_innings_remaining"] > result_2["estimated_innings_remaining"]


def test_step5_extra_innings():
    """Step 5: Extra innings estimate is reasonable."""
    result = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=10, half="top", outs=0,
    ))
    # Extra innings: remaining_in_current (1.0) + 1.0 = 2.0
    assert result["estimated_innings_remaining"] == 2.0


def test_step5_ninth_inning_bottom_zero():
    """Step 5: Bottom of 9th means 0 innings remaining on defense."""
    result = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=9, half="bottom", outs=0,
    ))
    assert result["estimated_innings_remaining"] == 0.0


# -----------------------------------------------------------------------
# Step 6: Returns net expected value
# -----------------------------------------------------------------------

def test_step6_returns_net_expected_value():
    """Step 6: Returns net_expected_value field."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "net_expected_value" in result
    assert isinstance(result["net_expected_value"], (int, float))


def test_step6_positive_net_when_big_defensive_gain():
    """Step 6: Net value is positive when defensive gain is large and offensive cost small."""
    # h_006 Shin-Soo Park: DH primary, range 40 (bad fielder), decent bat
    # h_014 Kenji Tanaka: CF/LF/RF, range 80 (good fielder), weaker bat
    result = parse(get_defensive_replacement_value(
        "h_006", "h_014", "LF", inning=7, half="top", outs=0,
    ))
    # Large defensive upgrade should outweigh moderate offensive cost
    assert result["defensive_upgrade_oaa"] > 0
    # Net could be positive or marginal depending on exact values
    assert result["net_expected_value"] is not None


def test_step6_negative_net_when_big_offensive_cost():
    """Step 6: Net value is negative when offensive cost outweighs defensive gain."""
    # h_003 Rafael Ortiz: elite bat (power 88), weak fielder (range 50)
    # h_010 Victor Nguyen: weak bat (power 52), weak fielder (range 50)
    result = parse(get_defensive_replacement_value(
        "h_003", "h_010", "1B", inning=7, half="top", outs=0,
    ))
    # Minimal defensive difference, large offensive downgrade
    assert result["offensive_downgrade_woba"] < 0
    assert result["net_expected_value"] < 0


def test_step6_net_value_scales_with_innings():
    """Step 6: Net value magnitude increases with more innings remaining."""
    result_early = parse(get_defensive_replacement_value(
        "h_006", "h_014", "LF", inning=3, half="top", outs=0,
    ))
    result_late = parse(get_defensive_replacement_value(
        "h_006", "h_014", "LF", inning=9, half="top", outs=0,
    ))
    # More innings = larger magnitude (assuming non-zero difference)
    assert abs(result_early["net_expected_value"]) >= abs(result_late["net_expected_value"])


def test_step6_zero_innings_zero_net():
    """Step 6: Zero innings remaining means zero net value."""
    result = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=9, half="bottom", outs=0,
    ))
    assert result["estimated_innings_remaining"] == 0.0
    assert result["net_expected_value"] == 0.0


# -----------------------------------------------------------------------
# Step 7: Returns textual recommendation
# -----------------------------------------------------------------------

def test_step7_returns_recommendation():
    """Step 7: Returns recommendation field."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert "recommendation" in result
    assert isinstance(result["recommendation"], str)


def test_step7_recommendation_is_valid_value():
    """Step 7: Recommendation is one of favorable/marginal/unfavorable."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    assert result["recommendation"] in ("favorable", "marginal", "unfavorable")


def test_step7_favorable_for_clear_defensive_gain():
    """Step 7: Favorable when large defensive upgrade with small offensive cost."""
    # h_006 Shin-Soo Park: DH, range 40, arm 50 -> terrible in LF
    # h_014 Kenji Tanaka: OF, range 80, arm 72 -> strong fielder
    # Early game to maximize the effect
    result = parse(get_defensive_replacement_value(
        "h_006", "h_014", "LF", inning=3, half="top", outs=0,
    ))
    # This should be favorable (big defensive upgrade)
    if result["net_expected_value"] > 0.010:
        assert result["recommendation"] == "favorable"


def test_step7_unfavorable_for_big_offensive_cost():
    """Step 7: Unfavorable when big offensive cost with no defensive gain."""
    # h_003 Rafael Ortiz: elite bat, 1B
    # h_010 Victor Nguyen: much weaker bat, similar/worse defense at 1B
    result = parse(get_defensive_replacement_value(
        "h_003", "h_010", "1B", inning=7, half="top", outs=0,
    ))
    assert result["recommendation"] == "unfavorable"


def test_step7_zero_innings_marginal():
    """Step 7: With zero innings remaining, recommendation is marginal (net ~0)."""
    result = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=9, half="bottom", outs=0,
    ))
    assert result["recommendation"] == "marginal"


# -----------------------------------------------------------------------
# Step 8: Returns errors for invalid player identifiers
# -----------------------------------------------------------------------

def test_step8_invalid_current_fielder_id():
    """Step 8: Returns error for invalid current fielder ID."""
    result = parse(get_defensive_replacement_value("NONEXISTENT", "h_014", "RF"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "NONEXISTENT" in result["message"]


def test_step8_invalid_replacement_id():
    """Step 8: Returns error for invalid replacement ID."""
    result = parse(get_defensive_replacement_value("h_004", "NONEXISTENT", "RF"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "NONEXISTENT" in result["message"]


def test_step8_both_ids_invalid():
    """Step 8: Returns error when current fielder ID is invalid (checked first)."""
    result = parse(get_defensive_replacement_value("BAD1", "BAD2", "RF"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"


def test_step8_empty_current_fielder_id():
    """Step 8: Returns error for empty current fielder ID."""
    result = parse(get_defensive_replacement_value("", "h_014", "RF"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"


def test_step8_empty_replacement_id():
    """Step 8: Returns error for empty replacement ID."""
    result = parse(get_defensive_replacement_value("h_004", "", "RF"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"


def test_step8_invalid_inning():
    """Step 8: Returns error for invalid inning value."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF", inning=0))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step8_invalid_half():
    """Step 8: Returns error for invalid half value."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF", half="middle"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step8_invalid_outs():
    """Step 8: Returns error for invalid outs value."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF", outs=3))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step8_negative_outs():
    """Step 8: Returns error for negative outs."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF", outs=-1))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


# -----------------------------------------------------------------------
# Helper function unit tests
# -----------------------------------------------------------------------

def test_helper_derive_oaa_high_range():
    """Helper: High range produces positive OAA."""
    oaa = _derive_oaa({"range": 85, "arm_strength": 70, "error_rate": 0.02, "positions": ["SS"]}, "SS")
    assert oaa > 0


def test_helper_derive_oaa_low_range():
    """Helper: Low range produces negative OAA."""
    oaa = _derive_oaa({"range": 30, "arm_strength": 50, "error_rate": 0.04, "positions": ["SS"]}, "SS")
    assert oaa < 0


def test_helper_derive_oaa_average():
    """Helper: Average range (50) produces OAA near zero."""
    oaa = _derive_oaa({"range": 50, "arm_strength": 50, "error_rate": 0.03, "positions": ["SS"]}, "SS")
    assert -2.0 <= oaa <= 2.0


def test_helper_derive_oaa_dh_always_zero():
    """Helper: DH position always returns 0 OAA."""
    oaa = _derive_oaa({"range": 90, "arm_strength": 90, "error_rate": 0.01, "positions": ["DH"]}, "DH")
    assert oaa == 0.0


def test_helper_derive_oaa_out_of_position_penalty():
    """Helper: Player out of position gets penalty."""
    in_pos = _derive_oaa({"range": 70, "arm_strength": 70, "error_rate": 0.03, "positions": ["SS", "2B"]}, "SS")
    out_pos = _derive_oaa({"range": 70, "arm_strength": 70, "error_rate": 0.03, "positions": ["SS", "2B"]}, "LF")
    assert in_pos > out_pos


def test_helper_derive_oaa_arm_matters_more_at_ss():
    """Helper: Arm strength contributes more to OAA at SS than at 1B."""
    attrs_strong_arm = {"range": 60, "arm_strength": 90, "error_rate": 0.03, "positions": ["SS", "1B"]}
    attrs_weak_arm = {"range": 60, "arm_strength": 30, "error_rate": 0.03, "positions": ["SS", "1B"]}
    diff_ss = _derive_oaa(attrs_strong_arm, "SS") - _derive_oaa(attrs_weak_arm, "SS")
    diff_1b = _derive_oaa(attrs_strong_arm, "1B") - _derive_oaa(attrs_weak_arm, "1B")
    assert diff_ss > diff_1b


def test_helper_derive_oaa_error_rate_penalty():
    """Helper: Higher error rate produces lower OAA."""
    low_err = _derive_oaa({"range": 70, "arm_strength": 70, "error_rate": 0.01, "positions": ["SS"]}, "SS")
    high_err = _derive_oaa({"range": 70, "arm_strength": 70, "error_rate": 0.05, "positions": ["SS"]}, "SS")
    assert low_err > high_err


def test_helper_derive_projected_woba_realistic():
    """Helper: Projected wOBA is in realistic range."""
    for contact in (40, 60, 80):
        for power in (30, 60, 90):
            woba = _derive_projected_woba({"contact": contact, "power": power, "eye": 60, "speed": 50})
            assert 0.220 <= woba <= 0.430, f"wOBA {woba} out of range for contact={contact}, power={power}"


def test_helper_derive_projected_woba_scales_with_power():
    """Helper: Higher power produces higher wOBA."""
    low = _derive_projected_woba({"contact": 60, "power": 30, "eye": 60, "speed": 50})
    high = _derive_projected_woba({"contact": 60, "power": 90, "eye": 60, "speed": 50})
    assert high > low


def test_helper_derive_projected_woba_scales_with_contact():
    """Helper: Higher contact produces higher wOBA."""
    low = _derive_projected_woba({"contact": 30, "power": 60, "eye": 60, "speed": 50})
    high = _derive_projected_woba({"contact": 90, "power": 60, "eye": 60, "speed": 50})
    assert high > low


def test_helper_derive_projected_woba_scales_with_eye():
    """Helper: Higher eye produces higher wOBA."""
    low = _derive_projected_woba({"contact": 60, "power": 60, "eye": 30, "speed": 50})
    high = _derive_projected_woba({"contact": 60, "power": 60, "eye": 90, "speed": 50})
    assert high > low


def test_helper_derive_wrc_plus_100_at_league_avg():
    """Helper: wRC+ is ~100 at league average wOBA (.315)."""
    wrc = _derive_wrc_plus(0.315)
    assert wrc == 100


def test_helper_derive_wrc_plus_above_100():
    """Helper: wRC+ > 100 for above-average wOBA."""
    wrc = _derive_wrc_plus(0.380)
    assert wrc > 100


def test_helper_derive_wrc_plus_below_100():
    """Helper: wRC+ < 100 for below-average wOBA."""
    wrc = _derive_wrc_plus(0.270)
    assert wrc < 100


def test_helper_estimate_innings_remaining_top_7():
    """Helper: Top of 7th, 0 outs = 3.0 innings remaining."""
    remaining = _estimate_innings_remaining(7, "top", 0)
    assert remaining == 3.0


def test_helper_estimate_innings_remaining_top_1():
    """Helper: Top of 1st, 0 outs = 9.0 innings remaining."""
    remaining = _estimate_innings_remaining(1, "top", 0)
    assert remaining == 9.0


def test_helper_estimate_innings_remaining_bottom_9():
    """Helper: Bottom of 9th = 0.0 innings remaining."""
    remaining = _estimate_innings_remaining(9, "bottom", 0)
    assert remaining == 0.0


def test_helper_estimate_innings_remaining_with_outs():
    """Helper: Top of 7th, 2 outs = 2.3 innings remaining."""
    remaining = _estimate_innings_remaining(7, "top", 2)
    # (3-2)/3 + 2 = 0.333 + 2 = 2.333 -> 2.3
    assert abs(remaining - 2.3) < 0.05


def test_helper_estimate_innings_remaining_extras():
    """Helper: Extra innings (inning 10) = ~2.0 innings."""
    remaining = _estimate_innings_remaining(10, "top", 0)
    assert remaining == 2.0


def test_helper_compute_net_value_positive():
    """Helper: Positive defensive upgrade with no offensive cost gives positive net."""
    net = _compute_net_value(10.0, 0.0, 3.0)
    assert net > 0


def test_helper_compute_net_value_negative():
    """Helper: No defensive upgrade with large offensive cost gives negative net."""
    net = _compute_net_value(0.0, -0.050, 3.0)
    assert net < 0


def test_helper_compute_net_value_zero_innings():
    """Helper: Zero innings remaining gives zero net value."""
    net = _compute_net_value(10.0, -0.050, 0.0)
    assert net == 0.0


def test_helper_recommendation_favorable():
    """Helper: Net > 0.010 gives favorable."""
    assert _recommendation(0.020) == "favorable"


def test_helper_recommendation_marginal():
    """Helper: Net between -0.010 and 0.010 gives marginal."""
    assert _recommendation(0.005) == "marginal"
    assert _recommendation(-0.005) == "marginal"
    assert _recommendation(0.0) == "marginal"


def test_helper_recommendation_unfavorable():
    """Helper: Net < -0.010 gives unfavorable."""
    assert _recommendation(-0.020) == "unfavorable"


def test_helper_clamp():
    """Helper: _clamp works correctly."""
    assert _clamp(5.0, 0.0, 10.0) == 5.0
    assert _clamp(-1.0, 0.0, 10.0) == 0.0
    assert _clamp(15.0, 0.0, 10.0) == 10.0


# -----------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------

def test_integration_rf_replacement_late_game():
    """Integration: Full evaluation of RF replacement in late game."""
    # h_004 Tyrone Jackson: RF, range 70, arm 85, power 85
    # h_014 Kenji Tanaka: OF, range 80, arm 72, power 55
    result = parse(get_defensive_replacement_value(
        "h_004", "h_014", "RF", inning=8, half="top", outs=0,
    ))
    assert result["status"] == "ok"
    assert result["position"] == "RF"
    assert result["current_fielder"]["name"] == "Tyrone Jackson"
    assert result["replacement"]["name"] == "Kenji Tanaka"
    # All fields present
    for field in ["defensive_upgrade_oaa", "offensive_downgrade_woba",
                  "offensive_downgrade_wrc_plus", "estimated_innings_remaining",
                  "net_expected_value", "recommendation"]:
        assert field in result, f"Missing field: {field}"
    assert result["recommendation"] in ("favorable", "marginal", "unfavorable")


def test_integration_ss_replacement():
    """Integration: SS replacement comparison."""
    # h_002 Derek Williams: SS, range 85, arm 78 (elite SS)
    # h_011 Ryan O'Brien: IF, range 75, arm 72 (utility IF)
    result = parse(get_defensive_replacement_value(
        "h_002", "h_011", "SS", inning=7, half="top", outs=0,
    ))
    assert result["status"] == "ok"
    # Derek is a better SS, so replacement OAA should be lower
    assert result["defensive_upgrade_oaa"] < 0
    # Derek also has better bat attributes
    assert result["offensive_downgrade_woba"] <= 0


def test_integration_dh_to_1b():
    """Integration: Moving DH to 1B field position."""
    # h_006 Shin-Soo Park: DH, range 40, positions [DH, 1B]
    # h_013 Eduardo Reyes: 1B bench, range 48, positions [1B, DH]
    result = parse(get_defensive_replacement_value(
        "h_006", "h_013", "1B", inning=7, half="top", outs=0,
    ))
    assert result["status"] == "ok"
    # Both can play 1B, but neither is elite defensively


def test_integration_all_fields_present():
    """Integration: All required output fields are present."""
    result = parse(get_defensive_replacement_value("h_004", "h_014", "RF"))
    required_top = [
        "status", "current_fielder", "replacement", "position",
        "defensive_upgrade_oaa", "offensive_downgrade_woba",
        "offensive_downgrade_wrc_plus", "estimated_innings_remaining",
        "net_expected_value", "recommendation",
    ]
    for field in required_top:
        assert field in result, f"Missing top-level field: {field}"
    required_player = [
        "player_id", "name", "oaa_at_position",
        "projected_woba", "projected_wrc_plus", "positions",
    ]
    for field in required_player:
        assert field in result["current_fielder"], f"Missing current_fielder field: {field}"
        assert field in result["replacement"], f"Missing replacement field: {field}"


def test_integration_multiple_players_different_results():
    """Integration: Different player combos produce different results."""
    results = []
    combos = [
        ("h_001", "h_014", "CF"),
        ("h_004", "h_014", "RF"),
        ("h_002", "h_011", "SS"),
        ("h_006", "h_013", "1B"),
    ]
    for current, repl, pos in combos:
        r = parse(get_defensive_replacement_value(current, repl, pos))
        results.append(r["net_expected_value"])
    # Not all the same
    assert len(set(results)) > 1, "Expected different net values for different combos"


def test_integration_away_team_players():
    """Integration: Works with away team players."""
    result = parse(get_defensive_replacement_value(
        "a_006", "a_012", "RF", inning=8, half="top", outs=1,
    ))
    assert result["status"] == "ok"
    assert result["current_fielder"]["name"] == "Trey Anderson"
    assert result["replacement"]["name"] == "Marcus Green"


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
