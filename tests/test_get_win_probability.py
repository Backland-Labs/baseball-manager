# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_win_probability tool.

Verifies all feature requirements from features.json:
1. Accepts inning, half (top/bottom), outs, base state, and score differential
2. Optionally accepts home/away indicator for the managed team
3. Returns current win probability for the managed team
4. Returns leverage index for the current situation
5. Returns conditional win probability if a run scores in this state
6. Returns conditional win probability if the inning ends scoreless
7. Values are derived from pre-computed win probability tables
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_win_probability import get_win_probability


def parse(result: str) -> dict:
    """Parse JSON response. For success responses, merge data into top level for easy access."""
    d = json.loads(result)
    if d.get("status") == "ok" and "data" in d:
        return {"status": "ok", "tool": d.get("tool"), **d["data"]}
    return d


# -----------------------------------------------------------------------
# Step 1: Accepts inning, half (top/bottom), outs, base state, and score diff
# -----------------------------------------------------------------------

def test_step1_basic_call():
    """Step 1: Accepts basic game state parameters."""
    result = parse(get_win_probability(5, "TOP", 1, False, False, False, 0))
    assert result["status"] == "ok"
    assert result["game_state"]["inning"] == 5
    assert result["game_state"]["half"] == "TOP"
    assert result["game_state"]["outs"] == 1
    assert result["game_state"]["score_differential"] == 0


def test_step1_all_base_states():
    """Step 1: Accepts all 8 base state combinations."""
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
        result = parse(get_win_probability(5, "TOP", 1, first, second, third, 0))
        assert result["status"] == "ok", f"Failed for ({first},{second},{third})"
        assert result["game_state"]["runners"]["first"] == first
        assert result["game_state"]["runners"]["second"] == second
        assert result["game_state"]["runners"]["third"] == third


def test_step1_all_out_counts():
    """Step 1: Accepts 0, 1, and 2 outs."""
    for outs in (0, 1, 2):
        result = parse(get_win_probability(5, "TOP", outs, False, False, False, 0))
        assert result["status"] == "ok"
        assert result["game_state"]["outs"] == outs


def test_step1_all_innings():
    """Step 1: Accepts innings 1 through 9 and extra innings."""
    for inning in range(1, 13):
        result = parse(get_win_probability(inning, "TOP", 0, False, False, False, 0))
        assert result["status"] == "ok", f"Failed for inning {inning}"


def test_step1_both_halves():
    """Step 1: Accepts TOP and BOTTOM half."""
    for half in ("TOP", "BOTTOM"):
        result = parse(get_win_probability(5, half, 0, False, False, False, 0))
        assert result["status"] == "ok"
        assert result["game_state"]["half"] == half


def test_step1_score_differentials():
    """Step 1: Accepts various score differentials (positive = leading)."""
    for diff in (-5, -3, -1, 0, 1, 3, 5):
        result = parse(get_win_probability(5, "TOP", 0, False, False, False, diff))
        assert result["status"] == "ok"
        assert result["game_state"]["score_differential"] == diff


def test_step1_case_insensitive_half():
    """Step 1: Accepts case-insensitive half values."""
    for half in ("top", "Top", "TOP", "bottom", "Bottom", "BOTTOM"):
        result = parse(get_win_probability(5, half, 0, False, False, False, 0))
        assert result["status"] == "ok"


def test_step1_invalid_inning():
    """Step 1: Rejects invalid inning values."""
    result = parse(get_win_probability(0, "TOP", 0, False, False, False, 0))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"

    result = parse(get_win_probability(-1, "TOP", 0, False, False, False, 0))
    assert result["status"] == "error"


def test_step1_invalid_half():
    """Step 1: Rejects invalid half values."""
    result = parse(get_win_probability(5, "MIDDLE", 0, False, False, False, 0))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step1_invalid_outs():
    """Step 1: Rejects invalid outs values."""
    for bad_outs in (-1, 3, 4):
        result = parse(get_win_probability(5, "TOP", bad_outs, False, False, False, 0))
        assert result["status"] == "error"
        assert result["error_code"] == "INVALID_PARAMETER"


# -----------------------------------------------------------------------
# Step 2: Optionally accepts home/away indicator
# -----------------------------------------------------------------------

def test_step2_default_away():
    """Step 2: Default (no managed_team_home) treats managed team as away."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    assert result["status"] == "ok"
    assert result["game_state"]["managed_team_home"] is False


def test_step2_explicit_home():
    """Step 2: managed_team_home=True sets managed team as home."""
    result = parse(get_win_probability(
        5, "TOP", 0, False, False, False, 0, managed_team_home=True
    ))
    assert result["status"] == "ok"
    assert result["game_state"]["managed_team_home"] is True


def test_step2_explicit_away():
    """Step 2: managed_team_home=False sets managed team as away."""
    result = parse(get_win_probability(
        5, "TOP", 0, False, False, False, 0, managed_team_home=False
    ))
    assert result["status"] == "ok"
    assert result["game_state"]["managed_team_home"] is False


def test_step2_home_away_complementary():
    """Step 2: Home team WP + away team WP should equal ~1.0 for same state."""
    away_result = parse(get_win_probability(
        5, "TOP", 0, False, False, False, 2, managed_team_home=False
    ))
    home_result = parse(get_win_probability(
        5, "TOP", 0, False, False, False, -2, managed_team_home=True
    ))
    # Both teams are in the same game state (away leads by 2)
    # Away team WP + Home team WP should be ~1.0
    total = away_result["win_probability"] + home_result["win_probability"]
    assert abs(total - 1.0) < 0.02, f"WP sum {total} not close to 1.0"


# -----------------------------------------------------------------------
# Step 3: Returns current win probability for the managed team
# -----------------------------------------------------------------------

def test_step3_wp_present():
    """Step 3: Response includes win_probability field."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    assert "win_probability" in result
    assert isinstance(result["win_probability"], float)


def test_step3_wp_range():
    """Step 3: Win probability is between 0.01 and 0.99."""
    for inning in range(1, 12):
        for half in ("TOP", "BOTTOM"):
            for diff in (-5, -2, 0, 2, 5):
                result = parse(get_win_probability(inning, half, 0, False, False, False, diff))
                wp = result["win_probability"]
                assert 0.01 <= wp <= 0.99, f"WP {wp} out of range for inning={inning}, half={half}, diff={diff}"


def test_step3_wp_tied_game_near_50():
    """Step 3: Tied game in early innings should be near 50%."""
    result = parse(get_win_probability(1, "TOP", 0, False, False, False, 0))
    wp = result["win_probability"]
    assert 0.40 <= wp <= 0.55, f"Tied game WP {wp} too far from 0.50"


def test_step3_wp_increases_with_lead():
    """Step 3: WP should increase as the managed team leads by more."""
    wps = []
    for diff in (-3, -1, 0, 1, 3):
        result = parse(get_win_probability(5, "TOP", 0, False, False, False, diff))
        wps.append(result["win_probability"])
    # WP should be monotonically increasing with score differential
    for i in range(len(wps) - 1):
        assert wps[i] < wps[i + 1], f"WP not increasing: {wps}"


def test_step3_wp_late_inning_lead_higher():
    """Step 3: A 1-run lead in the 9th should have higher WP than in the 1st."""
    early = parse(get_win_probability(1, "TOP", 0, False, False, False, 1))
    late = parse(get_win_probability(9, "TOP", 0, False, False, False, 1))
    assert late["win_probability"] > early["win_probability"], \
        f"Late WP ({late['win_probability']}) not > early WP ({early['win_probability']})"


def test_step3_wp_home_advantage():
    """Step 3: Home team should have slightly higher WP in tied game (last at-bat advantage)."""
    away_wp = parse(get_win_probability(
        5, "TOP", 0, False, False, False, 0, managed_team_home=False
    ))["win_probability"]
    home_wp = parse(get_win_probability(
        5, "TOP", 0, False, False, False, 0, managed_team_home=True
    ))["win_probability"]
    assert home_wp > away_wp, f"Home WP ({home_wp}) not > away WP ({away_wp})"


def test_step3_large_lead_high_wp():
    """Step 3: A large lead should result in very high WP."""
    result = parse(get_win_probability(9, "TOP", 0, False, False, False, 5))
    assert result["win_probability"] >= 0.90, f"5-run lead in 9th: WP {result['win_probability']} not >= 0.90"


def test_step3_large_deficit_low_wp():
    """Step 3: A large deficit should result in very low WP."""
    result = parse(get_win_probability(9, "TOP", 0, False, False, False, -5))
    assert result["win_probability"] < 0.10, f"5-run deficit in 9th: WP {result['win_probability']} not < 0.10"


def test_step3_runners_on_affect_wp():
    """Step 3: Runners on base should affect WP for the batting team."""
    empty = parse(get_win_probability(
        7, "TOP", 0, False, False, False, 0, managed_team_home=False
    ))["win_probability"]
    loaded = parse(get_win_probability(
        7, "TOP", 0, True, True, True, 0, managed_team_home=False
    ))["win_probability"]
    # Away team is batting in TOP, runners on base help them
    assert loaded > empty, f"Loaded WP ({loaded}) not > empty WP ({empty}) in TOP"


# -----------------------------------------------------------------------
# Step 4: Returns leverage index for the current situation
# -----------------------------------------------------------------------

def test_step4_li_present():
    """Step 4: Response includes leverage_index field."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    assert "leverage_index" in result
    assert isinstance(result["leverage_index"], float)


def test_step4_li_positive():
    """Step 4: Leverage index is always positive."""
    for inning in (1, 5, 9):
        for diff in (-5, 0, 5):
            result = parse(get_win_probability(inning, "TOP", 0, False, False, False, diff))
            assert result["leverage_index"] > 0, f"LI not positive for inn={inning}, diff={diff}"


def test_step4_li_average_near_1():
    """Step 4: Tied game, middle innings should have LI roughly near 1.0."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    li = result["leverage_index"]
    assert 0.3 <= li <= 3.0, f"Average-situation LI {li} too far from 1.0"


def test_step4_li_high_leverage_situation():
    """Step 4: Close game, late inning, runners on = high leverage."""
    # Tied game, 9th inning, top half (away batting), runners on 1st and 2nd, 1 out
    result = parse(get_win_probability(9, "TOP", 1, True, True, False, 0))
    li = result["leverage_index"]
    assert li > 1.0, f"9th inning tied game runners on LI {li} not > 1.0"


def test_step4_li_blowout_low():
    """Step 4: Blowout should have low leverage."""
    result = parse(get_win_probability(7, "TOP", 0, False, False, False, 8))
    li = result["leverage_index"]
    assert li < 1.5, f"Blowout LI {li} not < 1.5"


# -----------------------------------------------------------------------
# Step 5: Returns conditional WP if a run scores
# -----------------------------------------------------------------------

def test_step5_wp_if_run_present():
    """Step 5: Response includes wp_if_run_scores field."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    assert "wp_if_run_scores" in result
    assert isinstance(result["wp_if_run_scores"], float)


def test_step5_wp_if_run_range():
    """Step 5: Conditional WP if run scores is between 0.01 and 0.99."""
    result = parse(get_win_probability(5, "TOP", 0, True, False, False, 0))
    wp = result["wp_if_run_scores"]
    assert 0.01 <= wp <= 0.99


def test_step5_wp_if_run_higher_than_current():
    """Step 5: WP if managed team scores should be higher than current WP."""
    result = parse(get_win_probability(
        7, "TOP", 0, True, False, False, 0, managed_team_home=False
    ))
    # Away team batting in TOP -- a run should help them
    assert result["wp_if_run_scores"] > result["win_probability"], \
        f"WP_run ({result['wp_if_run_scores']}) not > current WP ({result['win_probability']})"


def test_step5_wp_if_run_effect_stronger_in_close_games():
    """Step 5: A run's WP impact should be larger in close games than blowouts."""
    close = parse(get_win_probability(7, "TOP", 0, False, False, False, 0))
    close_delta = close["wp_if_run_scores"] - close["win_probability"]

    blowout = parse(get_win_probability(7, "TOP", 0, False, False, False, 5))
    blowout_delta = blowout["wp_if_run_scores"] - blowout["win_probability"]

    assert close_delta > blowout_delta, \
        f"Close game delta ({close_delta}) not > blowout delta ({blowout_delta})"


# -----------------------------------------------------------------------
# Step 6: Returns conditional WP if inning ends scoreless
# -----------------------------------------------------------------------

def test_step6_wp_if_scoreless_present():
    """Step 6: Response includes wp_if_inning_ends_scoreless field."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    assert "wp_if_inning_ends_scoreless" in result
    assert isinstance(result["wp_if_inning_ends_scoreless"], float)


def test_step6_wp_if_scoreless_range():
    """Step 6: Conditional WP if scoreless is between 0.01 and 0.99."""
    result = parse(get_win_probability(5, "TOP", 0, True, False, False, 0))
    wp = result["wp_if_scoreless_end"] if "wp_if_scoreless_end" in result else result["wp_if_inning_ends_scoreless"]
    assert 0.01 <= wp <= 0.99


def test_step6_wp_if_scoreless_lower_for_batting_team():
    """Step 6: Scoreless inning should lower WP for the batting team in a tied game."""
    result = parse(get_win_probability(
        7, "TOP", 0, True, True, False, 0, managed_team_home=False
    ))
    # Away team batting in TOP with runners on -- scoreless outcome is worse for them
    # compared to having runners on (current state)
    # The scoreless WP should be less than current WP since batting team loses their runners
    wp_current = result["win_probability"]
    wp_scoreless = result["wp_if_inning_ends_scoreless"]
    assert wp_scoreless < wp_current, \
        f"WP_scoreless ({wp_scoreless}) not < current WP ({wp_current}) for batting team with runners"


def test_step6_run_vs_scoreless_makes_sense():
    """Step 6: WP_if_run > WP_current > WP_if_scoreless for batting team with runners."""
    result = parse(get_win_probability(
        7, "TOP", 0, True, False, False, 0, managed_team_home=False
    ))
    wp = result["win_probability"]
    wp_run = result["wp_if_run_scores"]
    wp_scoreless = result["wp_if_inning_ends_scoreless"]
    # For the batting team: run helps, scoreless hurts
    assert wp_run > wp, f"WP_run ({wp_run}) not > WP ({wp})"
    assert wp_scoreless < wp, f"WP_scoreless ({wp_scoreless}) not < WP ({wp})"


# -----------------------------------------------------------------------
# Step 7: Values derived from pre-computed win probability tables
# -----------------------------------------------------------------------

def test_step7_deterministic():
    """Step 7: Same inputs always produce the same output."""
    r1 = parse(get_win_probability(7, "BOTTOM", 1, True, False, False, -1))
    r2 = parse(get_win_probability(7, "BOTTOM", 1, True, False, False, -1))
    assert r1["win_probability"] == r2["win_probability"]
    assert r1["leverage_index"] == r2["leverage_index"]
    assert r1["wp_if_run_scores"] == r2["wp_if_run_scores"]
    assert r1["wp_if_inning_ends_scoreless"] == r2["wp_if_inning_ends_scoreless"]


def test_step7_distinct_states_different_wp():
    """Step 7: Different game states produce different win probabilities."""
    r1 = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    r2 = parse(get_win_probability(9, "BOTTOM", 2, True, True, True, -3))
    assert r1["win_probability"] != r2["win_probability"]


def test_step7_extra_innings():
    """Step 7: Extra innings (10+) produce valid results."""
    for inning in (10, 11, 12):
        result = parse(get_win_probability(inning, "TOP", 0, False, False, False, 0))
        assert result["status"] == "ok"
        assert 0.01 <= result["win_probability"] <= 0.99


def test_step7_extreme_differential_clamp():
    """Step 7: Extreme score differentials are handled (clamped) gracefully."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 15))
    assert result["status"] == "ok"
    assert result["win_probability"] > 0.90

    result = parse(get_win_probability(5, "TOP", 0, False, False, False, -15))
    assert result["status"] == "ok"
    assert result["win_probability"] < 0.10


# -----------------------------------------------------------------------
# Additional integration and edge case tests
# -----------------------------------------------------------------------

def test_response_structure():
    """Full response has all required top-level fields."""
    result = parse(get_win_probability(5, "TOP", 0, False, False, False, 0))
    assert "status" in result
    assert "game_state" in result
    assert "win_probability" in result
    assert "leverage_index" in result
    assert "wp_if_run_scores" in result
    assert "wp_if_inning_ends_scoreless" in result


def test_game_state_echo():
    """Response echoes back the game state for verification."""
    result = parse(get_win_probability(7, "BOTTOM", 2, True, False, True, -3, managed_team_home=True))
    gs = result["game_state"]
    assert gs["inning"] == 7
    assert gs["half"] == "BOTTOM"
    assert gs["outs"] == 2
    assert gs["runners"]["first"] is True
    assert gs["runners"]["second"] is False
    assert gs["runners"]["third"] is True
    assert gs["score_differential"] == -3
    assert gs["managed_team_home"] is True


def test_json_parseable():
    """All common states return valid JSON."""
    for inning in (1, 5, 9):
        for half in ("TOP", "BOTTOM"):
            for outs in (0, 1, 2):
                for diff in (-3, 0, 3):
                    raw = get_win_probability(inning, half, outs, False, False, False, diff)
                    data = json.loads(raw)
                    assert isinstance(data, dict)


def test_wp_symmetric_score():
    """WP at +N for away should be complement of WP at -N for home (approximately)."""
    # Away team leading by 2
    away_up_2 = parse(get_win_probability(
        5, "TOP", 0, False, False, False, 2, managed_team_home=False
    ))["win_probability"]
    # Home team trailing by 2 (same game state, from home perspective)
    home_down_2 = parse(get_win_probability(
        5, "TOP", 0, False, False, False, -2, managed_team_home=True
    ))["win_probability"]
    # These should be the same game from different perspectives: away up 2
    # away_up_2 + home_down_2 should be ~1.0
    total = away_up_2 + home_down_2
    assert abs(total - 1.0) < 0.02, f"Symmetric check: {away_up_2} + {home_down_2} = {total}"


if __name__ == "__main__":
    tests = [
        test_step1_basic_call,
        test_step1_all_base_states,
        test_step1_all_out_counts,
        test_step1_all_innings,
        test_step1_both_halves,
        test_step1_score_differentials,
        test_step1_case_insensitive_half,
        test_step1_invalid_inning,
        test_step1_invalid_half,
        test_step1_invalid_outs,
        test_step2_default_away,
        test_step2_explicit_home,
        test_step2_explicit_away,
        test_step2_home_away_complementary,
        test_step3_wp_present,
        test_step3_wp_range,
        test_step3_wp_tied_game_near_50,
        test_step3_wp_increases_with_lead,
        test_step3_wp_late_inning_lead_higher,
        test_step3_wp_home_advantage,
        test_step3_large_lead_high_wp,
        test_step3_large_deficit_low_wp,
        test_step3_runners_on_affect_wp,
        test_step4_li_present,
        test_step4_li_positive,
        test_step4_li_average_near_1,
        test_step4_li_high_leverage_situation,
        test_step4_li_blowout_low,
        test_step5_wp_if_run_present,
        test_step5_wp_if_run_range,
        test_step5_wp_if_run_higher_than_current,
        test_step5_wp_if_run_effect_stronger_in_close_games,
        test_step6_wp_if_scoreless_present,
        test_step6_wp_if_scoreless_range,
        test_step6_wp_if_scoreless_lower_for_batting_team,
        test_step6_run_vs_scoreless_makes_sense,
        test_step7_deterministic,
        test_step7_distinct_states_different_wp,
        test_step7_extra_innings,
        test_step7_extreme_differential_clamp,
        test_response_structure,
        test_game_state_echo,
        test_json_parseable,
        test_wp_symmetric_score,
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
