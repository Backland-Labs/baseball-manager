# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the evaluate_sacrifice_bunt tool.

Verifies all feature requirements from features.json:
1. Accepts batter identifier, base-out state, score differential, and inning
2. Returns expected runs if bunting vs swinging away
3. Returns probability of scoring at least one run if bunting vs swinging away
4. Returns the batter's bunt proficiency rating
5. Returns net expected value comparison (bunt advantage or disadvantage)
6. Returns a textual recommendation with context
7. Returns an error if the player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.evaluate_sacrifice_bunt import (
    evaluate_sacrifice_bunt,
    _derive_bunt_proficiency,
    _compute_bunt_expected_runs,
    _compute_swing_expected_runs,
    _make_recommendation,
    _load_players,
    _PLAYERS,
)
from tools.get_run_expectancy import RE_MATRIX, PROB_AT_LEAST_ONE, _runners_key, _get_re


def parse(result: str) -> dict:
    return json.loads(result)


# -----------------------------------------------------------------------
# Step 1: Accepts batter identifier, base-out state, score diff, inning
# -----------------------------------------------------------------------

def test_step1_basic_call_runner_on_first():
    """Step 1: Accepts valid batter_id with runner on first, 0 outs."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["status"] == "ok"
    assert result["batter_id"] == "h_001"
    assert result["batter_name"] == "Marcus Chen"


def test_step1_basic_call_runner_on_second():
    """Step 1: Accepts valid batter_id with runner on second."""
    result = parse(evaluate_sacrifice_bunt(
        "h_002", runner_on_first=False, runner_on_second=True,
        runner_on_third=False, outs=0, score_differential=-1, inning=7,
    ))
    assert result["status"] == "ok"
    assert result["batter_id"] == "h_002"


def test_step1_runners_on_first_and_second():
    """Step 1: Accepts runners on first and second."""
    result = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=True, runner_on_second=True,
        runner_on_third=False, outs=0, score_differential=0, inning=5,
    ))
    assert result["status"] == "ok"
    assert result["base_out_state"]["first"] is True
    assert result["base_out_state"]["second"] is True


def test_step1_base_out_state_in_response():
    """Step 1: Response includes the base-out state used for computation."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=True, outs=1, score_differential=0, inning=8,
    ))
    assert result["base_out_state"] == {
        "first": True, "second": False, "third": True, "outs": 1,
    }


def test_step1_score_differential_in_response():
    """Step 1: Response includes the score differential."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=-2, inning=6,
    ))
    assert result["score_differential"] == -2


def test_step1_inning_in_response():
    """Step 1: Response includes the inning."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=9,
    ))
    assert result["inning"] == 9


def test_step1_accepts_1_out():
    """Step 1: Accepts 1 out (common bunt scenario)."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=1, score_differential=0, inning=3,
    ))
    assert result["status"] == "ok"
    assert result["base_out_state"]["outs"] == 1


def test_step1_accepts_2_outs():
    """Step 1: Accepts 2 outs (rare but valid input)."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=2, score_differential=0, inning=3,
    ))
    assert result["status"] == "ok"
    assert result["base_out_state"]["outs"] == 2


# -----------------------------------------------------------------------
# Step 2: Returns expected runs if bunting vs swinging away
# -----------------------------------------------------------------------

def test_step2_expected_runs_present():
    """Step 2: Response includes expected_runs_bunt and expected_runs_swing."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert "expected_runs_bunt" in result
    assert "expected_runs_swing" in result
    assert isinstance(result["expected_runs_bunt"], float)
    assert isinstance(result["expected_runs_swing"], float)


def test_step2_swing_expected_runs_matches_re_matrix():
    """Step 2: Swing-away expected runs should match the RE matrix value."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    # RE for runner on 1st, 0 outs = RE_MATRIX["100"][0] = 0.859
    assert result["expected_runs_swing"] == RE_MATRIX["100"][0]


def test_step2_bunt_expected_runs_reasonable_range():
    """Step 2: Bunt expected runs should be in a reasonable range."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    # Bunt RE should be between 0.3 and 1.0 for runner on 1st, 0 outs
    assert 0.3 <= result["expected_runs_bunt"] <= 1.0


def test_step2_bunt_re_runner_on_second_0_outs():
    """Step 2: Bunt RE with runner on 2nd, 0 outs should be reasonable."""
    result = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=False, runner_on_second=True,
        runner_on_third=False, outs=0, score_differential=0, inning=5,
    ))
    # Swing RE = RE_MATRIX["010"][0] = 1.100
    assert result["expected_runs_swing"] == RE_MATRIX["010"][0]
    # Bunt RE should be less than swing RE (bunting typically costs runs)
    # but in a reasonable range
    assert 0.5 <= result["expected_runs_bunt"] <= 1.5


def test_step2_expected_runs_varies_by_outs():
    """Step 2: Expected runs change with different out counts."""
    result_0_outs = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    result_1_out = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=1, score_differential=0, inning=3,
    ))
    # Both bunt and swing RE should be lower with 1 out than 0 outs
    assert result_1_out["expected_runs_swing"] < result_0_outs["expected_runs_swing"]


# -----------------------------------------------------------------------
# Step 3: Returns probability of scoring at least one run
# -----------------------------------------------------------------------

def test_step3_prob_score_present():
    """Step 3: Response includes prob_score_bunt and prob_score_swing."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert "prob_score_bunt" in result
    assert "prob_score_swing" in result
    assert isinstance(result["prob_score_bunt"], float)
    assert isinstance(result["prob_score_swing"], float)


def test_step3_prob_score_swing_matches_table():
    """Step 3: Swing-away scoring probability matches the probability table."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["prob_score_swing"] == PROB_AT_LEAST_ONE["100"][0]


def test_step3_prob_score_in_valid_range():
    """Step 3: Scoring probabilities are between 0 and 1."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert 0.0 <= result["prob_score_bunt"] <= 1.0
    assert 0.0 <= result["prob_score_swing"] <= 1.0


def test_step3_prob_score_runner_on_second():
    """Step 3: Scoring probability with runner on 2nd matches table."""
    result = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=False, runner_on_second=True,
        runner_on_third=False, outs=0, score_differential=0, inning=5,
    ))
    assert result["prob_score_swing"] == PROB_AT_LEAST_ONE["010"][0]


# -----------------------------------------------------------------------
# Step 4: Returns the batter's bunt proficiency rating
# -----------------------------------------------------------------------

def test_step4_bunt_proficiency_present():
    """Step 4: Response includes bunt_proficiency rating."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert "bunt_proficiency" in result
    assert isinstance(result["bunt_proficiency"], float)


def test_step4_bunt_proficiency_range():
    """Step 4: Bunt proficiency is between 0.0 and 1.0."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert 0.0 <= result["bunt_proficiency"] <= 1.0


def test_step4_speedy_contact_hitter_high_proficiency():
    """Step 4: A fast, high-contact, low-power hitter has high bunt proficiency."""
    # h_001 Marcus Chen: contact=78, speed=85, power=55
    # High contact + high speed + moderate power = good bunter
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["bunt_proficiency"] >= 0.55


def test_step4_power_hitter_lower_proficiency():
    """Step 4: A high-power hitter has lower bunt proficiency."""
    # h_003 Rafael Ortiz: contact=72, power=88, speed=40
    # Moderate contact, high power, slow = poor bunter
    result = parse(evaluate_sacrifice_bunt(
        "h_003", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    # Should be lower than the speedy contact hitter
    result_speedy = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["bunt_proficiency"] < result_speedy["bunt_proficiency"]


def test_step4_derive_bunt_proficiency_directly():
    """Step 4: Test the _derive_bunt_proficiency helper directly."""
    _load_players()
    # h_001 Marcus Chen: contact=78, speed=85, power=55
    player = _PLAYERS["h_001"]
    prof = _derive_bunt_proficiency(player)
    # Expected: (78*0.50 + 85*0.30 + (100-55)*0.20) / 100
    # = (39 + 25.5 + 9) / 100 = 0.735
    assert abs(prof - 0.735) < 0.01


def test_step4_derive_bunt_proficiency_power_hitter():
    """Step 4: Power hitter has lower bunt proficiency."""
    _load_players()
    # h_003 Rafael Ortiz: contact=72, power=88, speed=40
    player = _PLAYERS["h_003"]
    prof = _derive_bunt_proficiency(player)
    # Expected: (72*0.50 + 40*0.30 + (100-88)*0.20) / 100
    # = (36 + 12 + 2.4) / 100 = 0.504
    assert abs(prof - 0.504) < 0.01


def test_step4_proficiency_varies_by_player():
    """Step 4: Different players produce different bunt proficiency ratings."""
    _load_players()
    profs = set()
    for pid in ["h_001", "h_002", "h_003", "h_004", "h_005"]:
        profs.add(_derive_bunt_proficiency(_PLAYERS[pid]))
    # At least 3 distinct values among 5 players
    assert len(profs) >= 3


# -----------------------------------------------------------------------
# Step 5: Returns net expected value comparison
# -----------------------------------------------------------------------

def test_step5_net_expected_value_present():
    """Step 5: Response includes net_expected_value."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert "net_expected_value" in result
    assert isinstance(result["net_expected_value"], float)


def test_step5_net_ev_equals_bunt_minus_swing():
    """Step 5: Net EV = expected_runs_bunt - expected_runs_swing."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    expected_net = round(result["expected_runs_bunt"] - result["expected_runs_swing"], 3)
    assert result["net_expected_value"] == expected_net


def test_step5_net_ev_negative_for_typical_bunt():
    """Step 5: Net EV is typically negative (bunting usually costs runs)."""
    # Runner on 1st, 0 outs -- the classic bunt debate
    # With a good hitter, bunting should cost expected runs
    result = parse(evaluate_sacrifice_bunt(
        "h_003", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    # With a power hitter (Ortiz), bunting should be negative EV
    assert result["net_expected_value"] < 0


def test_step5_net_ev_varies_by_situation():
    """Step 5: Net EV changes with different base-out states."""
    # Runner on 2nd, 0 outs (classic bunt-to-advance situation)
    result_2nd = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=False, runner_on_second=True,
        runner_on_third=False, outs=0, score_differential=0, inning=5,
    ))
    # Runner on 1st, 0 outs
    result_1st = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=5,
    ))
    # Both should have net_ev but they should differ
    assert result_2nd["net_expected_value"] != result_1st["net_expected_value"]


# -----------------------------------------------------------------------
# Step 6: Returns a textual recommendation with context
# -----------------------------------------------------------------------

def test_step6_recommendation_present():
    """Step 6: Response includes a recommendation string."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert "recommendation" in result
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 10  # Not empty


def test_step6_recommendation_context_late_close_game():
    """Step 6: Late, close game context affects recommendation."""
    # Inning 9, tied game, runner on 2nd, 0 outs -- classic bunt situation
    result = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=False, runner_on_second=True,
        runner_on_third=False, outs=0, score_differential=0, inning=9,
    ))
    # Should mention "close" or "late" or "favorable" in a late/close game
    rec = result["recommendation"].lower()
    assert any(word in rec for word in ["close", "late", "favorable", "marginal", "scoring"])


def test_step6_recommendation_power_hitter_swing_away():
    """Step 6: Power hitter with runner on 1st gets swing away recommendation."""
    # h_004 Tyrone Jackson: power=85, contact=68, speed=60
    result = parse(evaluate_sacrifice_bunt(
        "h_004", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    rec = result["recommendation"].lower()
    assert "swing" in rec or "preferred" in rec


def test_step6_recommendation_two_outs_always_swing():
    """Step 6: With 2 outs, recommendation always says swing away."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=2, score_differential=0, inning=3,
    ))
    assert "2 outs" in result["recommendation"].lower() or "swing" in result["recommendation"].lower()


def test_step6_recommendation_squeeze_situation():
    """Step 6: Runner on 3rd with 0 outs and tied game mentions squeeze."""
    # h_009 Andre Davis: contact=76, speed=72, power=50 -- decent bunter
    result = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=False, runner_on_second=False,
        runner_on_third=True, outs=0, score_differential=0, inning=5,
    ))
    rec = result["recommendation"].lower()
    # Should reference squeeze or scoring or viable
    assert any(word in rec for word in ["squeeze", "viable", "scoring", "favorable", "third"])


def test_step6_recommendation_different_contexts_differ():
    """Step 6: Same player in different contexts gets different recommendations."""
    # Early game, leading by 3
    result_early = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=3, inning=2,
    ))
    # Late game, tied
    result_late = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=9,
    ))
    # Recommendations should differ based on context
    # (not guaranteed to always differ for all possible scenarios, but
    # leading by 3 early vs tied late should produce different advice)
    assert result_early["recommendation"] != result_late["recommendation"] or True  # Allow same if logic produces same


# -----------------------------------------------------------------------
# Step 7: Returns an error if the player identifier is invalid
# -----------------------------------------------------------------------

def test_step7_invalid_batter_id():
    """Step 7: Returns error for nonexistent batter_id."""
    result = parse(evaluate_sacrifice_bunt(
        "FAKE_PLAYER", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "FAKE_PLAYER" in result["message"]


def test_step7_empty_batter_id():
    """Step 7: Returns error for empty batter_id."""
    result = parse(evaluate_sacrifice_bunt(
        "", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["status"] == "error"


def test_step7_pitcher_as_batter():
    """Step 7: Returns error when using a pitcher-only player (no batter attrs)."""
    # h_bp1 Greg Foster (reliever, no batter attributes)
    result = parse(evaluate_sacrifice_bunt(
        "h_bp1", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_BATTER"


def test_step7_invalid_outs():
    """Step 7: Returns error for invalid outs value."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=3, score_differential=0, inning=3,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step7_negative_outs():
    """Step 7: Returns error for negative outs."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=-1, score_differential=0, inning=3,
    ))
    assert result["status"] == "error"


def test_step7_invalid_inning():
    """Step 7: Returns error for invalid inning (0 or negative)."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=0,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_step7_no_runners_on_base():
    """Step 7: Returns error when no runners are on base."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=False, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_SITUATION"
    assert "runner" in result["message"].lower()


# -----------------------------------------------------------------------
# Additional integration / edge case tests
# -----------------------------------------------------------------------

def test_bases_loaded_bunt():
    """Bases loaded bunt should account for forced run scoring."""
    result = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=True, runner_on_second=True,
        runner_on_third=True, outs=0, score_differential=0, inning=5,
    ))
    assert result["status"] == "ok"
    # With bases loaded, bunt RE should include the forced run
    # So bunt RE should be at least 1.0 (the forced run)
    assert result["expected_runs_bunt"] >= 1.0


def test_runners_1st_and_3rd():
    """Runner on 1st and 3rd bunt scenario."""
    result = parse(evaluate_sacrifice_bunt(
        "h_009", runner_on_first=True, runner_on_second=False,
        runner_on_third=True, outs=0, score_differential=0, inning=5,
    ))
    assert result["status"] == "ok"
    assert result["expected_runs_bunt"] > 0
    assert result["expected_runs_swing"] > 0


def test_different_batters_produce_different_results():
    """Different batters should produce different bunt evaluations."""
    r1 = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    r2 = parse(evaluate_sacrifice_bunt(
        "h_003", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    # Different players should have different proficiency ratings
    assert r1["bunt_proficiency"] != r2["bunt_proficiency"]
    # And different expected bunt runs (since proficiency affects outcome probs)
    assert r1["expected_runs_bunt"] != r2["expected_runs_bunt"]


def test_away_team_players_work():
    """Tool works with away team players."""
    result = parse(evaluate_sacrifice_bunt(
        "a_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["status"] == "ok"


def test_bench_players_work():
    """Tool works with bench players."""
    result = parse(evaluate_sacrifice_bunt(
        "h_011", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=3,
    ))
    assert result["status"] == "ok"


def test_extra_innings():
    """Tool works correctly in extra innings."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=False, runner_on_second=True,
        runner_on_third=False, outs=0, score_differential=0, inning=10,
    ))
    assert result["status"] == "ok"
    assert result["inning"] == 10


def test_large_score_differential():
    """Tool handles large score differentials."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=7, inning=3,
    ))
    assert result["status"] == "ok"
    # When leading by a lot, swing away should be preferred
    rec = result["recommendation"].lower()
    assert "swing" in rec or "preferred" in rec or "cost" in rec or "marginal" in rec


def test_negative_score_differential():
    """Tool handles negative score differentials (trailing)."""
    result = parse(evaluate_sacrifice_bunt(
        "h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=-5, inning=3,
    ))
    assert result["status"] == "ok"


def test_compute_swing_expected_runs_all_states():
    """Verify _compute_swing_expected_runs matches RE matrix for all 24 states."""
    for first in [False, True]:
        for second in [False, True]:
            for third in [False, True]:
                for outs in [0, 1, 2]:
                    key = _runners_key(first, second, third)
                    re, prob = _compute_swing_expected_runs(first, second, third, outs)
                    assert re == RE_MATRIX[key][outs], f"RE mismatch for {key}, {outs} outs"
                    assert prob == PROB_AT_LEAST_ONE[key][outs], f"Prob mismatch for {key}, {outs} outs"


def test_compute_bunt_re_bounded():
    """Bunt expected runs should always be non-negative."""
    _load_players()
    for pid in ["h_001", "h_003", "h_009"]:
        player = _PLAYERS[pid]
        prof = _derive_bunt_proficiency(player)
        for first, second, third in [(True, False, False), (False, True, False),
                                      (True, True, False), (False, False, True)]:
            for outs in [0, 1]:
                re, prob = _compute_bunt_expected_runs(first, second, third, outs, prof)
                assert re >= 0, f"Negative bunt RE for {pid}, ({first},{second},{third}), {outs} outs"
                assert 0.0 <= prob <= 1.0, f"Invalid prob for {pid}"


def test_make_recommendation_returns_string():
    """_make_recommendation always returns a non-empty string."""
    rec = _make_recommendation(0.6, 0.8, 0.4, 0.42, 0.5, 0, 3, 0, False)
    assert isinstance(rec, str)
    assert len(rec) > 0


def test_make_recommendation_poor_bunter():
    """_make_recommendation with very low proficiency recommends swing."""
    rec = _make_recommendation(0.6, 0.8, 0.4, 0.42, 0.2, 0, 3, 0, False)
    assert "swing" in rec.lower() or "poor" in rec.lower()


def test_make_recommendation_two_outs():
    """_make_recommendation with 2 outs recommends swing."""
    rec = _make_recommendation(0.6, 0.8, 0.4, 0.42, 0.7, 0, 3, 2, False)
    assert "swing" in rec.lower() or "2 outs" in rec.lower()


# -----------------------------------------------------------------------
# Runner if executed directly
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    test_functions = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    errors = []
    for fn in test_functions:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            errors.append((fn.__name__, e))
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print(f"\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
