# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the input_validation_layer feature.

Verifies all feature requirements from features.json:
1. Each tool has a Pydantic input model with required and optional parameters
2. Parameters are type-checked before processing
3. Enum parameters (vs_hand: L/R, recency_window) reject invalid values
4. Player identifiers are validated as valid MLB player IDs
5. Game state inputs are validated against MatchupState/RosterState/OpponentRosterState schemas
6. Validation errors include which parameter failed and what was expected
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from tools.validation import (
    GetBatterStatsInput,
    GetPitcherStatsInput,
    GetMatchupDataInput,
    GetRunExpectancyInput,
    GetWinProbabilityInput,
    EvaluateStolenBaseInput,
    EvaluateSacrificeBuntInput,
    GetBullpenStatusInput,
    GetPitcherFatigueAssessmentInput,
    GetDefensivePositioningInput,
    GetDefensiveReplacementValueInput,
    GetPlatoonComparisonInput,
    TOOL_INPUT_MODELS,
    validate_tool_input,
    validate_game_state,
    is_valid_player_id,
    ValidationErrorDetail,
)

from tools.get_batter_stats import get_batter_stats
from tools.get_pitcher_stats import get_pitcher_stats
from tools.get_matchup_data import get_matchup_data
from tools.get_run_expectancy import get_run_expectancy
from tools.get_win_probability import get_win_probability
from tools.evaluate_stolen_base import evaluate_stolen_base
from tools.evaluate_sacrifice_bunt import evaluate_sacrifice_bunt
from tools.get_bullpen_status import get_bullpen_status
from tools.get_pitcher_fatigue_assessment import get_pitcher_fatigue_assessment
from tools.get_defensive_positioning import get_defensive_positioning
from tools.get_defensive_replacement_value import get_defensive_replacement_value
from tools.get_platoon_comparison import get_platoon_comparison


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def raw_parse(result: str) -> dict:
    return json.loads(result)


# -----------------------------------------------------------------------
# Step 1: Each tool has a Pydantic input model with required and optional
# parameters
# -----------------------------------------------------------------------

def test_step1_all_tools_have_input_models():
    """Step 1a: Every tool has a corresponding Pydantic input model."""
    expected_tools = {
        "get_batter_stats", "get_pitcher_stats", "get_matchup_data",
        "get_run_expectancy", "get_win_probability",
        "evaluate_stolen_base", "evaluate_sacrifice_bunt",
        "get_bullpen_status", "get_pitcher_fatigue_assessment",
        "get_defensive_positioning", "get_defensive_replacement_value",
        "get_platoon_comparison",
    }
    assert set(TOOL_INPUT_MODELS.keys()) == expected_tools, \
        f"Missing models for: {expected_tools - set(TOOL_INPUT_MODELS.keys())}"


def test_step1_models_are_pydantic():
    """Step 1b: All input models are Pydantic BaseModel subclasses."""
    from pydantic import BaseModel
    for tool_name, model_cls in TOOL_INPUT_MODELS.items():
        assert issubclass(model_cls, BaseModel), \
            f"{tool_name}: model {model_cls.__name__} is not a Pydantic BaseModel"


def test_step1_batter_stats_has_required_and_optional():
    """Step 1c: GetBatterStatsInput has required player_id and optional splits."""
    # Required field -- should fail without player_id
    try:
        GetBatterStatsInput()
        assert False, "Should have raised ValidationError for missing player_id"
    except ValidationError:
        pass

    # With only required field
    m = GetBatterStatsInput(player_id="h_001")
    assert m.player_id == "h_001"
    assert m.vs_hand is None
    assert m.home_away is None
    assert m.recency_window is None


def test_step1_pitcher_stats_has_required_and_optional():
    """Step 1d: GetPitcherStatsInput has required player_id and optional splits."""
    try:
        GetPitcherStatsInput()
        assert False, "Should have raised ValidationError for missing player_id"
    except ValidationError:
        pass

    m = GetPitcherStatsInput(player_id="h_sp1")
    assert m.player_id == "h_sp1"
    assert m.vs_hand is None


def test_step1_matchup_data_requires_both_ids():
    """Step 1e: GetMatchupDataInput requires both batter_id and pitcher_id."""
    try:
        GetMatchupDataInput(batter_id="h_001")
        assert False, "Should have raised ValidationError for missing pitcher_id"
    except ValidationError:
        pass

    try:
        GetMatchupDataInput(pitcher_id="a_sp1")
        assert False, "Should have raised ValidationError for missing batter_id"
    except ValidationError:
        pass

    m = GetMatchupDataInput(batter_id="h_001", pitcher_id="a_sp1")
    assert m.batter_id == "h_001"
    assert m.pitcher_id == "a_sp1"


def test_step1_run_expectancy_all_required():
    """Step 1f: GetRunExpectancyInput requires all four parameters."""
    try:
        GetRunExpectancyInput(runner_on_first=True)
        assert False, "Should fail without all required fields"
    except ValidationError:
        pass

    m = GetRunExpectancyInput(
        runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=1
    )
    assert m.outs == 1


def test_step1_win_probability_required_and_optional():
    """Step 1g: GetWinProbabilityInput has required game state and optional managed_team_home."""
    m = GetWinProbabilityInput(
        inning=5, half="TOP", outs=1,
        runner_on_first=True, runner_on_second=False, runner_on_third=False,
        score_differential=2,
    )
    assert m.managed_team_home is None

    m2 = GetWinProbabilityInput(
        inning=5, half="BOTTOM", outs=0,
        runner_on_first=False, runner_on_second=False, runner_on_third=False,
        score_differential=-1, managed_team_home=True,
    )
    assert m2.managed_team_home is True


def test_step1_stolen_base_required_and_defaults():
    """Step 1h: EvaluateStolenBaseInput has required IDs and default runners/outs."""
    m = EvaluateStolenBaseInput(
        runner_id="h_001", target_base=2,
        pitcher_id="a_sp1", catcher_id="a_008",
    )
    assert m.runner_on_first is False
    assert m.outs == 0


def test_step1_sacrifice_bunt_all_required():
    """Step 1i: EvaluateSacrificeBuntInput requires all fields."""
    try:
        EvaluateSacrificeBuntInput(batter_id="h_001")
        assert False, "Should fail without all required fields"
    except ValidationError:
        pass

    m = EvaluateSacrificeBuntInput(
        batter_id="h_001", runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=0, score_differential=0, inning=5,
    )
    assert m.inning == 5


def test_step1_bullpen_status_all_optional():
    """Step 1j: GetBullpenStatusInput has all-optional parameters with defaults."""
    m = GetBullpenStatusInput()
    assert m.team.value == "home"
    assert m.used_pitcher_ids is None


def test_step1_fatigue_assessment_required_and_defaults():
    """Step 1k: GetPitcherFatigueAssessmentInput requires pitcher_id, rest have defaults."""
    m = GetPitcherFatigueAssessmentInput(pitcher_id="h_sp1")
    assert m.pitch_count == 0
    assert m.innings_pitched == 0.0
    assert m.times_through_order == 1
    assert m.runs_allowed == 0
    assert m.in_current_game is True


def test_step1_defensive_positioning_all_required():
    """Step 1l: GetDefensivePositioningInput requires all fields."""
    m = GetDefensivePositioningInput(
        batter_id="h_001", pitcher_id="a_sp1", outs=0,
        runner_on_first=True, runner_on_second=False, runner_on_third=False,
        score_differential=0, inning=5,
    )
    assert m.batter_id == "h_001"


def test_step1_replacement_value_required_and_defaults():
    """Step 1m: GetDefensiveReplacementValueInput requires IDs and position, rest have defaults."""
    m = GetDefensiveReplacementValueInput(
        current_fielder_id="h_002", replacement_id="h_011", position="SS",
    )
    assert m.inning == 7
    assert m.half == "top"
    assert m.outs == 0


def test_step1_platoon_comparison_all_required():
    """Step 1n: GetPlatoonComparisonInput requires all three player IDs."""
    try:
        GetPlatoonComparisonInput(current_batter_id="h_004", pinch_hitter_id="h_012")
        assert False, "Should fail without pitcher_id"
    except ValidationError:
        pass

    m = GetPlatoonComparisonInput(
        current_batter_id="h_004", pinch_hitter_id="h_012", pitcher_id="a_sp1",
    )
    assert m.pitcher_id == "a_sp1"


# -----------------------------------------------------------------------
# Step 2: Parameters are type-checked before processing
# -----------------------------------------------------------------------

def test_step2_outs_must_be_int():
    """Step 2a: outs parameter rejects non-integer values."""
    try:
        GetRunExpectancyInput(
            runner_on_first=True, runner_on_second=False,
            runner_on_third=False, outs="two"
        )
        assert False, "Should reject string for outs"
    except ValidationError:
        pass


def test_step2_outs_range_enforced():
    """Step 2b: outs must be 0, 1, or 2."""
    try:
        GetRunExpectancyInput(
            runner_on_first=True, runner_on_second=False,
            runner_on_third=False, outs=3
        )
        assert False, "Should reject outs=3"
    except ValidationError:
        pass

    try:
        GetRunExpectancyInput(
            runner_on_first=True, runner_on_second=False,
            runner_on_third=False, outs=-1
        )
        assert False, "Should reject outs=-1"
    except ValidationError:
        pass


def test_step2_inning_must_be_positive():
    """Step 2c: inning must be >= 1."""
    try:
        GetWinProbabilityInput(
            inning=0, half="TOP", outs=0,
            runner_on_first=False, runner_on_second=False, runner_on_third=False,
            score_differential=0,
        )
        assert False, "Should reject inning=0"
    except ValidationError:
        pass


def test_step2_pitch_count_non_negative():
    """Step 2d: pitch_count must be >= 0."""
    try:
        GetPitcherFatigueAssessmentInput(pitcher_id="h_sp1", pitch_count=-5)
        assert False, "Should reject negative pitch count"
    except ValidationError:
        pass


def test_step2_innings_pitched_non_negative():
    """Step 2e: innings_pitched must be >= 0."""
    try:
        GetPitcherFatigueAssessmentInput(pitcher_id="h_sp1", innings_pitched=-1.0)
        assert False, "Should reject negative innings pitched"
    except ValidationError:
        pass


def test_step2_times_through_order_minimum():
    """Step 2f: times_through_order must be >= 1."""
    try:
        GetPitcherFatigueAssessmentInput(pitcher_id="h_sp1", times_through_order=0)
        assert False, "Should reject times_through_order=0"
    except ValidationError:
        pass


def test_step2_target_base_range():
    """Step 2g: target_base must be 2, 3, or 4."""
    try:
        EvaluateStolenBaseInput(
            runner_id="h_001", target_base=1,
            pitcher_id="a_sp1", catcher_id="a_008",
        )
        assert False, "Should reject target_base=1"
    except ValidationError:
        pass

    try:
        EvaluateStolenBaseInput(
            runner_id="h_001", target_base=5,
            pitcher_id="a_sp1", catcher_id="a_008",
        )
        assert False, "Should reject target_base=5"
    except ValidationError:
        pass


def test_step2_runner_booleans_type_checked():
    """Step 2h: Runner boolean parameters are type-checked."""
    # Pydantic coerces truthy values to bool, but non-bool-like values should fail
    try:
        GetRunExpectancyInput(
            runner_on_first="not_a_bool", runner_on_second=False,
            runner_on_third=False, outs=0,
        )
        assert False, "Should reject non-boolean-like string for boolean"
    except ValidationError:
        pass


def test_step2_score_differential_is_int():
    """Step 2i: score_differential must be an integer."""
    try:
        GetWinProbabilityInput(
            inning=5, half="TOP", outs=1,
            runner_on_first=False, runner_on_second=False, runner_on_third=False,
            score_differential="two",
        )
        assert False, "Should reject string for score_differential"
    except ValidationError:
        pass


def test_step2_valid_types_pass():
    """Step 2j: Valid types pass validation without errors."""
    m = GetRunExpectancyInput(
        runner_on_first=True, runner_on_second=False,
        runner_on_third=False, outs=2,
    )
    assert m.outs == 2
    assert m.runner_on_first is True


# -----------------------------------------------------------------------
# Step 3: Enum parameters reject invalid values
# -----------------------------------------------------------------------

def test_step3_vs_hand_enum_rejects_invalid():
    """Step 3a: vs_hand only accepts 'L' or 'R'."""
    try:
        GetBatterStatsInput(player_id="h_001", vs_hand="X")
        assert False, "Should reject vs_hand='X'"
    except ValidationError:
        pass

    try:
        GetBatterStatsInput(player_id="h_001", vs_hand="left")
        assert False, "Should reject vs_hand='left'"
    except ValidationError:
        pass


def test_step3_vs_hand_enum_accepts_valid():
    """Step 3b: vs_hand accepts 'L' and 'R'."""
    m = GetBatterStatsInput(player_id="h_001", vs_hand="L")
    assert m.vs_hand.value == "L"

    m = GetBatterStatsInput(player_id="h_001", vs_hand="R")
    assert m.vs_hand.value == "R"


def test_step3_home_away_enum_rejects_invalid():
    """Step 3c: home_away only accepts 'home' or 'away'."""
    try:
        GetBatterStatsInput(player_id="h_001", home_away="neutral")
        assert False, "Should reject home_away='neutral'"
    except ValidationError:
        pass


def test_step3_home_away_enum_accepts_valid():
    """Step 3d: home_away accepts 'home' and 'away'."""
    m = GetBatterStatsInput(player_id="h_001", home_away="home")
    assert m.home_away.value == "home"

    m = GetBatterStatsInput(player_id="h_001", home_away="away")
    assert m.home_away.value == "away"


def test_step3_recency_window_enum_rejects_invalid():
    """Step 3e: recency_window only accepts the four valid values."""
    try:
        GetBatterStatsInput(player_id="h_001", recency_window="last_3")
        assert False, "Should reject recency_window='last_3'"
    except ValidationError:
        pass


def test_step3_recency_window_enum_accepts_valid():
    """Step 3f: recency_window accepts all four valid values."""
    for window in ("last_7", "last_14", "last_30", "season"):
        m = GetBatterStatsInput(player_id="h_001", recency_window=window)
        assert m.recency_window.value == window


def test_step3_half_inning_rejects_invalid():
    """Step 3g: half parameter rejects invalid values."""
    try:
        GetWinProbabilityInput(
            inning=5, half="MIDDLE", outs=1,
            runner_on_first=False, runner_on_second=False, runner_on_third=False,
            score_differential=0,
        )
        assert False, "Should reject half='MIDDLE'"
    except ValidationError:
        pass


def test_step3_half_inning_accepts_valid():
    """Step 3h: half parameter accepts 'TOP' and 'BOTTOM'."""
    m = GetWinProbabilityInput(
        inning=5, half="TOP", outs=1,
        runner_on_first=False, runner_on_second=False, runner_on_third=False,
        score_differential=0,
    )
    assert m.half.upper() == "TOP"


def test_step3_team_enum_rejects_invalid():
    """Step 3i: team parameter rejects invalid values."""
    try:
        GetBullpenStatusInput(team="both")
        assert False, "Should reject team='both'"
    except ValidationError:
        pass


def test_step3_position_enum_rejects_invalid():
    """Step 3j: position parameter rejects invalid positions."""
    try:
        GetDefensiveReplacementValueInput(
            current_fielder_id="h_002", replacement_id="h_011", position="DH2",
        )
        assert False, "Should reject position='DH2'"
    except ValidationError:
        pass


def test_step3_position_enum_accepts_valid():
    """Step 3k: position parameter accepts all valid fielding positions."""
    for pos in ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"):
        m = GetDefensiveReplacementValueInput(
            current_fielder_id="h_002", replacement_id="h_011", position=pos,
        )
        assert m.position.value == pos


def test_step3_pitcher_stats_enums_consistent_with_batter():
    """Step 3l: GetPitcherStatsInput uses the same enum types as GetBatterStatsInput."""
    try:
        GetPitcherStatsInput(player_id="h_sp1", vs_hand="X")
        assert False, "Should reject vs_hand='X'"
    except ValidationError:
        pass

    try:
        GetPitcherStatsInput(player_id="h_sp1", home_away="road")
        assert False, "Should reject home_away='road'"
    except ValidationError:
        pass

    try:
        GetPitcherStatsInput(player_id="h_sp1", recency_window="last_3")
        assert False, "Should reject invalid recency_window"
    except ValidationError:
        pass


# -----------------------------------------------------------------------
# Step 4: Player identifiers are validated as valid MLB player IDs
# -----------------------------------------------------------------------

def test_step4_empty_player_id_rejected():
    """Step 4a: Empty string is rejected as a player ID."""
    try:
        GetBatterStatsInput(player_id="")
        assert False, "Should reject empty player_id"
    except ValidationError:
        pass


def test_step4_whitespace_player_id_rejected():
    """Step 4b: Whitespace-only string is rejected as a player ID."""
    try:
        GetBatterStatsInput(player_id="   ")
        assert False, "Should reject whitespace-only player_id"
    except ValidationError:
        pass


def test_step4_valid_roster_id_accepted():
    """Step 4c: Standard roster IDs (h_001, a_sp1) are accepted."""
    m = GetBatterStatsInput(player_id="h_001")
    assert m.player_id == "h_001"

    m = GetBatterStatsInput(player_id="a_sp1")
    assert m.player_id == "a_sp1"


def test_step4_numeric_mlb_id_accepted():
    """Step 4d: Numeric MLB player IDs (e.g., '660271') are accepted."""
    m = GetBatterStatsInput(player_id="660271")
    assert m.player_id == "660271"


def test_step4_all_tools_validate_player_ids():
    """Step 4e: All tools that accept player IDs reject empty strings."""
    player_id_tools_and_fields = [
        (GetBatterStatsInput, {"player_id": ""}),
        (GetPitcherStatsInput, {"player_id": ""}),
        (GetMatchupDataInput, {"batter_id": "", "pitcher_id": "a_sp1"}),
        (GetMatchupDataInput, {"batter_id": "h_001", "pitcher_id": ""}),
        (EvaluateStolenBaseInput, {"runner_id": "", "target_base": 2, "pitcher_id": "a_sp1", "catcher_id": "a_008"}),
        (EvaluateSacrificeBuntInput, {"batter_id": "", "runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "outs": 0, "score_differential": 0, "inning": 5}),
        (GetPitcherFatigueAssessmentInput, {"pitcher_id": ""}),
        (GetDefensivePositioningInput, {"batter_id": "", "pitcher_id": "a_sp1", "outs": 0, "runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "score_differential": 0, "inning": 5}),
        (GetDefensiveReplacementValueInput, {"current_fielder_id": "", "replacement_id": "h_011", "position": "SS"}),
        (GetPlatoonComparisonInput, {"current_batter_id": "", "pinch_hitter_id": "h_012", "pitcher_id": "a_sp1"}),
    ]

    for model_cls, kwargs in player_id_tools_and_fields:
        try:
            model_cls(**kwargs)
            assert False, f"{model_cls.__name__}: should reject empty player ID with args {kwargs}"
        except ValidationError:
            pass


def test_step4_is_valid_player_id_helper():
    """Step 4f: The is_valid_player_id helper correctly validates IDs."""
    assert is_valid_player_id("h_001") is True
    assert is_valid_player_id("660271") is True
    assert is_valid_player_id("a_sp1") is True
    assert is_valid_player_id("") is False
    assert is_valid_player_id("   ") is False
    assert is_valid_player_id(123) is False  # type: ignore


def test_step4_tools_reject_invalid_player_ids_at_runtime():
    """Step 4g: Tools return INVALID_PLAYER_ID error for non-existent player IDs."""
    result = raw_parse(get_batter_stats("nonexistent_player"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"

    result = raw_parse(get_pitcher_stats("nonexistent_player"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"


# -----------------------------------------------------------------------
# Step 5: Game state inputs are validated against schemas
# -----------------------------------------------------------------------

def test_step5_valid_matchup_state_passes():
    """Step 5a: A valid MatchupState dict passes validation."""
    state = {
        "inning": 5,
        "half": "TOP",
        "outs": 1,
        "count": {"balls": 2, "strikes": 1},
        "runners": {},
        "score": {"home": 3, "away": 2},
        "batting_team": "AWAY",
        "batter": {"player_id": "h_001", "name": "Test Batter", "bats": "R", "lineup_position": 4},
        "pitcher": {"player_id": "a_sp1", "name": "Test Pitcher", "throws": "R"},
        "on_deck_batter": {"player_id": "h_002", "name": "Next Batter", "bats": "L"},
    }
    valid, error = validate_game_state(matchup_state=state)
    assert valid, f"Should pass validation: {error}"


def test_step5_invalid_inning_rejected():
    """Step 5b: MatchupState with inning=0 is rejected."""
    state = {
        "inning": 0,
        "half": "TOP",
        "outs": 1,
        "batting_team": "AWAY",
        "batter": {"player_id": "h_001", "name": "Test", "bats": "R", "lineup_position": 4},
        "pitcher": {"player_id": "a_sp1", "name": "Test", "throws": "R"},
        "on_deck_batter": {"player_id": "h_002", "name": "Test", "bats": "L"},
    }
    valid, error = validate_game_state(matchup_state=state)
    assert not valid
    assert "inning" in error.lower() or "MatchupState" in error


def test_step5_invalid_outs_rejected():
    """Step 5c: MatchupState with outs=3 is rejected."""
    state = {
        "inning": 5,
        "half": "TOP",
        "outs": 3,
        "batting_team": "AWAY",
        "batter": {"player_id": "h_001", "name": "Test", "bats": "R", "lineup_position": 4},
        "pitcher": {"player_id": "a_sp1", "name": "Test", "throws": "R"},
        "on_deck_batter": {"player_id": "h_002", "name": "Test", "bats": "L"},
    }
    valid, error = validate_game_state(matchup_state=state)
    assert not valid
    assert "outs" in error.lower() or "MatchupState" in error


def test_step5_invalid_half_rejected():
    """Step 5d: MatchupState with invalid half is rejected."""
    state = {
        "inning": 5,
        "half": "MIDDLE",
        "outs": 1,
        "batting_team": "AWAY",
        "batter": {"player_id": "h_001", "name": "Test", "bats": "R", "lineup_position": 4},
        "pitcher": {"player_id": "a_sp1", "name": "Test", "throws": "R"},
        "on_deck_batter": {"player_id": "h_002", "name": "Test", "bats": "L"},
    }
    valid, error = validate_game_state(matchup_state=state)
    assert not valid


def test_step5_valid_roster_state_passes():
    """Step 5e: A valid RosterState dict passes validation."""
    state = {
        "our_lineup": [
            {"player_id": f"h_{i:03d}", "name": f"Player {i}", "position": "1B", "bats": "R"}
            for i in range(1, 10)
        ],
        "our_lineup_position": 3,
        "bench": [],
        "bullpen": [],
        "mound_visits_remaining": 5,
        "challenge_available": True,
    }
    valid, error = validate_game_state(roster_state=state)
    assert valid, f"Should pass: {error}"


def test_step5_invalid_lineup_position_rejected():
    """Step 5f: RosterState with lineup_position > 8 is rejected."""
    state = {
        "our_lineup": [
            {"player_id": f"h_{i:03d}", "name": f"Player {i}", "position": "1B", "bats": "R"}
            for i in range(1, 10)
        ],
        "our_lineup_position": 9,
    }
    valid, error = validate_game_state(roster_state=state)
    assert not valid
    assert "our_lineup_position" in error or "RosterState" in error


def test_step5_valid_opponent_roster_passes():
    """Step 5g: A valid OpponentRosterState dict passes validation."""
    state = {
        "their_lineup": [
            {"player_id": f"a_{i:03d}", "name": f"Player {i}", "position": "CF", "bats": "L"}
            for i in range(1, 10)
        ],
        "their_lineup_position": 0,
    }
    valid, error = validate_game_state(opponent_roster_state=state)
    assert valid, f"Should pass: {error}"


def test_step5_missing_required_batter_field_rejected():
    """Step 5h: MatchupState with missing batter name is rejected."""
    state = {
        "inning": 5,
        "half": "TOP",
        "outs": 1,
        "batting_team": "AWAY",
        "batter": {"player_id": "h_001", "bats": "R", "lineup_position": 4},  # missing 'name'
        "pitcher": {"player_id": "a_sp1", "name": "Test", "throws": "R"},
        "on_deck_batter": {"player_id": "h_002", "name": "Test", "bats": "L"},
    }
    valid, error = validate_game_state(matchup_state=state)
    assert not valid
    assert "name" in error.lower() or "batter" in error.lower()


def test_step5_all_three_validated_together():
    """Step 5i: All three game state components can be validated together."""
    matchup = {
        "inning": 5,
        "half": "TOP",
        "outs": 1,
        "batting_team": "AWAY",
        "batter": {"player_id": "h_001", "name": "Test Batter", "bats": "R", "lineup_position": 4},
        "pitcher": {"player_id": "a_sp1", "name": "Test Pitcher", "throws": "R"},
        "on_deck_batter": {"player_id": "h_002", "name": "Next Batter", "bats": "L"},
    }
    roster = {
        "our_lineup": [
            {"player_id": f"h_{i:03d}", "name": f"Player {i}", "position": "1B", "bats": "R"}
            for i in range(1, 10)
        ],
        "our_lineup_position": 3,
    }
    opponent = {
        "their_lineup": [
            {"player_id": f"a_{i:03d}", "name": f"Player {i}", "position": "CF", "bats": "L"}
            for i in range(1, 10)
        ],
        "their_lineup_position": 0,
    }
    valid, error = validate_game_state(matchup, roster, opponent)
    assert valid, f"Should pass: {error}"


def test_step5_multiple_errors_reported():
    """Step 5j: When multiple game state components have errors, all are reported."""
    matchup = {"inning": 0, "half": "TOP", "outs": 1, "batting_team": "AWAY",
               "batter": {"player_id": "h_001", "name": "Test", "bats": "R", "lineup_position": 4},
               "pitcher": {"player_id": "a_sp1", "name": "Test", "throws": "R"},
               "on_deck_batter": {"player_id": "h_002", "name": "Test", "bats": "L"}}
    roster = {"our_lineup": [], "our_lineup_position": 10}
    valid, error = validate_game_state(matchup, roster)
    assert not valid
    assert "MatchupState" in error
    assert "RosterState" in error


# -----------------------------------------------------------------------
# Step 6: Validation errors include which parameter failed and what was expected
# -----------------------------------------------------------------------

def test_step6_validate_tool_input_reports_parameter_name():
    """Step 6a: validate_tool_input includes parameter name in error."""
    valid, error = validate_tool_input("get_run_expectancy",
                                       runner_on_first=True, runner_on_second=False,
                                       runner_on_third=False, outs=5)
    assert not valid
    assert "outs" in error.lower()


def test_step6_validate_tool_input_reports_expected():
    """Step 6b: validate_tool_input error message describes what was expected."""
    valid, error = validate_tool_input("get_batter_stats",
                                       player_id="h_001", vs_hand="X")
    assert not valid
    assert "vs_hand" in error.lower() or "hand" in error.lower()


def test_step6_validate_tool_input_success():
    """Step 6c: validate_tool_input returns (True, None) for valid inputs."""
    valid, error = validate_tool_input("get_batter_stats", player_id="h_001")
    assert valid
    assert error is None


def test_step6_validate_tool_input_unknown_tool():
    """Step 6d: validate_tool_input rejects unknown tool names."""
    valid, error = validate_tool_input("nonexistent_tool", foo="bar")
    assert not valid
    assert "unknown" in error.lower() or "Unknown" in error


def test_step6_validation_error_detail_formatting():
    """Step 6e: ValidationErrorDetail produces clear error messages."""
    detail = ValidationErrorDetail("outs", "integer 0-2", 5)
    assert "outs" in str(detail)
    assert "0-2" in str(detail)
    assert "5" in str(detail)

    d = detail.to_dict()
    assert d["parameter"] == "outs"
    assert d["expected"] == "integer 0-2"
    assert "5" in d["got"]


def test_step6_pydantic_error_includes_location():
    """Step 6f: Pydantic validation errors include field location."""
    try:
        GetRunExpectancyInput(
            runner_on_first=True, runner_on_second=False,
            runner_on_third=False, outs=10,
        )
        assert False, "Should have raised"
    except ValidationError as e:
        errors = e.errors()
        assert len(errors) > 0
        # Should indicate which field failed
        locs = [".".join(str(x) for x in err["loc"]) for err in errors]
        assert any("outs" in loc for loc in locs), f"Error should mention 'outs': {locs}"


def test_step6_tool_returns_error_with_parameter_info():
    """Step 6g: Tools include parameter information in error messages."""
    result = raw_parse(get_batter_stats("h_001", vs_hand="X"))
    assert result["status"] == "error"
    assert "vs_hand" in result["message"]
    assert "L" in result["message"] or "R" in result["message"]


def test_step6_tool_outs_error_includes_value():
    """Step 6h: Tool outs validation error mentions the invalid value."""
    result = raw_parse(get_run_expectancy(True, False, False, 5))
    assert result["status"] == "error"
    assert "5" in result["message"] or "outs" in result["message"].lower()


# -----------------------------------------------------------------------
# Additional: validate_tool_input works for all 12 tools
# -----------------------------------------------------------------------

def test_validate_tool_input_all_tools_valid():
    """All 12 tools pass validation with correct inputs via validate_tool_input."""
    valid_inputs = [
        ("get_batter_stats", {"player_id": "h_001"}),
        ("get_pitcher_stats", {"player_id": "h_sp1"}),
        ("get_matchup_data", {"batter_id": "h_001", "pitcher_id": "a_sp1"}),
        ("get_run_expectancy", {"runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "outs": 0}),
        ("get_win_probability", {"inning": 5, "half": "TOP", "outs": 1, "runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "score_differential": 2}),
        ("evaluate_stolen_base", {"runner_id": "h_001", "target_base": 2, "pitcher_id": "a_sp1", "catcher_id": "a_008"}),
        ("evaluate_sacrifice_bunt", {"batter_id": "h_001", "runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "outs": 0, "score_differential": 0, "inning": 5}),
        ("get_bullpen_status", {}),
        ("get_pitcher_fatigue_assessment", {"pitcher_id": "h_sp1"}),
        ("get_defensive_positioning", {"batter_id": "h_001", "pitcher_id": "a_sp1", "outs": 0, "runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "score_differential": 0, "inning": 5}),
        ("get_defensive_replacement_value", {"current_fielder_id": "h_002", "replacement_id": "h_011", "position": "SS"}),
        ("get_platoon_comparison", {"current_batter_id": "h_004", "pinch_hitter_id": "h_012", "pitcher_id": "a_sp1"}),
    ]
    for tool_name, kwargs in valid_inputs:
        valid, error = validate_tool_input(tool_name, **kwargs)
        assert valid, f"{tool_name}: should pass with {kwargs}, got error: {error}"


def test_validate_tool_input_all_tools_invalid():
    """All 12 tools fail validation with obviously wrong inputs via validate_tool_input."""
    invalid_inputs = [
        ("get_batter_stats", {"player_id": ""}),
        ("get_pitcher_stats", {"player_id": "", "vs_hand": "X"}),
        ("get_matchup_data", {"batter_id": "", "pitcher_id": ""}),
        ("get_run_expectancy", {"runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "outs": 5}),
        ("get_win_probability", {"inning": 0, "half": "MIDDLE", "outs": 0, "runner_on_first": False, "runner_on_second": False, "runner_on_third": False, "score_differential": 0}),
        ("evaluate_stolen_base", {"runner_id": "", "target_base": 0, "pitcher_id": "a_sp1", "catcher_id": "a_008"}),
        ("evaluate_sacrifice_bunt", {"batter_id": "", "runner_on_first": True, "runner_on_second": False, "runner_on_third": False, "outs": 5, "score_differential": 0, "inning": 0}),
        ("get_bullpen_status", {"team": "neither"}),
        ("get_pitcher_fatigue_assessment", {"pitcher_id": "", "pitch_count": -5}),
        ("get_defensive_positioning", {"batter_id": "", "pitcher_id": "", "outs": 5, "runner_on_first": False, "runner_on_second": False, "runner_on_third": False, "score_differential": 0, "inning": 0}),
        ("get_defensive_replacement_value", {"current_fielder_id": "", "replacement_id": "", "position": "XX"}),
        ("get_platoon_comparison", {"current_batter_id": "", "pinch_hitter_id": "", "pitcher_id": ""}),
    ]
    for tool_name, kwargs in invalid_inputs:
        valid, error = validate_tool_input(tool_name, **kwargs)
        assert not valid, f"{tool_name}: should fail with {kwargs}"
        assert error is not None and len(error) > 0, f"{tool_name}: error message should be non-empty"


# -----------------------------------------------------------------------
# Additional: Existing tool validation still works as before
# -----------------------------------------------------------------------

def test_existing_tool_validation_batter_stats():
    """Existing inline validation in get_batter_stats still works."""
    result = raw_parse(get_batter_stats("h_001", vs_hand="L"))
    assert result["status"] == "ok"

    result = raw_parse(get_batter_stats("h_001", home_away="home"))
    assert result["status"] == "ok"

    result = raw_parse(get_batter_stats("h_001", recency_window="last_7"))
    assert result["status"] == "ok"


def test_existing_tool_validation_run_expectancy():
    """Existing inline validation in get_run_expectancy still works."""
    result = raw_parse(get_run_expectancy(True, False, False, 0))
    assert result["status"] == "ok"

    result = raw_parse(get_run_expectancy(True, False, False, 5))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_existing_tool_validation_stolen_base():
    """Existing inline validation in evaluate_stolen_base still works."""
    result = raw_parse(evaluate_stolen_base("h_001", 2, "a_sp1", "a_008", True, False, False, 0))
    assert result["status"] == "ok"

    result = raw_parse(evaluate_stolen_base("h_001", 2, "a_sp1", "a_008", False, False, False, 0))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_SITUATION"


def test_existing_tool_validation_win_probability():
    """Existing inline validation in get_win_probability still works."""
    result = raw_parse(get_win_probability(5, "TOP", 1, True, False, False, 2))
    assert result["status"] == "ok"

    result = raw_parse(get_win_probability(0, "TOP", 0, False, False, False, 0))
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

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
