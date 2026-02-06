# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the structured_tool_responses feature.

Verifies all feature requirements from features.json:
1. Every tool returns a JSON object with a consistent top-level structure
   (success/error status, data payload)
2. Numeric values use consistent units and formats (e.g., batting average as 0.300)
3. Player references include both the MLB player ID and display name
4. Error responses include an error code and a human-readable message
5. When data is unavailable for a field, the field is present with a null value
   and an explanation, not omitted
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.response import success_response, error_response, player_ref, unavailable

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
# Helpers
# ---------------------------------------------------------------------------

def raw_parse(result: str) -> dict:
    """Parse JSON string without merging data -- returns raw envelope."""
    return json.loads(result)


# ---------------------------------------------------------------------------
# Tool invocations that produce success responses (one per tool)
# ---------------------------------------------------------------------------

def _success_calls() -> list[tuple[str, str]]:
    """Return (tool_name, json_result) for one success call per tool."""
    return [
        ("get_batter_stats", get_batter_stats("h_001")),
        ("get_pitcher_stats", get_pitcher_stats("h_sp1")),
        ("get_matchup_data", get_matchup_data("h_001", "a_sp1")),
        ("get_run_expectancy", get_run_expectancy(True, False, False, 0)),
        ("get_win_probability", get_win_probability(1, "TOP", 0, False, False, False, 0)),
        ("evaluate_stolen_base", evaluate_stolen_base("h_001", 2, "a_sp1", "a_008", True, False, False, 0)),
        ("evaluate_sacrifice_bunt", evaluate_sacrifice_bunt("h_001", True, False, False, 0, 0, 5)),
        ("get_bullpen_status", get_bullpen_status("home")),
        ("get_pitcher_fatigue_assessment", get_pitcher_fatigue_assessment("h_sp1", 60, 4.0, 2, 2)),
        ("get_defensive_positioning", get_defensive_positioning("h_001", "a_sp1", 0, True, False, False, 0, 5)),
        ("get_defensive_replacement_value", get_defensive_replacement_value("h_002", "h_011", "SS")),
        ("get_platoon_comparison", get_platoon_comparison("h_004", "h_012", "a_sp1")),
    ]


def _error_calls() -> list[tuple[str, str]]:
    """Return (tool_name, json_result) for one error call per tool."""
    return [
        ("get_batter_stats", get_batter_stats("nonexistent")),
        ("get_pitcher_stats", get_pitcher_stats("nonexistent")),
        ("get_matchup_data", get_matchup_data("nonexistent", "a_sp1")),
        ("get_run_expectancy", get_run_expectancy(True, False, False, 5)),
        ("get_win_probability", get_win_probability(0, "TOP", 0, False, False, False, 0)),
        ("evaluate_stolen_base", evaluate_stolen_base("nonexistent", 2, "a_sp1", "a_008", True, False, False, 0)),
        ("evaluate_sacrifice_bunt", evaluate_sacrifice_bunt("nonexistent", True, False, False, 0, 0, 5)),
        ("get_bullpen_status", get_bullpen_status("invalid_team")),
        ("get_pitcher_fatigue_assessment", get_pitcher_fatigue_assessment("nonexistent")),
        ("get_defensive_positioning", get_defensive_positioning("nonexistent", "a_sp1", 0, True, False, False, 0, 5)),
        ("get_defensive_replacement_value", get_defensive_replacement_value("nonexistent", "h_011", "SS")),
        ("get_platoon_comparison", get_platoon_comparison("nonexistent", "h_012", "a_sp1")),
    ]


# -----------------------------------------------------------------------
# Step 1: Every tool returns a JSON object with a consistent top-level
# structure (success/error status, data payload)
# -----------------------------------------------------------------------

def test_step1_success_envelope_structure():
    """Step 1a: Every success response has status, tool, and data keys at top level."""
    for tool_name, result_json in _success_calls():
        envelope = raw_parse(result_json)
        assert "status" in envelope, f"{tool_name}: missing 'status'"
        assert envelope["status"] == "ok", f"{tool_name}: status should be 'ok'"
        assert "tool" in envelope, f"{tool_name}: missing 'tool'"
        assert envelope["tool"] == tool_name, f"{tool_name}: tool field mismatch, got {envelope['tool']}"
        assert "data" in envelope, f"{tool_name}: missing 'data'"
        assert isinstance(envelope["data"], dict), f"{tool_name}: 'data' should be a dict"


def test_step1_error_envelope_structure():
    """Step 1b: Every error response has status, tool, error_code, and message."""
    for tool_name, result_json in _error_calls():
        envelope = raw_parse(result_json)
        assert "status" in envelope, f"{tool_name}: missing 'status'"
        assert envelope["status"] == "error", f"{tool_name}: status should be 'error'"
        assert "tool" in envelope, f"{tool_name}: missing 'tool'"
        assert envelope["tool"] == tool_name, f"{tool_name}: tool field mismatch, got {envelope['tool']}"
        assert "error_code" in envelope, f"{tool_name}: missing 'error_code'"
        assert "message" in envelope, f"{tool_name}: missing 'message'"


def test_step1_success_no_status_in_data():
    """Step 1c: The data payload itself should not contain a 'status' key (it lives at the envelope level)."""
    for tool_name, result_json in _success_calls():
        envelope = raw_parse(result_json)
        assert "status" not in envelope["data"], f"{tool_name}: 'status' should not be inside data"


def test_step1_all_responses_valid_json():
    """Step 1d: Every tool response is valid JSON."""
    for tool_name, result_json in _success_calls():
        try:
            json.loads(result_json)
        except json.JSONDecodeError:
            raise AssertionError(f"{tool_name}: success response is not valid JSON")

    for tool_name, result_json in _error_calls():
        try:
            json.loads(result_json)
        except json.JSONDecodeError:
            raise AssertionError(f"{tool_name}: error response is not valid JSON")


def test_step1_envelope_has_exactly_three_keys_on_success():
    """Step 1e: Success envelope has exactly status, tool, data -- nothing extra."""
    for tool_name, result_json in _success_calls():
        envelope = raw_parse(result_json)
        assert set(envelope.keys()) == {"status", "tool", "data"}, \
            f"{tool_name}: unexpected top-level keys: {set(envelope.keys())}"


def test_step1_envelope_has_exactly_four_keys_on_error():
    """Step 1f: Error envelope has exactly status, tool, error_code, message."""
    for tool_name, result_json in _error_calls():
        envelope = raw_parse(result_json)
        assert set(envelope.keys()) == {"status", "tool", "error_code", "message"}, \
            f"{tool_name}: unexpected top-level keys: {set(envelope.keys())}"


# -----------------------------------------------------------------------
# Step 2: Numeric values use consistent units and formats
# -----------------------------------------------------------------------

def test_step2_batting_average_format():
    """Step 2a: Batting average is a float between 0 and 1 (e.g. 0.300, not 300)."""
    result = raw_parse(get_batter_stats("h_001"))
    data = result["data"]
    avg = data["traditional"]["AVG"]
    assert isinstance(avg, float), "AVG should be a float"
    assert 0.0 < avg < 1.0, f"AVG should be 0-1 range, got {avg}"


def test_step2_obp_slg_ops_format():
    """Step 2b: OBP, SLG, OPS are floats in 0-2 range."""
    result = raw_parse(get_batter_stats("h_001"))
    data = result["data"]
    for key in ("OBP", "SLG", "OPS"):
        val = data["traditional"][key]
        assert isinstance(val, float), f"{key} should be a float"
        assert 0.0 < val < 2.0, f"{key} should be 0-2 range, got {val}"


def test_step2_probabilities_are_0_to_1():
    """Step 2c: Probabilities are expressed as 0.0 to 1.0, not percentages."""
    result = raw_parse(get_run_expectancy(True, False, False, 0))
    data = result["data"]
    prob = data["prob_scoring_at_least_one"]
    assert isinstance(prob, float), "prob_scoring_at_least_one should be a float"
    assert 0.0 <= prob <= 1.0, f"Probability should be 0-1 range, got {prob}"


def test_step2_win_probability_is_0_to_1():
    """Step 2d: Win probability is between 0 and 1."""
    result = raw_parse(get_win_probability(5, "TOP", 1, True, False, False, 0))
    data = result["data"]
    wp = data["win_probability"]
    assert isinstance(wp, float), "win_probability should be a float"
    assert 0.0 <= wp <= 1.0, f"Win probability should be 0-1 range, got {wp}"


def test_step2_era_format():
    """Step 2e: ERA is a positive float (typical range 1.0-8.0)."""
    result = raw_parse(get_pitcher_stats("h_sp1"))
    data = result["data"]
    era = data["traditional"]["ERA"]
    assert isinstance(era, float), "ERA should be a float"
    assert 0.0 < era < 15.0, f"ERA should be in realistic range, got {era}"


def test_step2_pct_rates_as_proportions():
    """Step 2f: K%, BB%, chase rate, whiff rate are proportions (0-1), not percentages."""
    result = raw_parse(get_batter_stats("h_001"))
    data = result["data"]
    disc = data["plate_discipline"]
    for key in ("K_pct", "BB_pct", "chase_rate", "whiff_rate"):
        val = disc[key]
        assert isinstance(val, float), f"{key} should be a float"
        assert 0.0 <= val <= 1.0, f"{key} should be 0-1 range, got {val}"


def test_step2_wrc_plus_is_integer():
    """Step 2g: wRC+ is an integer (100 = league average)."""
    result = raw_parse(get_batter_stats("h_001"))
    data = result["data"]
    wrc_plus = data["advanced"]["wRC_plus"]
    assert isinstance(wrc_plus, int), "wRC+ should be an integer"
    assert 40 <= wrc_plus <= 200, f"wRC+ should be realistic, got {wrc_plus}"


def test_step2_exit_velocity_in_mph():
    """Step 2h: Exit velocity is in mph (typical range 80-100)."""
    result = raw_parse(get_batter_stats("h_003"))
    data = result["data"]
    ev = data["batted_ball"]["exit_velocity"]
    assert isinstance(ev, (int, float)), "exit_velocity should be numeric"
    assert 80.0 <= ev <= 100.0, f"Exit velocity should be realistic mph, got {ev}"


def test_step2_leverage_index_positive():
    """Step 2i: Leverage index is a positive float."""
    result = raw_parse(get_win_probability(7, "BOTTOM", 2, True, True, False, -1))
    data = result["data"]
    li = data["leverage_index"]
    assert isinstance(li, (int, float)), "leverage_index should be numeric"
    assert li >= 0.0, f"Leverage index should be non-negative, got {li}"


def test_step2_expected_runs_reasonable():
    """Step 2j: Expected runs is a non-negative float in realistic range."""
    result = raw_parse(get_run_expectancy(True, True, True, 0))
    data = result["data"]
    er = data["expected_runs"]
    assert isinstance(er, float), "expected_runs should be a float"
    assert 0.0 <= er <= 5.0, f"Expected runs should be realistic, got {er}"


# -----------------------------------------------------------------------
# Step 3: Player references include both the MLB player ID and display name
# -----------------------------------------------------------------------

def test_step3_batter_stats_player_ref():
    """Step 3a: get_batter_stats returns player_id and player_name."""
    result = raw_parse(get_batter_stats("h_001"))
    data = result["data"]
    assert "player_id" in data, "Missing player_id"
    assert "player_name" in data, "Missing player_name"
    assert data["player_id"] == "h_001"
    assert isinstance(data["player_name"], str)
    assert len(data["player_name"]) > 0


def test_step3_pitcher_stats_player_ref():
    """Step 3b: get_pitcher_stats returns player_id and player_name."""
    result = raw_parse(get_pitcher_stats("h_sp1"))
    data = result["data"]
    assert "player_id" in data, "Missing player_id"
    assert "player_name" in data, "Missing player_name"
    assert data["player_id"] == "h_sp1"


def test_step3_matchup_data_player_refs():
    """Step 3c: get_matchup_data returns both batter and pitcher with IDs and names."""
    result = raw_parse(get_matchup_data("h_001", "a_sp1"))
    data = result["data"]
    assert "batter_id" in data
    assert "batter_name" in data
    assert "pitcher_id" in data
    assert "pitcher_name" in data
    assert data["batter_id"] == "h_001"
    assert data["pitcher_id"] == "a_sp1"
    assert isinstance(data["batter_name"], str)
    assert isinstance(data["pitcher_name"], str)


def test_step3_fatigue_player_ref():
    """Step 3d: get_pitcher_fatigue_assessment returns pitcher_id and pitcher_name."""
    result = raw_parse(get_pitcher_fatigue_assessment("h_sp1", 60, 4.0))
    data = result["data"]
    assert "pitcher_id" in data
    assert "pitcher_name" in data
    assert data["pitcher_id"] == "h_sp1"


def test_step3_defensive_positioning_player_refs():
    """Step 3e: get_defensive_positioning returns both batter and pitcher with IDs and names."""
    result = raw_parse(get_defensive_positioning("h_001", "a_sp1", 0, True, False, False, 0, 5))
    data = result["data"]
    assert "batter_id" in data
    assert "batter_name" in data
    assert "pitcher_id" in data
    assert "pitcher_name" in data


def test_step3_replacement_value_player_refs():
    """Step 3f: get_defensive_replacement_value returns player refs for both."""
    result = raw_parse(get_defensive_replacement_value("h_002", "h_011", "SS"))
    data = result["data"]
    assert "current_fielder" in data
    assert "replacement" in data
    cf = data["current_fielder"]
    rp = data["replacement"]
    assert "player_id" in cf
    assert "name" in cf
    assert "player_id" in rp
    assert "name" in rp


def test_step3_platoon_player_refs():
    """Step 3g: get_platoon_comparison returns player refs for batter, pinch hitter, pitcher."""
    result = raw_parse(get_platoon_comparison("h_004", "h_012", "a_sp1"))
    data = result["data"]
    assert "current_batter" in data
    assert "pinch_hitter" in data
    assert "pitcher" in data
    for section in ("current_batter", "pinch_hitter", "pitcher"):
        assert "player_id" in data[section], f"Missing player_id in {section}"
        assert "name" in data[section], f"Missing name in {section}"


def test_step3_stolen_base_player_ref():
    """Step 3h: evaluate_stolen_base returns runner/pitcher/catcher refs."""
    result = raw_parse(evaluate_stolen_base("h_001", 2, "a_sp1", "a_008", True, False, False, 0))
    data = result["data"]
    # Should contain runner, pitcher, and catcher references with IDs and names
    assert "runner_id" in data
    assert "runner_name" in data
    assert "pitcher_id" in data
    assert "pitcher_name" in data
    assert "catcher_id" in data
    assert "catcher_name" in data


def test_step3_sacrifice_bunt_player_ref():
    """Step 3i: evaluate_sacrifice_bunt returns batter_id and batter_name."""
    result = raw_parse(evaluate_sacrifice_bunt("h_001", True, False, False, 0, 0, 5))
    data = result["data"]
    assert "batter_id" in data
    assert "batter_name" in data


# -----------------------------------------------------------------------
# Step 4: Error responses include an error code and a human-readable message
# -----------------------------------------------------------------------

def test_step4_all_errors_have_code_and_message():
    """Step 4a: Every error response has error_code and message fields."""
    for tool_name, result_json in _error_calls():
        envelope = raw_parse(result_json)
        assert envelope["status"] == "error", f"{tool_name}: expected error status"
        code = envelope["error_code"]
        msg = envelope["message"]
        assert isinstance(code, str), f"{tool_name}: error_code should be a string"
        assert len(code) > 0, f"{tool_name}: error_code should not be empty"
        assert isinstance(msg, str), f"{tool_name}: message should be a string"
        assert len(msg) > 0, f"{tool_name}: message should not be empty"


def test_step4_error_codes_are_uppercase():
    """Step 4b: Error codes use UPPER_SNAKE_CASE convention."""
    for tool_name, result_json in _error_calls():
        envelope = raw_parse(result_json)
        code = envelope["error_code"]
        assert code == code.upper(), f"{tool_name}: error_code should be uppercase, got '{code}'"
        # Should only contain letters and underscores
        assert all(c.isalpha() or c == "_" for c in code), \
            f"{tool_name}: error_code should be UPPER_SNAKE_CASE, got '{code}'"


def test_step4_invalid_player_error():
    """Step 4c: Invalid player ID returns INVALID_PLAYER_ID error code."""
    result = raw_parse(get_batter_stats("nonexistent"))
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "not found" in result["message"].lower()


def test_step4_invalid_parameter_error():
    """Step 4d: Invalid parameter returns INVALID_PARAMETER error code."""
    result = raw_parse(get_batter_stats("h_001", vs_hand="X"))
    assert result["error_code"] == "INVALID_PARAMETER"
    assert len(result["message"]) > 10  # Should be descriptive


def test_step4_not_a_batter_error():
    """Step 4e: Player without batting attributes returns NOT_A_BATTER."""
    result = raw_parse(get_batter_stats("h_bp1"))
    assert result["error_code"] == "NOT_A_BATTER"


def test_step4_not_a_pitcher_error():
    """Step 4f: Player without pitching attributes returns NOT_A_PITCHER."""
    result = raw_parse(get_pitcher_stats("h_001"))
    assert result["error_code"] == "NOT_A_PITCHER"


def test_step4_invalid_situation_error():
    """Step 4g: Invalid situation returns INVALID_SITUATION."""
    result = raw_parse(evaluate_sacrifice_bunt("h_001", False, False, False, 0, 0, 5))
    assert result["error_code"] == "INVALID_SITUATION"


def test_step4_error_messages_mention_parameter():
    """Step 4h: Error messages mention what was wrong."""
    result = raw_parse(get_run_expectancy(True, False, False, 5))
    assert "outs" in result["message"].lower() or "5" in result["message"]


def test_step4_error_no_data_key():
    """Step 4i: Error responses should NOT have a 'data' key."""
    for tool_name, result_json in _error_calls():
        envelope = raw_parse(result_json)
        assert "data" not in envelope, f"{tool_name}: error response should not have 'data'"


# -----------------------------------------------------------------------
# Step 5: When data is unavailable for a field, the field is present
# with a null value and an explanation, not omitted
# -----------------------------------------------------------------------

def test_step5_matchup_no_history_unavailable_fields():
    """Step 5a: When no matchup history exists, matchup_stats field is present with null+explanation."""
    # h_003 vs a_sp1 has 0 PA
    result = raw_parse(get_matchup_data("h_003", "a_sp1"))
    data = result["data"]

    # matchup_stats should be present (not omitted) with unavailable pattern
    assert "matchup_stats" in data, "matchup_stats should be present even when no history"
    ms = data["matchup_stats"]
    assert ms["value"] is None, "matchup_stats value should be null when unavailable"
    assert "explanation" in ms, "matchup_stats should have an explanation"
    assert len(ms["explanation"]) > 0, "explanation should not be empty"


def test_step5_matchup_no_history_outcome_dist_unavailable():
    """Step 5b: outcome_distribution is present with null+explanation when no history."""
    result = raw_parse(get_matchup_data("h_003", "a_sp1"))
    data = result["data"]

    assert "outcome_distribution" in data, "outcome_distribution should be present"
    od = data["outcome_distribution"]
    assert od["value"] is None, "outcome_distribution value should be null"
    assert "explanation" in od


def test_step5_similarity_woba_always_present():
    """Step 5c: similarity_projected_wOBA is always present, even with no history."""
    result = raw_parse(get_matchup_data("h_003", "a_sp1"))
    data = result["data"]
    assert "similarity_projected_wOBA" in data, "similarity_projected_wOBA should always be present"
    assert data["similarity_projected_wOBA"] is not None


def test_step5_pitch_vulnerability_always_present():
    """Step 5d: pitch_type_vulnerability is always present, even with no history."""
    result = raw_parse(get_matchup_data("h_003", "a_sp1"))
    data = result["data"]
    assert "pitch_type_vulnerability" in data, "pitch_type_vulnerability should always be present"
    assert len(data["pitch_type_vulnerability"]) >= 3


def test_step5_unavailable_helper():
    """Step 5e: The unavailable() helper produces the correct structure."""
    result = unavailable("No data available for this field")
    assert result["value"] is None
    assert result["explanation"] == "No data available for this field"


def test_step5_batter_splits_null_when_not_specified():
    """Step 5f: Split fields are present as null when not specified, not omitted."""
    result = raw_parse(get_batter_stats("h_001"))
    data = result["data"]
    splits = data["splits"]
    assert "vs_hand" in splits, "vs_hand should be present"
    assert splits["vs_hand"] is None, "vs_hand should be null when not specified"
    assert "home_away" in splits
    assert splits["home_away"] is None
    assert "recency_window" in splits
    assert splits["recency_window"] is None


def test_step5_pitcher_splits_null_when_not_specified():
    """Step 5g: Pitcher split fields are present as null when not specified."""
    result = raw_parse(get_pitcher_stats("h_sp1"))
    data = result["data"]
    splits = data["splits"]
    assert "vs_hand" in splits and splits["vs_hand"] is None
    assert "home_away" in splits and splits["home_away"] is None
    assert "recency_window" in splits and splits["recency_window"] is None


def test_step5_matchup_with_history_has_populated_fields():
    """Step 5h: When history exists, matchup_stats and outcome_distribution are populated dicts."""
    result = raw_parse(get_matchup_data("h_001", "a_sp1"))
    data = result["data"]

    assert data["matchup_stats"] is not None
    assert isinstance(data["matchup_stats"], dict)
    assert "AVG" in data["matchup_stats"]

    assert data["outcome_distribution"] is not None
    assert isinstance(data["outcome_distribution"], dict)
    assert "groundball" in data["outcome_distribution"]


# -----------------------------------------------------------------------
# Response helper unit tests
# -----------------------------------------------------------------------

def test_helper_success_response():
    """Verify success_response builds correct envelope."""
    result = json.loads(success_response("test_tool", {"key": "value"}))
    assert result == {"status": "ok", "tool": "test_tool", "data": {"key": "value"}}


def test_helper_error_response():
    """Verify error_response builds correct envelope."""
    result = json.loads(error_response("test_tool", "TEST_ERROR", "Something broke"))
    assert result == {
        "status": "error",
        "tool": "test_tool",
        "error_code": "TEST_ERROR",
        "message": "Something broke",
    }


def test_helper_player_ref():
    """Verify player_ref builds correct dict."""
    ref = player_ref("h_001", "Marcus Chen")
    assert ref == {"player_id": "h_001", "player_name": "Marcus Chen"}


def test_helper_unavailable():
    """Verify unavailable builds correct dict."""
    result = unavailable("No data for this field")
    assert result == {"value": None, "explanation": "No data for this field"}


# -----------------------------------------------------------------------
# Cross-tool consistency checks
# -----------------------------------------------------------------------

def test_cross_tool_all_tools_return_tool_name():
    """All 12 tools include their tool name in the response."""
    expected_tools = {
        "get_batter_stats", "get_pitcher_stats", "get_matchup_data",
        "get_run_expectancy", "get_win_probability",
        "evaluate_stolen_base", "evaluate_sacrifice_bunt",
        "get_bullpen_status", "get_pitcher_fatigue_assessment",
        "get_defensive_positioning", "get_defensive_replacement_value",
        "get_platoon_comparison",
    }
    seen_tools = set()
    for tool_name, result_json in _success_calls():
        envelope = raw_parse(result_json)
        seen_tools.add(envelope["tool"])
    assert seen_tools == expected_tools, f"Missing tools: {expected_tools - seen_tools}"


def test_cross_tool_error_tool_name_matches():
    """All 12 tools include their tool name in error responses too."""
    for tool_name, result_json in _error_calls():
        envelope = raw_parse(result_json)
        assert envelope["tool"] == tool_name, \
            f"Error response tool mismatch: expected {tool_name}, got {envelope['tool']}"


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
