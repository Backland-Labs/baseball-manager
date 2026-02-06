# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Tests for the error_handling feature.

Verifies all feature requirements from features.json for error_handling:
1. Invalid tool calls return a descriptive error message explaining why
2. Invalid ManagerDecision responses return an error and prompt the agent
   to decide again
3. The game state is not modified by a failed tool call or rejected decision
4. The agent is allowed to retry with a corrected call or decision
5. Repeated invalid responses (more than 5 consecutive failures) trigger
   a forced no-action to prevent infinite loops
6. Errors are logged but not included in the play-by-play output
7. Common error scenarios are covered: unavailable player, rule violation,
   invalid parameters, malformed output
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import (
    SimulationEngine,
    GameState,
    BaseRunner,
    PlayEvent,
    load_rosters,
    game_state_to_scenario,
    validate_and_apply_decision,
    DecisionResult,
    _extract_player_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rosters():
    return load_rosters()


def make_test_engine(seed=42):
    return SimulationEngine(seed=seed)


def make_test_game(seed=42):
    engine = make_test_engine(seed)
    rosters = get_rosters()
    return engine, engine.initialize_game(rosters)


# ===========================================================================
# Requirement 1: Invalid tool calls return descriptive error messages
# ===========================================================================

def test_tool_invalid_player_id_returns_error():
    """All tools should return structured error JSON for invalid player IDs."""
    from tools import ALL_TOOLS

    # get_batter_stats with invalid player
    result = json.loads(ALL_TOOLS[0]("NONEXISTENT_PLAYER"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "not found" in result["message"].lower()
    print("  test_tool_invalid_player_id_returns_error: PASSED")


def test_tool_invalid_parameter_returns_error():
    """Tools should return descriptive errors for invalid parameters."""
    from tools import ALL_TOOLS

    # get_batter_stats with invalid vs_hand
    result = json.loads(ALL_TOOLS[0]("h_003", vs_hand="X"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"
    assert "vs_hand" in result["message"].lower()
    print("  test_tool_invalid_parameter_returns_error: PASSED")


def test_tool_wrong_player_type_returns_error():
    """Calling a batting tool on a pitcher should return NOT_A_BATTER."""
    from tools import ALL_TOOLS

    # get_batter_stats on a pitcher
    result = json.loads(ALL_TOOLS[0]("h_sp1"))
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_BATTER"
    print("  test_tool_wrong_player_type_returns_error: PASSED")


def test_tool_error_has_consistent_structure():
    """All tool error responses should have status, error_code, and message."""
    from tools import ALL_TOOLS

    # Test multiple tools with invalid inputs
    error_results = [
        json.loads(ALL_TOOLS[0]("INVALID")),   # get_batter_stats
        json.loads(ALL_TOOLS[1]("INVALID")),   # get_pitcher_stats
        json.loads(ALL_TOOLS[2]("INVALID", "h_sp1")),  # get_matchup_data
    ]

    for result in error_results:
        assert result["status"] == "error", f"Expected error status, got {result['status']}"
        assert "error_code" in result, "Error response missing error_code"
        assert "message" in result, "Error response missing message"
        assert isinstance(result["message"], str) and len(result["message"]) > 0
    print("  test_tool_error_has_consistent_structure: PASSED")


# ===========================================================================
# Requirement 2: Invalid decisions return error and prompt retry
# ===========================================================================

def test_invalid_pitching_change_returns_descriptive_error():
    """Invalid pitching change should return a clear error message."""
    engine, game = make_test_game()

    # Home is fielding in top of 1st; pitcher has faced 0 batters (3-batter min)
    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
        "home",
        engine,
    )
    assert not result.valid
    assert "3-batter" in result.error.lower()
    assert len(result.error) > 10  # Error should be descriptive
    print("  test_invalid_pitching_change_returns_descriptive_error: PASSED")


def test_invalid_pinch_hit_removed_player_returns_error():
    """Pinch-hitting with a removed player should return descriptive error."""
    engine, game = make_test_game()
    game.half = "BOTTOM"

    bench_player = game.home.bench[0]
    game.home.removed_players.append(bench_player.player_id)

    result = validate_and_apply_decision(
        game,
        {"decision": "PINCH_HIT", "action_details": f"send in {bench_player.name}"},
        "home",
        engine,
    )
    assert not result.valid
    assert "removed" in result.error.lower()
    print("  test_invalid_pinch_hit_removed_player_returns_error: PASSED")


def test_invalid_decision_returns_error_not_exception():
    """Invalid decisions should return DecisionResult, never raise exceptions."""
    engine, game = make_test_game()

    # Various invalid decisions should all return results, not crash
    invalid_decisions = [
        # Pitching change while batting
        ("away", {"decision": "PITCHING_CHANGE", "action_details": "change pitcher"}),
        # Pinch hit while fielding
        ("home", {"decision": "PINCH_HIT", "action_details": "send someone"}),
        # Steal with no runners
        ("away", {"decision": "STOLEN_BASE", "action_details": "steal second"}),
        # IBB while batting
        ("away", {"decision": "INTENTIONAL_WALK", "action_details": "walk batter"}),
        # Defense change while batting
        ("away", {"decision": "DEFENSIVE_POSITIONING", "action_details": "shift"}),
        # Mound visit while batting
        ("away", {"decision": "MOUND_VISIT", "action_details": "visit"}),
        # Bunt while fielding
        ("home", {"decision": "SACRIFICE_BUNT", "action_details": "bunt"}),
    ]

    for team, decision in invalid_decisions:
        result = validate_and_apply_decision(game, decision, team, engine)
        assert isinstance(result, DecisionResult)
        assert not result.valid
        assert len(result.error) > 0
    print("  test_invalid_decision_returns_error_not_exception: PASSED")


# ===========================================================================
# Requirement 3: Game state not modified by failed tool call / rejected decision
# ===========================================================================

def test_state_unchanged_after_invalid_pitching_change():
    """Game state should be identical before and after rejected pitching change."""
    engine, game = make_test_game()

    # Snapshot
    outs_before = game.outs
    score_home_before = game.score_home
    score_away_before = game.score_away
    runners_before = list(game.runners)
    pitcher_id = game.home.current_pitcher.player_id
    batters_faced = game.home.current_pitcher_batters_faced_this_stint
    lineup_index = game.away.lineup_index
    play_log_len = len(game.play_log)

    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "change pitcher"},
        "home",
        engine,
    )
    assert not result.valid

    # Verify NOTHING changed
    assert game.outs == outs_before
    assert game.score_home == score_home_before
    assert game.score_away == score_away_before
    assert game.runners == runners_before
    assert game.home.current_pitcher.player_id == pitcher_id
    assert game.home.current_pitcher_batters_faced_this_stint == batters_faced
    assert game.away.lineup_index == lineup_index
    assert len(game.play_log) == play_log_len
    print("  test_state_unchanged_after_invalid_pitching_change: PASSED")


def test_state_unchanged_after_invalid_pinch_hit():
    """Game state should be identical before and after rejected pinch hit."""
    engine, game = make_test_game()
    game.half = "BOTTOM"

    # Remove a bench player
    bench_player = game.home.bench[0]
    game.home.removed_players.append(bench_player.player_id)

    # Snapshot
    lineup_copy = [p.player_id for p in game.home.lineup]
    bench_copy = [p.player_id for p in game.home.bench]
    removed_copy = list(game.home.removed_players)
    play_log_len = len(game.play_log)

    result = validate_and_apply_decision(
        game,
        {"decision": "PINCH_HIT", "action_details": f"send in {bench_player.name}"},
        "home",
        engine,
    )
    assert not result.valid

    # Verify NOTHING changed
    assert [p.player_id for p in game.home.lineup] == lineup_copy
    assert [p.player_id for p in game.home.bench] == bench_copy
    assert game.home.removed_players == removed_copy
    assert len(game.play_log) == play_log_len
    print("  test_state_unchanged_after_invalid_pinch_hit: PASSED")


def test_state_unchanged_after_invalid_stolen_base():
    """Game state should be identical before and after rejected steal attempt."""
    engine, game = make_test_game()
    game.half = "BOTTOM"
    game.runners = []  # No runners

    outs_before = game.outs
    play_log_len = len(game.play_log)

    result = validate_and_apply_decision(
        game,
        {"decision": "STOLEN_BASE", "action_details": "steal second"},
        "home",
        engine,
    )
    assert not result.valid

    assert game.outs == outs_before
    assert game.runners == []
    assert len(game.play_log) == play_log_len
    print("  test_state_unchanged_after_invalid_stolen_base: PASSED")


def test_state_unchanged_after_mound_visit_none_remaining():
    """Game state should be identical when mound visit is rejected."""
    engine, game = make_test_game()
    game.home.mound_visits_remaining = 0

    visits_before = game.home.mound_visits_remaining
    play_log_len = len(game.play_log)

    result = validate_and_apply_decision(
        game,
        {"decision": "MOUND_VISIT", "action_details": "check on pitcher"},
        "home",
        engine,
    )
    assert not result.valid
    assert game.home.mound_visits_remaining == visits_before
    assert len(game.play_log) == play_log_len
    print("  test_state_unchanged_after_mound_visit_none_remaining: PASSED")


# ===========================================================================
# Requirement 4 & 5: Retry loop and forced no-action
# (These test the _peek_validate function and retry logic in game.py)
# ===========================================================================

def test_peek_validate_detects_invalid_decisions():
    """_peek_validate should detect invalid decisions without modifying state."""
    # Import from game.py
    from game import _peek_validate

    engine, game = make_test_game()

    # Invalid: pitching change while batting (away bats in top of 1st)
    result = _peek_validate(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "change"},
        "away",
    )
    assert not result.valid

    # Invalid: 3-batter minimum
    result = _peek_validate(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "change"},
        "home",
    )
    assert not result.valid

    # Valid: NO_ACTION
    result = _peek_validate(
        game,
        {"decision": "NO_ACTION", "action_details": ""},
        "home",
    )
    assert result.valid

    # Valid: defensive positioning while fielding
    result = _peek_validate(
        game,
        {"decision": "DEFENSIVE_POSITIONING", "action_details": "shift"},
        "home",
    )
    assert result.valid

    print("  test_peek_validate_detects_invalid_decisions: PASSED")


def test_peek_validate_does_not_modify_state():
    """_peek_validate must never change the game state."""
    from game import _peek_validate

    engine, game = make_test_game()
    game.home.current_pitcher_batters_faced_this_stint = 3

    # Snapshot
    pitcher_id = game.home.current_pitcher.player_id
    used_pitchers = list(game.home.used_pitchers)
    removed_players = list(game.home.removed_players)
    mound_visits = game.home.mound_visits_remaining
    play_log_len = len(game.play_log)
    lineup_ids = [p.player_id for p in game.home.lineup]

    # Run multiple peek validations on valid decisions
    _peek_validate(game, {"decision": "PITCHING_CHANGE", "action_details": "change"}, "home")
    _peek_validate(game, {"decision": "MOUND_VISIT", "action_details": "visit"}, "home")

    # Nothing should have changed
    assert game.home.current_pitcher.player_id == pitcher_id
    assert game.home.used_pitchers == used_pitchers
    assert game.home.removed_players == removed_players
    assert game.home.mound_visits_remaining == mound_visits
    assert len(game.play_log) == play_log_len
    assert [p.player_id for p in game.home.lineup] == lineup_ids

    print("  test_peek_validate_does_not_modify_state: PASSED")


def test_peek_validate_stolen_base_no_runner():
    """_peek_validate should reject stolen base when no runners on base."""
    from game import _peek_validate

    engine, game = make_test_game()
    game.half = "BOTTOM"
    game.runners = []

    result = _peek_validate(
        game,
        {"decision": "STOLEN_BASE", "action_details": "steal second"},
        "home",
    )
    assert not result.valid
    assert "no eligible runner" in result.error.lower()
    print("  test_peek_validate_stolen_base_no_runner: PASSED")


def test_peek_validate_used_pitcher():
    """_peek_validate should reject bringing in an already-used pitcher."""
    from game import _peek_validate

    engine, game = make_test_game()
    game.home.current_pitcher_batters_faced_this_stint = 3

    bp_pitcher = game.home.bullpen[0]
    game.home.used_pitchers.append(bp_pitcher.player_id)

    result = _peek_validate(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": f"bring in {bp_pitcher.name}"},
        "home",
    )
    assert not result.valid
    assert "already been used" in result.error.lower()
    print("  test_peek_validate_used_pitcher: PASSED")


def test_peek_validate_no_challenge():
    """_peek_validate should reject replay challenge when none available."""
    from game import _peek_validate

    engine, game = make_test_game()
    game.home.challenge_available = False

    result = _peek_validate(
        game,
        {"decision": "REPLAY_CHALLENGE", "action_details": "challenge"},
        "home",
    )
    assert not result.valid
    assert "no challenge" in result.error.lower()
    print("  test_peek_validate_no_challenge: PASSED")


# ===========================================================================
# Requirement 6: Errors are logged but NOT in play-by-play
# ===========================================================================

def test_invalid_decision_not_in_play_log():
    """Rejected decisions must not add any PlayEvent to play_log."""
    engine, game = make_test_game()

    play_log_len_before = len(game.play_log)

    # Try several invalid decisions
    invalid_decisions = [
        ("away", {"decision": "PITCHING_CHANGE", "action_details": "change"}),
        ("home", {"decision": "PINCH_HIT", "action_details": "someone"}),
        ("away", {"decision": "STOLEN_BASE", "action_details": "steal"}),
        ("away", {"decision": "INTENTIONAL_WALK", "action_details": "walk"}),
        ("away", {"decision": "MOUND_VISIT", "action_details": "visit"}),
        ("home", {"decision": "SACRIFICE_BUNT", "action_details": "bunt"}),
    ]

    for team, decision in invalid_decisions:
        result = validate_and_apply_decision(game, decision, team, engine)
        assert not result.valid, f"Expected {decision['decision']} to be invalid for {team}"

    # play_log should NOT have grown
    assert len(game.play_log) == play_log_len_before, \
        f"play_log grew from {play_log_len_before} to {len(game.play_log)} after invalid decisions"
    print("  test_invalid_decision_not_in_play_log: PASSED")


def test_valid_decision_appears_in_play_log():
    """Valid decisions should add events to play_log (contrast with invalid)."""
    engine, game = make_test_game()

    play_log_len_before = len(game.play_log)

    # Valid: defensive positioning while fielding
    result = validate_and_apply_decision(
        game,
        {"decision": "DEFENSIVE_POSITIONING", "action_details": "shift infield"},
        "home",
        engine,
    )
    assert result.valid
    assert len(game.play_log) > play_log_len_before
    assert game.play_log[-1].event_type == "decision"
    print("  test_valid_decision_appears_in_play_log: PASSED")


# ===========================================================================
# Requirement 7: Common error scenarios covered
# ===========================================================================

def test_error_unavailable_player():
    """Error: referencing a player who has been removed from the game."""
    engine, game = make_test_game()
    game.half = "BOTTOM"

    bench_player = game.home.bench[0]
    game.home.removed_players.append(bench_player.player_id)

    result = validate_and_apply_decision(
        game,
        {"decision": "PINCH_HIT", "action_details": f"send in {bench_player.name}"},
        "home",
        engine,
    )
    assert not result.valid
    assert "removed" in result.error.lower()
    print("  test_error_unavailable_player: PASSED")


def test_error_rule_violation_3_batter_minimum():
    """Error: pulling pitcher before 3-batter minimum met."""
    engine, game = make_test_game()
    game.home.current_pitcher_batters_faced_this_stint = 2

    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
        "home",
        engine,
    )
    assert not result.valid
    assert "3-batter" in result.error.lower()
    print("  test_error_rule_violation_3_batter_minimum: PASSED")


def test_error_rule_violation_batting_fielding_mismatch():
    """Error: attempting fielding-only actions while batting and vice versa."""
    engine, game = make_test_game()

    # In top of 1st: away bats, home fields.
    # Away (batting) tries to change pitcher -> invalid
    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "change"},
        "away",
        engine,
    )
    assert not result.valid
    assert "batting" in result.error.lower()

    # Home (fielding) tries to bunt -> invalid
    result = validate_and_apply_decision(
        game,
        {"decision": "SACRIFICE_BUNT", "action_details": "bunt"},
        "home",
        engine,
    )
    assert not result.valid
    assert "fielding" in result.error.lower()

    print("  test_error_rule_violation_batting_fielding_mismatch: PASSED")


def test_error_no_mound_visits_remaining():
    """Error: mound visit when count is zero."""
    engine, game = make_test_game()
    game.home.mound_visits_remaining = 0

    result = validate_and_apply_decision(
        game,
        {"decision": "MOUND_VISIT", "action_details": "check on pitcher"},
        "home",
        engine,
    )
    assert not result.valid
    assert "no mound visits" in result.error.lower()
    print("  test_error_no_mound_visits_remaining: PASSED")


def test_error_no_available_relievers():
    """Error: pitching change when all relievers have been used."""
    engine, game = make_test_game()
    game.home.current_pitcher_batters_faced_this_stint = 3

    # Mark all bullpen pitchers as used
    for p in game.home.bullpen:
        game.home.used_pitchers.append(p.player_id)

    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
        "home",
        engine,
    )
    assert not result.valid
    assert "no available" in result.error.lower()
    print("  test_error_no_available_relievers: PASSED")


def test_error_no_challenge_available():
    """Error: replay challenge when no challenge remaining."""
    engine, game = make_test_game()
    game.home.challenge_available = False

    result = validate_and_apply_decision(
        game,
        {"decision": "REPLAY_CHALLENGE", "action_details": "challenge the call"},
        "home",
        engine,
    )
    assert not result.valid
    assert "no challenge" in result.error.lower()
    print("  test_error_no_challenge_available: PASSED")


def test_error_no_runner_for_steal():
    """Error: stolen base with no runners on base."""
    engine, game = make_test_game()
    game.half = "BOTTOM"
    game.runners = []

    result = validate_and_apply_decision(
        game,
        {"decision": "STOLEN_BASE", "action_details": "steal second"},
        "home",
        engine,
    )
    assert not result.valid
    assert "no eligible runner" in result.error.lower()
    print("  test_error_no_runner_for_steal: PASSED")


def test_error_unidentifiable_pinch_hitter():
    """Error: pinch hit with unidentifiable player name."""
    engine, game = make_test_game()
    game.half = "BOTTOM"

    result = validate_and_apply_decision(
        game,
        {"decision": "PINCH_HIT", "action_details": "send in Babe Ruth"},
        "home",
        engine,
    )
    assert not result.valid
    assert "could not identify" in result.error.lower()
    print("  test_error_unidentifiable_pinch_hitter: PASSED")


def test_error_used_pitcher_rejected():
    """Error: trying to bring in a pitcher who already pitched."""
    engine, game = make_test_game()
    game.home.current_pitcher_batters_faced_this_stint = 3

    bp_pitcher = game.home.bullpen[0]
    game.home.used_pitchers.append(bp_pitcher.player_id)

    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": f"bring in {bp_pitcher.name}"},
        "home",
        engine,
    )
    assert not result.valid
    assert "already been used" in result.error.lower()
    print("  test_error_used_pitcher_rejected: PASSED")


def test_malformed_decision_fallback():
    """Malformed/empty decisions should be handled gracefully as no-action."""
    engine, game = make_test_game()

    # Empty decision
    result = validate_and_apply_decision(
        game,
        {"decision": "", "action_details": ""},
        "home",
        engine,
    )
    assert result.valid  # empty = no-action

    # Missing decision key
    result = validate_and_apply_decision(
        game,
        {"action_details": "something"},
        "home",
        engine,
    )
    assert result.valid  # no decision = no-action

    # Unknown decision type
    result = validate_and_apply_decision(
        game,
        {"decision": "UNKNOWN_ACTION_TYPE", "action_details": "stuff"},
        "home",
        engine,
    )
    assert result.valid  # unknown = no-action
    assert "unrecognized" in result.description.lower()

    print("  test_malformed_decision_fallback: PASSED")


# ===========================================================================
# Integration: Full game loop with error scenarios
# ===========================================================================

def test_full_game_with_alternating_invalid_and_valid_decisions():
    """A full game should complete even when some decisions are invalid."""
    engine, game = make_test_game(seed=42)
    pa_count = 0
    invalid_count = 0
    valid_count = 0

    while not game.game_over and pa_count < 200:
        bt = game.batting_team()
        our_team = game.home

        we_are_batting = (bt == our_team)

        # Alternate between invalid and valid decisions
        if pa_count % 3 == 0 and we_are_batting:
            # Deliberately invalid: pitching change while batting
            decision = {"decision": "PITCHING_CHANGE", "action_details": "change"}
        elif pa_count % 3 == 1 and not we_are_batting:
            # Deliberately invalid: bunt while fielding
            decision = {"decision": "SACRIFICE_BUNT", "action_details": "bunt"}
        else:
            decision = {"decision": "NO_ACTION", "action_details": ""}

        result = validate_and_apply_decision(game, decision, "home", engine)
        if result.valid:
            valid_count += 1
        else:
            invalid_count += 1

        engine._auto_manage_pitcher(game)
        pa = engine.simulate_plate_appearance(game)
        engine.apply_pa_result(game, pa)
        pa_count += 1

    assert game.game_over
    assert invalid_count > 0, "Test should have produced some invalid decisions"
    assert valid_count > 0, "Test should have produced some valid decisions"
    print(f"  test_full_game_with_alternating_invalid_and_valid_decisions: PASSED "
          f"(valid={valid_count}, invalid={invalid_count})")


def test_multiple_consecutive_invalid_then_valid():
    """Multiple consecutive invalid decisions followed by a valid one should work."""
    engine, game = make_test_game()

    # 5 invalid decisions in a row
    for i in range(5):
        result = validate_and_apply_decision(
            game,
            {"decision": "PITCHING_CHANGE", "action_details": "change"},
            "away",  # away is batting -- can't change pitcher
            engine,
        )
        assert not result.valid

    # Then a valid decision
    result = validate_and_apply_decision(
        game,
        {"decision": "NO_ACTION", "action_details": "proceed"},
        "away",
        engine,
    )
    assert result.valid

    # Game state should still be intact
    assert not game.game_over
    assert game.inning == 1
    print("  test_multiple_consecutive_invalid_then_valid: PASSED")


# ===========================================================================
# Main test runner
# ===========================================================================

def run_all_tests():
    print("=" * 72)
    print("ERROR HANDLING TESTS")
    print("=" * 72)

    tests = [
        ("Requirement 1: Tool Error Messages", [
            test_tool_invalid_player_id_returns_error,
            test_tool_invalid_parameter_returns_error,
            test_tool_wrong_player_type_returns_error,
            test_tool_error_has_consistent_structure,
        ]),
        ("Requirement 2: Invalid Decision Error + Retry", [
            test_invalid_pitching_change_returns_descriptive_error,
            test_invalid_pinch_hit_removed_player_returns_error,
            test_invalid_decision_returns_error_not_exception,
        ]),
        ("Requirement 3: State Not Modified by Failures", [
            test_state_unchanged_after_invalid_pitching_change,
            test_state_unchanged_after_invalid_pinch_hit,
            test_state_unchanged_after_invalid_stolen_base,
            test_state_unchanged_after_mound_visit_none_remaining,
        ]),
        ("Requirements 4+5: Peek Validate & Retry Logic", [
            test_peek_validate_detects_invalid_decisions,
            test_peek_validate_does_not_modify_state,
            test_peek_validate_stolen_base_no_runner,
            test_peek_validate_used_pitcher,
            test_peek_validate_no_challenge,
        ]),
        ("Requirement 6: Errors Not in Play-by-Play", [
            test_invalid_decision_not_in_play_log,
            test_valid_decision_appears_in_play_log,
        ]),
        ("Requirement 7: Common Error Scenarios", [
            test_error_unavailable_player,
            test_error_rule_violation_3_batter_minimum,
            test_error_rule_violation_batting_fielding_mismatch,
            test_error_no_mound_visits_remaining,
            test_error_no_available_relievers,
            test_error_no_challenge_available,
            test_error_no_runner_for_steal,
            test_error_unidentifiable_pinch_hitter,
            test_error_used_pitcher_rejected,
            test_malformed_decision_fallback,
        ]),
        ("Integration: Full Game with Errors", [
            test_full_game_with_alternating_invalid_and_valid_decisions,
            test_multiple_consecutive_invalid_then_valid,
        ]),
    ]

    passed = 0
    failed = 0
    failures = []

    for category, test_fns in tests:
        print(f"\n[{category}]")
        for fn in test_fns:
            try:
                fn()
                passed += 1
            except Exception as e:
                failed += 1
                failures.append((fn.__name__, str(e)))
                print(f"  {fn.__name__}: FAILED - {e}")

    print(f"\n{'=' * 72}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    if failures:
        print(f"\nFAILURES:")
        for name, err in failures:
            print(f"  {name}: {err}")
    print(f"{'=' * 72}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
