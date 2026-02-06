# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Tests for the agent_decision_loop feature.

Verifies all feature requirements from features.json for agent_decision_loop:
1. At each decision point, the agent is prompted with MatchupState, RosterState,
   OpponentRosterState, and a natural language decision prompt
2. The agent may call zero or more information-gathering tools
3. After gathering context, the agent responds with a structured ManagerDecision
4. The simulation engine validates the ManagerDecision against game state and MLB rules
5. If valid, the simulation applies it and advances to the next decision point
6. If invalid, an error is returned and the agent can try again
7. Decision points include each plate appearance (offensive and defensive)
8. The loop repeats until the game reaches a terminal state
9. The opposing team's decisions are handled by automated management
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import (
    SimulationEngine,
    SimPlayer,
    GameState,
    TeamState,
    BaseRunner,
    BatterGameStats,
    PitcherGameStats,
    PlayEvent,
    load_rosters,
    game_state_to_scenario,
    validate_and_apply_decision,
    DecisionResult,
    game_state_to_dict,
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


def make_test_batter(**overrides):
    defaults = dict(
        player_id="test_b", name="Test Batter", primary_position="CF",
        bats="R", throws="R", contact=75, power=65, speed=70, eye=70,
        avg_vs_l=0.280, avg_vs_r=0.270,
    )
    defaults.update(overrides)
    return SimPlayer(**defaults)


def make_test_pitcher(**overrides):
    defaults = dict(
        player_id="test_p", name="Test Pitcher", primary_position="SP",
        bats="R", throws="R", stuff=70, control=70, stamina=70,
        velocity=93.0, era_vs_l=3.50, era_vs_r=3.20, is_pitcher=True,
    )
    defaults.update(overrides)
    return SimPlayer(**defaults)


# ===========================================================================
# Test: game_state_to_scenario -- basic conversion
# ===========================================================================

def test_scenario_has_required_keys():
    """Scenario should contain all four required top-level keys."""
    engine, game = make_test_game()
    scenario = game_state_to_scenario(game, "home")

    assert "matchup_state" in scenario
    assert "roster_state" in scenario
    assert "opponent_roster_state" in scenario
    assert "decision_prompt" in scenario
    print("  test_scenario_has_required_keys: PASSED")


def test_scenario_matchup_state_fields():
    """MatchupState should contain all required fields."""
    engine, game = make_test_game()
    scenario = game_state_to_scenario(game, "home")
    ms = scenario["matchup_state"]

    assert "inning" in ms
    assert "half" in ms
    assert "outs" in ms
    assert "count" in ms
    assert "runners" in ms
    assert "score" in ms
    assert "batting_team" in ms
    assert "batter" in ms
    assert "pitcher" in ms
    assert "on_deck_batter" in ms

    assert ms["inning"] == 1
    assert ms["half"] == "TOP"
    assert ms["outs"] == 0

    # Batter fields
    batter = ms["batter"]
    assert "player_id" in batter
    assert "name" in batter
    assert "bats" in batter
    assert "lineup_position" in batter

    # Pitcher fields
    pitcher = ms["pitcher"]
    assert "player_id" in pitcher
    assert "name" in pitcher
    assert "throws" in pitcher
    assert "pitch_count_today" in pitcher
    assert "batters_faced_today" in pitcher
    assert "times_through_order" in pitcher
    assert "innings_pitched_today" in pitcher
    assert "runs_allowed_today" in pitcher
    assert "today_line" in pitcher

    print("  test_scenario_matchup_state_fields: PASSED")


def test_scenario_roster_state_fields():
    """RosterState should contain lineup, bench, bullpen, and game management fields."""
    engine, game = make_test_game()
    scenario = game_state_to_scenario(game, "home")
    rs = scenario["roster_state"]

    assert "our_lineup" in rs
    assert "our_lineup_position" in rs
    assert "bench" in rs
    assert "bullpen" in rs
    assert "mound_visits_remaining" in rs
    assert "challenge_available" in rs

    assert len(rs["our_lineup"]) == 9
    assert rs["mound_visits_remaining"] == 5
    assert rs["challenge_available"] is True

    # Verify lineup player structure
    for player in rs["our_lineup"]:
        assert "player_id" in player
        assert "name" in player
        assert "position" in player
        assert "bats" in player
        assert "in_game" in player

    # Verify bench player structure
    assert len(rs["bench"]) >= 5
    for player in rs["bench"]:
        assert "player_id" in player
        assert "name" in player
        assert "bats" in player
        assert "positions" in player
        assert "available" in player

    # Verify bullpen structure
    assert len(rs["bullpen"]) >= 8
    for pitcher in rs["bullpen"]:
        assert "player_id" in pitcher
        assert "name" in pitcher
        assert "throws" in pitcher
        assert "role" in pitcher
        assert "available" in pitcher

    print("  test_scenario_roster_state_fields: PASSED")


def test_scenario_opponent_roster_fields():
    """OpponentRosterState should contain lineup, bench, and bullpen."""
    engine, game = make_test_game()
    scenario = game_state_to_scenario(game, "home")
    ors = scenario["opponent_roster_state"]

    assert "their_lineup" in ors
    assert "their_lineup_position" in ors
    assert "their_bench" in ors
    assert "their_bullpen" in ors

    assert len(ors["their_lineup"]) == 9
    print("  test_scenario_opponent_roster_fields: PASSED")


def test_scenario_decision_prompt_is_descriptive():
    """Decision prompt should contain situation description."""
    engine, game = make_test_game()
    scenario = game_state_to_scenario(game, "home")
    prompt = scenario["decision_prompt"]

    assert isinstance(prompt, str)
    assert len(prompt) > 50
    # Should mention the inning
    assert "1st" in prompt or "Top" in prompt
    print("  test_scenario_decision_prompt_is_descriptive: PASSED")


def test_scenario_serializes_to_json():
    """Entire scenario should be JSON-serializable."""
    engine, game = make_test_game()
    scenario = game_state_to_scenario(game, "home")
    json_str = json.dumps(scenario, default=str)
    assert len(json_str) > 500
    parsed = json.loads(json_str)
    assert parsed["matchup_state"]["inning"] == 1
    print("  test_scenario_serializes_to_json: PASSED")


# ===========================================================================
# Test: game_state_to_scenario -- with runners
# ===========================================================================

def test_scenario_with_runners():
    """Scenario should correctly report runners on base."""
    engine, game = make_test_game()
    # Put runners on base
    batter = game.away.lineup[0]
    r1 = BaseRunner(player=batter, start_base=1)
    r3 = BaseRunner(player=game.away.lineup[2], start_base=3)
    game.runners = [r1, r3]

    scenario = game_state_to_scenario(game, "home")
    runners = scenario["matchup_state"]["runners"]

    assert runners["first"] is not None
    assert runners["first"]["player_id"] == batter.player_id
    assert runners["second"] is None
    assert runners["third"] is not None
    assert runners["third"]["player_id"] == game.away.lineup[2].player_id
    print("  test_scenario_with_runners: PASSED")


# ===========================================================================
# Test: game_state_to_scenario -- away team perspective
# ===========================================================================

def test_scenario_away_team_perspective():
    """Scenario from away team perspective should swap our/opponent rosters."""
    engine, game = make_test_game()
    home_scenario = game_state_to_scenario(game, "home")
    away_scenario = game_state_to_scenario(game, "away")

    # Home team's roster should be our_lineup in home scenario
    # and their_lineup in away scenario
    home_first_player = home_scenario["roster_state"]["our_lineup"][0]["player_id"]
    away_opp_first = away_scenario["opponent_roster_state"]["their_lineup"][0]["player_id"]
    assert home_first_player == away_opp_first

    # Away team's roster should be our_lineup in away scenario
    away_first_player = away_scenario["roster_state"]["our_lineup"][0]["player_id"]
    home_opp_first = home_scenario["opponent_roster_state"]["their_lineup"][0]["player_id"]
    assert away_first_player == home_opp_first

    print("  test_scenario_away_team_perspective: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- NO_ACTION
# ===========================================================================

def test_no_action_is_always_valid():
    """NO_ACTION should always be a valid decision."""
    engine, game = make_test_game()

    for decision_type in ["NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
                          "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER"]:
        result = validate_and_apply_decision(
            game,
            {"decision": decision_type, "action_details": "proceed"},
            "home",
            engine,
        )
        assert result.valid, f"{decision_type} should be valid"

    print("  test_no_action_is_always_valid: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- PITCHING_CHANGE
# ===========================================================================

def test_pitching_change_while_batting_is_invalid():
    """Cannot make pitching change while batting."""
    engine, game = make_test_game()
    # Top of 1st: away bats, home fields. If we manage home, we are fielding.
    # If we manage away, we are batting.
    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
        "away",  # away is batting in top of 1st
        engine,
    )
    assert not result.valid
    assert "batting" in result.error.lower()
    print("  test_pitching_change_while_batting_is_invalid: PASSED")


def test_pitching_change_3_batter_minimum():
    """Pitching change blocked by 3-batter minimum."""
    engine, game = make_test_game()
    # Home is fielding in top of 1st. Pitcher has faced 0 batters.
    assert game.home.current_pitcher_batters_faced_this_stint == 0

    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
        "home",
        engine,
    )
    assert not result.valid
    assert "3-batter" in result.error.lower()
    print("  test_pitching_change_3_batter_minimum: PASSED")


def test_pitching_change_after_3_batters():
    """Pitching change allowed after facing 3+ batters."""
    engine, game = make_test_game()
    # Simulate 3 batters
    game.home.current_pitcher_batters_faced_this_stint = 3

    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
        "home",
        engine,
    )
    assert result.valid, f"Should be valid after 3 batters: {result.error}"
    assert len(result.events) > 0
    assert "Pitching change" in result.events[0].description
    print("  test_pitching_change_after_3_batters: PASSED")


def test_pitching_change_by_name():
    """Pitching change should find the pitcher by name."""
    engine, game = make_test_game()
    game.home.current_pitcher_batters_faced_this_stint = 3

    # Get a bullpen pitcher's name
    bp_pitcher = game.home.bullpen[0]

    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": f"bring in {bp_pitcher.name}"},
        "home",
        engine,
    )
    assert result.valid, f"Should find pitcher by name: {result.error}"
    assert bp_pitcher.name in result.events[0].description
    assert game.home.current_pitcher.player_id == bp_pitcher.player_id
    print("  test_pitching_change_by_name: PASSED")


def test_pitching_change_used_pitcher_rejected():
    """Cannot bring in a pitcher who has already been used."""
    engine, game = make_test_game()
    game.home.current_pitcher_batters_faced_this_stint = 3

    # Mark first bullpen pitcher as used
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
    print("  test_pitching_change_used_pitcher_rejected: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- PINCH_HIT
# ===========================================================================

def test_pinch_hit_while_fielding_is_invalid():
    """Cannot pinch hit while fielding."""
    engine, game = make_test_game()
    result = validate_and_apply_decision(
        game,
        {"decision": "PINCH_HIT", "action_details": "send in someone"},
        "home",  # home is fielding in top of 1st
        engine,
    )
    assert not result.valid
    assert "fielding" in result.error.lower()
    print("  test_pinch_hit_while_fielding_is_invalid: PASSED")


def test_pinch_hit_while_batting():
    """Pinch hit should work when batting."""
    engine, game = make_test_game()
    # Move to bottom of 1st so home bats
    game.half = "BOTTOM"

    bench_player = game.home.bench[0]
    current_batter = game.home.current_batter()

    result = validate_and_apply_decision(
        game,
        {"decision": "PINCH_HIT", "action_details": f"send in {bench_player.name} to bat for {current_batter.name}"},
        "home",
        engine,
    )
    assert result.valid, f"Pinch hit should be valid: {result.error}"
    assert len(result.events) > 0
    assert "Pinch hitter" in result.events[0].description

    # Current batter should now be the bench player
    assert game.home.current_batter().player_id == bench_player.player_id
    # Old batter should be in removed list
    assert current_batter.player_id in game.home.removed_players

    print("  test_pinch_hit_while_batting: PASSED")


def test_pinch_hit_removed_player_rejected():
    """Cannot pinch hit with a player already removed from the game."""
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
    print("  test_pinch_hit_removed_player_rejected: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- STOLEN_BASE
# ===========================================================================

def test_stolen_base_while_fielding_is_invalid():
    """Cannot attempt steal while fielding."""
    engine, game = make_test_game()
    result = validate_and_apply_decision(
        game,
        {"decision": "STOLEN_BASE", "action_details": "steal second"},
        "home",  # home is fielding in top of 1st
        engine,
    )
    assert not result.valid
    assert "fielding" in result.error.lower()
    print("  test_stolen_base_while_fielding_is_invalid: PASSED")


def test_stolen_base_no_runner():
    """Stolen base with no eligible runner is invalid."""
    engine, game = make_test_game()
    # Switch to bottom half so home bats, and ensure no runners
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
    print("  test_stolen_base_no_runner: PASSED")


def test_stolen_base_with_runner():
    """Stolen base with runner on first should be valid."""
    engine, game = make_test_game()
    game.half = "BOTTOM"
    runner = BaseRunner(player=game.home.lineup[0], start_base=1)
    game.runners = [runner]

    result = validate_and_apply_decision(
        game,
        {"decision": "STOLEN_BASE", "action_details": "steal second"},
        "home",
        engine,
    )
    assert result.valid, f"Stolen base should be valid: {result.error}"
    assert len(result.events) > 0
    # Result should be either success or caught stealing
    desc = result.events[0].description
    assert "steals" in desc.lower() or "caught" in desc.lower()
    print("  test_stolen_base_with_runner: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- INTENTIONAL_WALK
# ===========================================================================

def test_ibb_while_batting_is_invalid():
    """Cannot issue intentional walk while batting."""
    engine, game = make_test_game()
    game.half = "BOTTOM"

    result = validate_and_apply_decision(
        game,
        {"decision": "INTENTIONAL_WALK", "action_details": "walk the batter"},
        "home",  # home is batting in bottom
        engine,
    )
    assert not result.valid
    assert "batting" in result.error.lower()
    print("  test_ibb_while_batting_is_invalid: PASSED")


def test_ibb_while_fielding():
    """Intentional walk should be valid when fielding."""
    engine, game = make_test_game()
    # Top of 1st: home is fielding

    batter_before = game.away.current_batter()
    result = validate_and_apply_decision(
        game,
        {"decision": "INTENTIONAL_WALK", "action_details": f"walk {batter_before.name}"},
        "home",
        engine,
    )
    assert result.valid, f"IBB should be valid: {result.error}"
    assert len(result.events) > 0
    assert "intentional walk" in result.events[0].description.lower()

    # Batter should now be on first
    assert any(r.start_base == 1 for r in game.runners)
    print("  test_ibb_while_fielding: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- DEFENSIVE_POSITIONING
# ===========================================================================

def test_defensive_positioning():
    """Defensive positioning should be valid when fielding."""
    engine, game = make_test_game()

    result = validate_and_apply_decision(
        game,
        {"decision": "DEFENSIVE_POSITIONING", "action_details": "shift infield to pull side"},
        "home",
        engine,
    )
    assert result.valid
    assert len(result.events) > 0
    print("  test_defensive_positioning: PASSED")


def test_defensive_positioning_while_batting_invalid():
    """Cannot adjust defense while batting."""
    engine, game = make_test_game()
    game.half = "BOTTOM"

    result = validate_and_apply_decision(
        game,
        {"decision": "DEFENSIVE_POSITIONING", "action_details": "shift"},
        "home",
        engine,
    )
    assert not result.valid
    print("  test_defensive_positioning_while_batting_invalid: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- MOUND_VISIT
# ===========================================================================

def test_mound_visit():
    """Mound visit should decrement count and be valid when fielding."""
    engine, game = make_test_game()
    assert game.home.mound_visits_remaining == 5

    result = validate_and_apply_decision(
        game,
        {"decision": "MOUND_VISIT", "action_details": "check on pitcher"},
        "home",
        engine,
    )
    assert result.valid
    assert game.home.mound_visits_remaining == 4
    print("  test_mound_visit: PASSED")


def test_mound_visit_none_remaining():
    """Mound visit with 0 remaining should be invalid."""
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
    print("  test_mound_visit_none_remaining: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- SACRIFICE_BUNT
# ===========================================================================

def test_sacrifice_bunt():
    """Sacrifice bunt should be valid when batting."""
    engine, game = make_test_game()
    game.half = "BOTTOM"

    result = validate_and_apply_decision(
        game,
        {"decision": "SACRIFICE_BUNT", "action_details": "bunt"},
        "home",
        engine,
    )
    assert result.valid
    print("  test_sacrifice_bunt: PASSED")


def test_sacrifice_bunt_while_fielding_invalid():
    """Cannot bunt while fielding."""
    engine, game = make_test_game()

    result = validate_and_apply_decision(
        game,
        {"decision": "SACRIFICE_BUNT", "action_details": "bunt"},
        "home",
        engine,
    )
    assert not result.valid
    print("  test_sacrifice_bunt_while_fielding_invalid: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- unknown decision type
# ===========================================================================

def test_unknown_decision_treated_as_no_action():
    """Unknown decision types should be treated as no action (valid)."""
    engine, game = make_test_game()

    result = validate_and_apply_decision(
        game,
        {"decision": "SOME_WEIRD_THING", "action_details": "whatever"},
        "home",
        engine,
    )
    assert result.valid
    assert "unrecognized" in result.description.lower()
    print("  test_unknown_decision_treated_as_no_action: PASSED")


# ===========================================================================
# Test: validate_and_apply_decision -- empty decision
# ===========================================================================

def test_empty_decision_is_no_action():
    """Empty decision string should be treated as no action."""
    engine, game = make_test_game()

    result = validate_and_apply_decision(
        game,
        {"decision": "", "action_details": ""},
        "home",
        engine,
    )
    assert result.valid
    print("  test_empty_decision_is_no_action: PASSED")


# ===========================================================================
# Test: _extract_player_id
# ===========================================================================

def test_extract_player_id_by_id():
    """Should find player by player_id in text."""
    engine, game = make_test_game()
    pid = _extract_player_id("bring in h_bp1 to pitch", game.home)
    assert pid == "h_bp1"
    print("  test_extract_player_id_by_id: PASSED")


def test_extract_player_id_by_name():
    """Should find player by name in text."""
    engine, game = make_test_game()
    pid = _extract_player_id("bring in Greg Foster to pitch", game.home)
    # Greg Foster should be in bullpen
    assert pid is not None
    print("  test_extract_player_id_by_name: PASSED")


def test_extract_player_id_not_found():
    """Should return None when player not found."""
    engine, game = make_test_game()
    pid = _extract_player_id("bring in Babe Ruth", game.home)
    assert pid is None
    print("  test_extract_player_id_not_found: PASSED")


# ===========================================================================
# Test: Scenario reflects game progress
# ===========================================================================

def test_scenario_updates_with_game_state():
    """Scenario should reflect updated game state after events."""
    engine, game = make_test_game()

    # Simulate a few plate appearances
    for _ in range(10):
        if game.game_over:
            break
        engine._auto_manage_pitcher(game)
        pa = engine.simulate_plate_appearance(game)
        engine.apply_pa_result(game, pa)

    # Generate scenario -- should reflect current game state
    scenario = game_state_to_scenario(game, "home")
    ms = scenario["matchup_state"]

    # The inning may have advanced
    assert ms["inning"] >= 1
    # Score should reflect actual state
    assert ms["score"]["home"] == game.score_home
    assert ms["score"]["away"] == game.score_away
    print("  test_scenario_updates_with_game_state: PASSED")


# ===========================================================================
# Test: Full game simulation with decision validation
# ===========================================================================

def test_full_game_with_no_action_decisions():
    """A full game should complete when all agent decisions are NO_ACTION."""
    engine, game = make_test_game(seed=42)

    while not game.game_over:
        if game.inning > 15:
            break

        # Always apply NO_ACTION (simulates an agent that never intervenes)
        result = validate_and_apply_decision(
            game,
            {"decision": "NO_ACTION", "action_details": "let play proceed"},
            "home",
            engine,
        )
        assert result.valid

        # Automated pitcher management for both teams
        engine._auto_manage_pitcher(game)

        # Simulate PA
        pa = engine.simulate_plate_appearance(game)
        engine.apply_pa_result(game, pa)

    assert game.game_over
    assert game.winning_team != ""
    print(f"  test_full_game_with_no_action_decisions: PASSED ({game.winning_team} wins {game.score_home}-{game.score_away})")


def test_full_game_with_mixed_decisions():
    """A full game with various agent decisions should complete normally."""
    engine, game = make_test_game(seed=100)
    decision_count = 0
    valid_count = 0
    invalid_count = 0

    while not game.game_over:
        if game.inning > 15:
            break

        bt = game.batting_team()
        ft = game.fielding_team()
        is_home = True
        our_team = game.home

        we_are_batting = (bt == our_team)
        we_are_fielding = (ft == our_team)

        # Make varied decisions based on game state
        decision = {"decision": "NO_ACTION", "action_details": ""}

        if we_are_fielding:
            pitcher = game.current_pitcher()
            pstats = ft.get_pitcher_stats(pitcher.player_id)
            # Try pitching change when pitcher is tired
            if pstats.pitches > 80 and ft.current_pitcher_batters_faced_this_stint >= 3:
                available = [p for p in our_team.bullpen if p.player_id not in our_team.used_pitchers]
                if available:
                    decision = {"decision": "PITCHING_CHANGE",
                                "action_details": f"bring in {available[0].name}"}

        elif we_are_batting:
            # Try stolen base if runner on first
            r1 = game.runner_on(1)
            if r1 and r1.player.speed > 70 and not game.runner_on(2):
                decision = {"decision": "STOLEN_BASE", "action_details": "steal second"}

        result = validate_and_apply_decision(game, decision, "home", engine)
        decision_count += 1
        if result.valid:
            valid_count += 1
        else:
            invalid_count += 1

        # Handle decisions that consume the PA
        if result.valid and decision.get("decision") in ("INTENTIONAL_WALK", "IBB"):
            if game.game_over:
                break
            continue
        if result.valid and decision.get("decision") in ("STOLEN_BASE", "STEAL"):
            if game.game_over or game.outs >= 3:
                continue

        engine._auto_manage_pitcher(game)
        pa = engine.simulate_plate_appearance(game)
        engine.apply_pa_result(game, pa)

    assert game.game_over
    assert valid_count > 0
    print(f"  test_full_game_with_mixed_decisions: PASSED "
          f"(decisions={decision_count}, valid={valid_count}, invalid={invalid_count})")


# ===========================================================================
# Test: Decision logging structure
# ===========================================================================

def test_decision_log_structure():
    """Decision log entries should have all required fields."""
    import time

    engine, game = make_test_game()
    scenario = game_state_to_scenario(game, "home")

    decision_dict = {
        "decision": "NO_ACTION",
        "action_details": "proceed",
        "confidence": 0.9,
        "reasoning": "Nothing to do",
        "key_factors": ["low leverage"],
        "risks": [],
    }

    entry = {
        "turn": 1,
        "inning": game.inning,
        "half": game.half,
        "outs": game.outs,
        "score_home": game.score_home,
        "score_away": game.score_away,
        "situation": game.situation_display(),
        "decision": decision_dict,
        "timestamp": time.time(),
    }

    # Verify it serializes to JSON
    json_str = json.dumps(entry, default=str)
    parsed = json.loads(json_str)
    assert parsed["turn"] == 1
    assert parsed["inning"] == 1
    assert parsed["decision"]["decision"] == "NO_ACTION"
    print("  test_decision_log_structure: PASSED")


# ===========================================================================
# Test: Game state not modified by invalid decision
# ===========================================================================

def test_invalid_decision_does_not_modify_state():
    """An invalid decision should not change game state."""
    engine, game = make_test_game()

    # Snapshot state
    outs_before = game.outs
    score_home_before = game.score_home
    score_away_before = game.score_away
    runners_before = len(game.runners)
    pitcher_id_before = game.home.current_pitcher.player_id

    # Attempt invalid pitching change (3-batter minimum)
    result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "change pitcher"},
        "home",
        engine,
    )
    assert not result.valid

    # Verify state unchanged
    assert game.outs == outs_before
    assert game.score_home == score_home_before
    assert game.score_away == score_away_before
    assert len(game.runners) == runners_before
    assert game.home.current_pitcher.player_id == pitcher_id_before
    print("  test_invalid_decision_does_not_modify_state: PASSED")


# ===========================================================================
# Test: Decision points at each plate appearance
# ===========================================================================

def test_decision_points_each_pa():
    """Verify we can generate a scenario at every plate appearance."""
    engine, game = make_test_game(seed=42)
    pa_count = 0

    while not game.game_over and pa_count < 50:
        # Generate scenario (this is what happens at each decision point)
        scenario = game_state_to_scenario(game, "home")
        assert scenario["matchup_state"]["inning"] >= 1
        assert "decision_prompt" in scenario

        engine._auto_manage_pitcher(game)
        pa = engine.simulate_plate_appearance(game)
        engine.apply_pa_result(game, pa)
        pa_count += 1

    assert pa_count > 0
    print(f"  test_decision_points_each_pa: PASSED ({pa_count} PAs with scenarios)")


# ===========================================================================
# Test: dry-run passes
# ===========================================================================

def test_dry_run_imports():
    """Verify game.py core functions import and scenario building works."""
    # The game.py module requires anthropic for import, so we test the
    # scenario conversion and decision validation here instead.
    from simulation import game_state_to_scenario, validate_and_apply_decision, DecisionResult

    engine = SimulationEngine(seed=42)
    rosters = load_rosters()
    game = engine.initialize_game(rosters)

    # Build scenario from game state
    scenario = game_state_to_scenario(game, "home")
    assert "matchup_state" in scenario
    assert "roster_state" in scenario
    assert "decision_prompt" in scenario

    # Validate a decision
    result = validate_and_apply_decision(
        game,
        {"decision": "NO_ACTION", "action_details": "test"},
        "home",
        engine,
    )
    assert result.valid
    assert isinstance(result, DecisionResult)

    print("  test_dry_run_imports: PASSED")


# ===========================================================================
# Test: Scenario prompt reflects batting/fielding correctly
# ===========================================================================

def test_scenario_batting_vs_fielding_prompt():
    """Decision prompt should reflect whether team is batting or fielding."""
    engine, game = make_test_game()

    # Top of 1st: away bats, home fields
    home_scenario = game_state_to_scenario(game, "home")
    assert "FIELDING" in home_scenario["decision_prompt"]

    away_scenario = game_state_to_scenario(game, "away")
    assert "BATTING" in away_scenario["decision_prompt"]

    # Switch to bottom
    game.half = "BOTTOM"
    home_scenario_bot = game_state_to_scenario(game, "home")
    assert "BATTING" in home_scenario_bot["decision_prompt"]

    away_scenario_bot = game_state_to_scenario(game, "away")
    assert "FIELDING" in away_scenario_bot["decision_prompt"]

    print("  test_scenario_batting_vs_fielding_prompt: PASSED")


# ===========================================================================
# Main test runner
# ===========================================================================

def run_all_tests():
    print("=" * 72)
    print("AGENT DECISION LOOP TESTS")
    print("=" * 72)

    tests = [
        ("Scenario Conversion -- Basic", [
            test_scenario_has_required_keys,
            test_scenario_matchup_state_fields,
            test_scenario_roster_state_fields,
            test_scenario_opponent_roster_fields,
            test_scenario_decision_prompt_is_descriptive,
            test_scenario_serializes_to_json,
        ]),
        ("Scenario Conversion -- Runners", [
            test_scenario_with_runners,
        ]),
        ("Scenario Conversion -- Perspective", [
            test_scenario_away_team_perspective,
            test_scenario_batting_vs_fielding_prompt,
        ]),
        ("Decision Validation -- NO_ACTION", [
            test_no_action_is_always_valid,
            test_empty_decision_is_no_action,
            test_unknown_decision_treated_as_no_action,
        ]),
        ("Decision Validation -- PITCHING_CHANGE", [
            test_pitching_change_while_batting_is_invalid,
            test_pitching_change_3_batter_minimum,
            test_pitching_change_after_3_batters,
            test_pitching_change_by_name,
            test_pitching_change_used_pitcher_rejected,
        ]),
        ("Decision Validation -- PINCH_HIT", [
            test_pinch_hit_while_fielding_is_invalid,
            test_pinch_hit_while_batting,
            test_pinch_hit_removed_player_rejected,
        ]),
        ("Decision Validation -- STOLEN_BASE", [
            test_stolen_base_while_fielding_is_invalid,
            test_stolen_base_no_runner,
            test_stolen_base_with_runner,
        ]),
        ("Decision Validation -- INTENTIONAL_WALK", [
            test_ibb_while_batting_is_invalid,
            test_ibb_while_fielding,
        ]),
        ("Decision Validation -- DEFENSE", [
            test_defensive_positioning,
            test_defensive_positioning_while_batting_invalid,
        ]),
        ("Decision Validation -- MOUND_VISIT", [
            test_mound_visit,
            test_mound_visit_none_remaining,
        ]),
        ("Decision Validation -- BUNT", [
            test_sacrifice_bunt,
            test_sacrifice_bunt_while_fielding_invalid,
        ]),
        ("Player ID Extraction", [
            test_extract_player_id_by_id,
            test_extract_player_id_by_name,
            test_extract_player_id_not_found,
        ]),
        ("State Integrity", [
            test_invalid_decision_does_not_modify_state,
        ]),
        ("Scenario Updates", [
            test_scenario_updates_with_game_state,
        ]),
        ("Decision Points", [
            test_decision_points_each_pa,
        ]),
        ("Decision Log", [
            test_decision_log_structure,
        ]),
        ("Full Game with Decisions", [
            test_full_game_with_no_action_decisions,
            test_full_game_with_mixed_decisions,
        ]),
        ("Integration", [
            test_dry_run_imports,
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
