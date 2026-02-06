# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "pytest"]
# ///
"""Comprehensive tests for game_state_ingestion module.

Tests cover all 9 steps from the features.json specification:
1. Accept raw game state JSON (MLB API or intermediate format)
2. Parse into MatchupState, RosterState, OpponentRosterState
3. Map MLB Stats API player IDs to model fields
4. Populate pitcher in-game stats from live feed
5. Populate runner data with sprint speed / SB rate
6. Populate bullpen availability and freshness
7. Populate bench availability from substitution history
8. Track mound visits remaining and replay challenge availability
9. Validate parsed models and return clear errors for invalid data
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    MatchupState,
    RosterState,
    OpponentRosterState,
    Runner,
    BullpenPitcher,
    BenchPlayer,
    LineupPlayer,
)
from game_state_ingestion import (
    detect_format,
    ingest_game_state,
    ingest_intermediate,
    ingest_mlb_api_feed,
    ingest_from_file,
    IngestionError,
    IngestionValidationError,
    _compute_times_through_order,
    _extract_half,
    _extract_batting_team,
    _safe_float,
    _safe_int,
    _extract_runner,
    _get_person,
)
from simulation import (
    load_rosters,
    SimPlayer,
    TeamState,
    GameState,
    SimulationEngine,
    game_state_to_scenario,
    BaseRunner,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _build_minimal_intermediate() -> dict:
    """Build a minimal valid intermediate-format payload."""
    return {
        "matchup_state": {
            "inning": 5,
            "half": "TOP",
            "outs": 1,
            "count": {"balls": 2, "strikes": 1},
            "runners": {},
            "score": {"home": 3, "away": 2},
            "batting_team": "AWAY",
            "batter": {
                "player_id": "12345",
                "name": "Mike Trout",
                "bats": "R",
                "lineup_position": 2,
            },
            "pitcher": {
                "player_id": "67890",
                "name": "Gerrit Cole",
                "throws": "R",
                "pitch_count_today": 78,
                "batters_faced_today": 18,
                "times_through_order": 2,
                "innings_pitched_today": 4.2,
                "runs_allowed_today": 2,
                "today_line": {
                    "IP": 4.2, "H": 5, "R": 2, "ER": 2, "BB": 1, "K": 7,
                },
            },
            "on_deck_batter": {
                "player_id": "11111",
                "name": "Shohei Ohtani",
                "bats": "L",
            },
        },
        "roster_state": {
            "our_lineup": [
                {"player_id": f"p{i}", "name": f"Player {i}",
                 "position": pos, "bats": "R", "in_game": True}
                for i, pos in enumerate(
                    ["CF", "SS", "DH", "1B", "RF", "3B", "LF", "C", "2B"], 1
                )
            ],
            "our_lineup_position": 3,
            "bench": [
                {"player_id": "b1", "name": "Bench One", "bats": "L",
                 "positions": ["1B", "LF"], "available": True},
                {"player_id": "b2", "name": "Bench Two", "bats": "R",
                 "positions": ["C", "3B"], "available": True},
            ],
            "bullpen": [
                {"player_id": "bp1", "name": "Reliever A", "throws": "R",
                 "role": "SETUP", "available": True, "freshness": "FRESH",
                 "pitches_last_3_days": [0, 0, 0],
                 "days_since_last_appearance": 3, "is_warming_up": False},
                {"player_id": "bp2", "name": "Reliever B", "throws": "L",
                 "role": "CLOSER", "available": True, "freshness": "MODERATE",
                 "pitches_last_3_days": [25, 0, 0],
                 "days_since_last_appearance": 1, "is_warming_up": False},
            ],
            "mound_visits_remaining": 4,
            "challenge_available": True,
        },
        "opponent_roster_state": {
            "their_lineup": [
                {"player_id": f"op{i}", "name": f"Opp Player {i}",
                 "position": pos, "bats": "R", "in_game": True}
                for i, pos in enumerate(
                    ["CF", "2B", "1B", "DH", "RF", "SS", "3B", "LF", "C"], 1
                )
            ],
            "their_lineup_position": 5,
            "their_bench": [
                {"player_id": "ob1", "name": "Opp Bench", "bats": "L",
                 "available": True},
            ],
            "their_bullpen": [
                {"player_id": "obp1", "name": "Opp Reliever", "throws": "R",
                 "role": "CLOSER", "available": True, "freshness": "FRESH"},
            ],
        },
    }


def _build_intermediate_with_runners() -> dict:
    """Build an intermediate payload with runners on base."""
    payload = _build_minimal_intermediate()
    payload["matchup_state"]["runners"] = {
        "first": {
            "player_id": "r1",
            "name": "Speed Demon",
            "sprint_speed": 30.0,
            "sb_success_rate": 0.85,
        },
        "third": {
            "player_id": "r3",
            "name": "Slow Runner",
            "sprint_speed": 25.0,
            "sb_success_rate": 0.55,
        },
    }
    return payload


def _build_mlb_api_feed() -> dict:
    """Build a minimal realistic MLB Stats API live game feed."""
    return {
        "gameData": {
            "status": {"abstractGameState": "Live"},
            "teams": {
                "home": {
                    "id": 111,
                    "name": "Boston Red Sox",
                },
                "away": {
                    "id": 147,
                    "name": "New York Yankees",
                },
            },
            "players": {
                "ID660271": {
                    "id": 660271,
                    "fullName": "Shohei Ohtani",
                    "batSide": {"code": "L"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "DH"},
                },
                "ID545361": {
                    "id": 545361,
                    "fullName": "Mike Trout",
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "CF"},
                },
                "ID543037": {
                    "id": 543037,
                    "fullName": "Gerrit Cole",
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "P"},
                },
                "ID605141": {
                    "id": 605141,
                    "fullName": "Mookie Betts",
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "SS"},
                },
                "ID592450": {
                    "id": 592450,
                    "fullName": "Aaron Judge",
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "RF"},
                },
                "ID608369": {
                    "id": 608369,
                    "fullName": "Rafael Devers",
                    "batSide": {"code": "L"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "3B"},
                },
                "ID666182": {
                    "id": 666182,
                    "fullName": "Julio Rodriguez",
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "CF"},
                },
                "ID571578": {
                    "id": 571578,
                    "fullName": "Clay Holmes",
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "P"},
                },
                "ID622663": {
                    "id": 622663,
                    "fullName": "Bench Guy",
                    "batSide": {"code": "L"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "1B"},
                },
            },
        },
        "liveData": {
            "linescore": {
                "currentInning": 6,
                "isTopInning": True,
                "outs": 1,
                "teams": {
                    "home": {"runs": 3, "hits": 7, "errors": 0},
                    "away": {"runs": 4, "hits": 8, "errors": 1},
                },
                "offense": {
                    "batter": {"id": 545361, "fullName": "Mike Trout"},
                    "ondeck": {"id": 660271, "fullName": "Shohei Ohtani"},
                    "first": {"id": 592450, "fullName": "Aaron Judge"},
                },
            },
            "plays": {
                "currentPlay": {
                    "matchup": {
                        "batter": {"id": 545361, "fullName": "Mike Trout"},
                        "pitcher": {"id": 543037, "fullName": "Gerrit Cole"},
                    },
                    "count": {"balls": 1, "strikes": 2, "outs": 1},
                },
            },
            "boxscore": {
                "teams": {
                    "home": {
                        "battingOrder": [
                            605141, 608369, 545361, 660271,
                            666182, 622663, 571578, 592450, 543037,
                        ],
                        "players": {
                            "ID605141": {
                                "person": {"fullName": "Mookie Betts"},
                                "position": {"abbreviation": "SS"},
                            },
                            "ID608369": {
                                "person": {"fullName": "Rafael Devers"},
                                "position": {"abbreviation": "3B"},
                            },
                            "ID545361": {
                                "person": {"fullName": "Mike Trout"},
                                "position": {"abbreviation": "CF"},
                            },
                            "ID660271": {
                                "person": {"fullName": "Shohei Ohtani"},
                                "position": {"abbreviation": "DH"},
                            },
                            "ID666182": {
                                "person": {"fullName": "Julio Rodriguez"},
                                "position": {"abbreviation": "CF"},
                            },
                            "ID622663": {
                                "person": {"fullName": "Bench Guy"},
                                "position": {"abbreviation": "1B"},
                            },
                            "ID571578": {
                                "person": {"fullName": "Clay Holmes"},
                                "position": {"abbreviation": "P"},
                            },
                            "ID592450": {
                                "person": {"fullName": "Aaron Judge"},
                                "position": {"abbreviation": "RF"},
                            },
                            "ID543037": {
                                "person": {"fullName": "Gerrit Cole"},
                                "position": {"abbreviation": "P"},
                                "stats": {
                                    "pitching": {
                                        "inningsPitched": "5.1",
                                        "hits": 8,
                                        "runs": 4,
                                        "earnedRuns": 3,
                                        "baseOnBalls": 2,
                                        "strikeOuts": 6,
                                        "numberOfPitches": 92,
                                        "battersFaced": 23,
                                        "homeRuns": 1,
                                    }
                                },
                            },
                        },
                        "pitchers": [543037],
                        "bullpen": [571578],
                        "bench": [622663],
                    },
                    "away": {
                        "battingOrder": [
                            592450, 545361, 660271, 608369,
                            666182, 622663, 571578, 605141, 543037,
                        ],
                        "players": {
                            "ID592450": {
                                "person": {"fullName": "Aaron Judge"},
                                "position": {"abbreviation": "RF"},
                            },
                            "ID545361": {
                                "person": {"fullName": "Mike Trout"},
                                "position": {"abbreviation": "CF"},
                            },
                            "ID660271": {
                                "person": {"fullName": "Shohei Ohtani"},
                                "position": {"abbreviation": "DH"},
                            },
                            "ID608369": {
                                "person": {"fullName": "Rafael Devers"},
                                "position": {"abbreviation": "3B"},
                            },
                            "ID666182": {
                                "person": {"fullName": "Julio Rodriguez"},
                                "position": {"abbreviation": "CF"},
                            },
                            "ID622663": {
                                "person": {"fullName": "Bench Guy"},
                                "position": {"abbreviation": "1B"},
                            },
                            "ID571578": {
                                "person": {"fullName": "Clay Holmes"},
                                "position": {"abbreviation": "P"},
                            },
                            "ID605141": {
                                "person": {"fullName": "Mookie Betts"},
                                "position": {"abbreviation": "SS"},
                            },
                            "ID543037": {
                                "person": {"fullName": "Gerrit Cole"},
                                "position": {"abbreviation": "P"},
                            },
                        },
                        "pitchers": [543037],
                        "bullpen": [571578],
                        "bench": [622663],
                    },
                },
            },
        },
    }


def _advance_one_pa(engine: SimulationEngine, game_state: GameState) -> None:
    """Advance the game by one plate appearance."""
    pa = engine.simulate_plate_appearance(game_state)
    engine.apply_pa_result(game_state, pa)


def _build_simulation_game_state() -> tuple[GameState, SimulationEngine]:
    """Build a GameState from sample rosters for testing."""
    rosters = load_rosters()
    engine = SimulationEngine(seed=42)
    game_state = engine.initialize_game(rosters)
    return game_state, engine


# ===================================================================
# Step 1: Accept raw game state JSON (MLB API or intermediate format)
# ===================================================================

class TestStep1AcceptRawPayload:
    """Accept a raw game state JSON payload in either format."""

    def test_accepts_intermediate_format_dict(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        assert "matchup_state" in result
        assert "roster_state" in result
        assert "opponent_roster_state" in result

    def test_accepts_intermediate_format_json_string(self):
        payload = _build_minimal_intermediate()
        json_str = json.dumps(payload)
        result = ingest_game_state(json_str)
        assert "matchup_state" in result
        assert "roster_state" in result
        assert "opponent_roster_state" in result

    def test_accepts_mlb_api_format(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed, managed_team_side="home")
        assert "matchup_state" in result
        assert "roster_state" in result
        assert "opponent_roster_state" in result

    def test_detect_format_mlb_api(self):
        feed = _build_mlb_api_feed()
        assert detect_format(feed) == "mlb_api"

    def test_detect_format_intermediate(self):
        payload = _build_minimal_intermediate()
        assert detect_format(payload) == "intermediate"

    def test_detect_format_unknown(self):
        assert detect_format({"random": "data"}) == "unknown"

    def test_rejects_unknown_format(self):
        try:
            ingest_game_state({"random": "data"})
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "Unrecognized" in str(e)

    def test_rejects_invalid_json_string(self):
        try:
            ingest_game_state("not valid json {{{")
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "Invalid JSON" in str(e)

    def test_rejects_non_dict_payload(self):
        try:
            ingest_game_state([1, 2, 3])
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "must be a dict" in str(e)

    def test_intermediate_from_simulation(self):
        """Verify that simulation's game_state_to_scenario output is accepted."""
        game_state, engine = _build_simulation_game_state()
        # Advance the game a bit to have more interesting state
        _advance_one_pa(engine, game_state)
        scenario = game_state_to_scenario(game_state, "home")
        result = ingest_game_state(scenario)
        assert isinstance(result["matchup_state"], MatchupState)


# ==============================================================
# Step 2: Parse into MatchupState, RosterState, OpponentRosterState
# ==============================================================

class TestStep2ParseIntoModels:
    """Parse the payload into the three Pydantic models."""

    def test_matchup_state_is_pydantic_model(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        assert isinstance(result["matchup_state"], MatchupState)

    def test_roster_state_is_pydantic_model(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        assert isinstance(result["roster_state"], RosterState)

    def test_opponent_roster_state_is_pydantic_model(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        assert isinstance(result["opponent_roster_state"], OpponentRosterState)

    def test_matchup_state_fields_correct(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.inning == 5
        assert ms.half.value == "TOP"
        assert ms.outs == 1
        assert ms.count.balls == 2
        assert ms.count.strikes == 1
        assert ms.score.home == 3
        assert ms.score.away == 2

    def test_roster_state_has_lineup(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert len(rs.our_lineup) == 9
        assert rs.our_lineup_position == 3

    def test_opponent_roster_has_lineup(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ors = result["opponent_roster_state"]
        assert len(ors.their_lineup) == 9
        assert ors.their_lineup_position == 5

    def test_mlb_api_produces_valid_models(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        assert isinstance(result["matchup_state"], MatchupState)
        assert isinstance(result["roster_state"], RosterState)
        assert isinstance(result["opponent_roster_state"], OpponentRosterState)

    def test_matchup_state_batter_info(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.batter.player_id == "12345"
        assert ms.batter.name == "Mike Trout"
        assert ms.batter.bats.value == "R"
        assert ms.batter.lineup_position == 2

    def test_matchup_state_pitcher_info(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.pitcher.player_id == "67890"
        assert ms.pitcher.name == "Gerrit Cole"
        assert ms.pitcher.throws.value == "R"

    def test_matchup_state_on_deck_batter(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.on_deck_batter.player_id == "11111"
        assert ms.on_deck_batter.name == "Shohei Ohtani"
        assert ms.on_deck_batter.bats.value == "L"

    def test_batting_team_parsed(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.batting_team.value == "AWAY"


# ==========================================================
# Step 3: Map MLB Stats API player IDs to model fields
# ==========================================================

class TestStep3MapPlayerIDs:
    """Map MLB Stats API player IDs to agent model fields."""

    def test_mlb_api_batter_id_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.batter.player_id == "545361"
        assert ms.batter.name == "Mike Trout"

    def test_mlb_api_pitcher_id_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.pitcher.player_id == "543037"
        assert ms.pitcher.name == "Gerrit Cole"

    def test_mlb_api_on_deck_id_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.on_deck_batter.player_id == "660271"
        assert ms.on_deck_batter.name == "Shohei Ohtani"

    def test_mlb_api_batter_handedness_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.batter.bats.value == "R"  # Trout bats R

    def test_mlb_api_pitcher_handedness_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.pitcher.throws.value == "R"  # Cole throws R

    def test_mlb_api_lineup_player_ids_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed, managed_team_side="home")
        rs = result["roster_state"]
        # Home lineup is from boxscore batting order
        assert any(p.player_id == "605141" for p in rs.our_lineup)

    def test_mlb_api_bench_player_ids_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed, managed_team_side="home")
        rs = result["roster_state"]
        assert any(p.player_id == "622663" for p in rs.bench)

    def test_mlb_api_bullpen_player_ids_mapped(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed, managed_team_side="home")
        rs = result["roster_state"]
        assert any(p.player_id == "571578" for p in rs.bullpen)

    def test_player_ids_are_strings(self):
        """Player IDs should be strings in the models, even from numeric API IDs."""
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert isinstance(ms.batter.player_id, str)
        assert isinstance(ms.pitcher.player_id, str)


# ================================================================
# Step 4: Populate pitcher in-game stats from live feed
# ================================================================

class TestStep4PitcherInGameStats:
    """Populate pitcher in-game stats from the live game feed."""

    def test_pitch_count_populated(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.pitcher.pitch_count_today == 92

    def test_batters_faced_populated(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.pitcher.batters_faced_today == 23

    def test_innings_pitched_populated(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.pitcher.innings_pitched_today == 5.1

    def test_runs_allowed_populated(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.pitcher.runs_allowed_today == 4

    def test_times_through_order_computed(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        # 23 batters faced -> 3rd time through (23-1)//9 + 1 = 3
        assert ms.pitcher.times_through_order == 3

    def test_today_line_populated(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        line = ms.pitcher.today_line
        assert line["IP"] == 5.1
        assert line["H"] == 8
        assert line["R"] == 4
        assert line["ER"] == 3
        assert line["BB"] == 2
        assert line["K"] == 6

    def test_pitcher_stats_from_intermediate(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.pitcher.pitch_count_today == 78
        assert ms.pitcher.batters_faced_today == 18
        assert ms.pitcher.times_through_order == 2
        assert ms.pitcher.innings_pitched_today == 4.2
        assert ms.pitcher.runs_allowed_today == 2

    def test_pitcher_stats_default_when_not_in_boxscore(self):
        """When pitcher stats aren't available, defaults are used."""
        feed = _build_mlb_api_feed()
        # Remove the pitcher's stats from boxscore
        home_players = feed["liveData"]["boxscore"]["teams"]["home"]["players"]
        del home_players["ID543037"]["stats"]
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.pitcher.pitch_count_today == 0
        assert ms.pitcher.batters_faced_today == 0

    def test_times_through_order_helper(self):
        assert _compute_times_through_order(0) == 1
        assert _compute_times_through_order(1) == 1
        assert _compute_times_through_order(9) == 1
        assert _compute_times_through_order(10) == 2
        assert _compute_times_through_order(18) == 2
        assert _compute_times_through_order(19) == 3
        assert _compute_times_through_order(27) == 3
        assert _compute_times_through_order(28) == 4


# ================================================================
# Step 5: Populate runner data with Statcast cross-reference
# ================================================================

class TestStep5RunnerData:
    """Populate runner data (sprint speed, SB success rate)."""

    def test_runners_populated_from_intermediate(self):
        payload = _build_intermediate_with_runners()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.runners.first is not None
        assert ms.runners.first.player_id == "r1"
        assert ms.runners.first.name == "Speed Demon"
        assert ms.runners.first.sprint_speed == 30.0
        assert ms.runners.first.sb_success_rate == 0.85

    def test_third_base_runner_populated(self):
        payload = _build_intermediate_with_runners()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.runners.third is not None
        assert ms.runners.third.player_id == "r3"
        assert ms.runners.third.sprint_speed == 25.0

    def test_empty_base_is_none(self):
        payload = _build_intermediate_with_runners()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.runners.second is None

    def test_no_runners_all_none(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ms = result["matchup_state"]
        assert ms.runners.first is None
        assert ms.runners.second is None
        assert ms.runners.third is None

    def test_mlb_api_runner_on_first(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.runners.first is not None
        assert ms.runners.first.player_id == "592450"
        assert ms.runners.first.name == "Aaron Judge"

    def test_mlb_api_runner_has_defaults(self):
        """When Statcast data isn't available, runners get defaults."""
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        runner = ms.runners.first
        assert runner is not None
        # Should have league-average defaults
        assert isinstance(runner.sprint_speed, float)
        assert runner.sprint_speed > 0
        assert isinstance(runner.sb_success_rate, float)
        assert 0.0 < runner.sb_success_rate <= 1.0

    def test_mlb_api_no_runner_on_second(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        ms = result["matchup_state"]
        assert ms.runners.second is None

    def test_extract_runner_helper_returns_none(self):
        assert _extract_runner(None, {}) is None

    def test_extract_runner_helper_returns_dict(self):
        runner_data = {"id": 123, "fullName": "Test Player"}
        result = _extract_runner(runner_data, {})
        assert result is not None
        assert result["player_id"] == "123"
        assert result["name"] == "Test Player"
        assert isinstance(result["sprint_speed"], float)
        assert isinstance(result["sb_success_rate"], float)

    def test_simulation_runners_preserved(self):
        """Runners from simulation game_state_to_scenario are preserved."""
        game_state, engine = _build_simulation_game_state()
        # Manually add a runner
        bt = game_state.batting_team()
        runner_player = bt.lineup[3]
        game_state.runners = [BaseRunner(player=runner_player, start_base=2)]

        scenario = game_state_to_scenario(game_state, "home")
        result = ingest_game_state(scenario)
        ms = result["matchup_state"]
        assert ms.runners.second is not None
        assert ms.runners.second.player_id == runner_player.player_id


# ================================================================
# Step 6: Populate bullpen availability and freshness
# ================================================================

class TestStep6BullpenAvailability:
    """Populate bullpen availability and freshness."""

    def test_bullpen_populated(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert len(rs.bullpen) == 2

    def test_bullpen_pitcher_fields(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        bp = rs.bullpen[0]
        assert bp.player_id == "bp1"
        assert bp.name == "Reliever A"
        assert bp.throws.value == "R"
        assert bp.role.value == "SETUP"
        assert bp.available is True
        assert bp.freshness.value == "FRESH"
        assert bp.is_warming_up is False

    def test_bullpen_freshness_states(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.bullpen[0].freshness.value == "FRESH"
        assert rs.bullpen[1].freshness.value == "MODERATE"

    def test_bullpen_pitch_counts(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.bullpen[0].pitches_last_3_days == [0, 0, 0]
        assert rs.bullpen[1].pitches_last_3_days == [25, 0, 0]

    def test_bullpen_days_since_last(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.bullpen[0].days_since_last_appearance == 3
        assert rs.bullpen[1].days_since_last_appearance == 1

    def test_mlb_api_bullpen_from_boxscore(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed, managed_team_side="home")
        rs = result["roster_state"]
        assert len(rs.bullpen) >= 1
        # Bullpen pitcher should have player_id as string
        for bp in rs.bullpen:
            assert isinstance(bp.player_id, str)
            assert isinstance(bp.name, str)

    def test_mlb_api_used_pitcher_marked_unavailable(self):
        """Pitchers already used in the game should be marked unavailable."""
        feed = _build_mlb_api_feed()
        # In our fixture, pitcher 543037 is in pitchers list for home team
        # And also in bullpen list
        # Let's add the pitcher to the bullpen list as well
        feed_copy = copy.deepcopy(feed)
        feed_copy["liveData"]["boxscore"]["teams"]["home"]["bullpen"] = [571578, 543037]
        feed_copy["liveData"]["boxscore"]["teams"]["home"]["pitchers"] = [543037]
        result = ingest_game_state(feed_copy, managed_team_side="home")
        rs = result["roster_state"]
        for bp in rs.bullpen:
            if bp.player_id == "543037":
                assert bp.available is False
            elif bp.player_id == "571578":
                assert bp.available is True

    def test_bullpen_roster_override(self):
        """Roster overrides for bullpen should be used."""
        feed = _build_mlb_api_feed()
        custom_bullpen = [
            {"player_id": "999", "name": "Custom Reliever", "throws": "L",
             "role": "CLOSER", "available": True, "freshness": "TIRED",
             "pitches_last_3_days": [30, 20, 0],
             "days_since_last_appearance": 0, "is_warming_up": True},
        ]
        result = ingest_game_state(
            feed, managed_team_side="home",
            roster_overrides={"bullpen": custom_bullpen},
        )
        rs = result["roster_state"]
        assert len(rs.bullpen) == 1
        assert rs.bullpen[0].name == "Custom Reliever"
        assert rs.bullpen[0].freshness.value == "TIRED"
        assert rs.bullpen[0].is_warming_up is True

    def test_simulation_bullpen_preserved(self):
        """Bullpen from simulation scenario should be ingested correctly."""
        game_state, engine = _build_simulation_game_state()
        scenario = game_state_to_scenario(game_state, "home")
        result = ingest_game_state(scenario)
        rs = result["roster_state"]
        assert len(rs.bullpen) > 0
        for bp in rs.bullpen:
            assert isinstance(bp, BullpenPitcher)


# ================================================================
# Step 7: Populate bench availability from substitution history
# ================================================================

class TestStep7BenchAvailability:
    """Populate bench availability based on substitution history."""

    def test_bench_populated(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert len(rs.bench) == 2

    def test_bench_player_fields(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        bp = rs.bench[0]
        assert bp.player_id == "b1"
        assert bp.name == "Bench One"
        assert bp.bats.value == "L"
        assert bp.positions == ["1B", "LF"]
        assert bp.available is True

    def test_bench_unavailable_player(self):
        payload = _build_minimal_intermediate()
        payload["roster_state"]["bench"][1]["available"] = False
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.bench[1].available is False

    def test_mlb_api_bench_from_boxscore(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed, managed_team_side="home")
        rs = result["roster_state"]
        assert len(rs.bench) >= 1
        for bp in rs.bench:
            assert isinstance(bp.player_id, str)

    def test_bench_roster_override(self):
        feed = _build_mlb_api_feed()
        custom_bench = [
            {"player_id": "777", "name": "Custom Bench", "bats": "S",
             "positions": ["C", "1B"], "available": True},
        ]
        result = ingest_game_state(
            feed, managed_team_side="home",
            roster_overrides={"bench": custom_bench},
        )
        rs = result["roster_state"]
        assert len(rs.bench) == 1
        assert rs.bench[0].name == "Custom Bench"
        assert rs.bench[0].bats.value == "S"

    def test_opponent_bench_populated(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        ors = result["opponent_roster_state"]
        assert len(ors.their_bench) == 1
        assert ors.their_bench[0].player_id == "ob1"
        assert ors.their_bench[0].bats.value == "L"

    def test_simulation_bench_preserved(self):
        """Bench from simulation scenario should be ingested correctly."""
        game_state, engine = _build_simulation_game_state()
        scenario = game_state_to_scenario(game_state, "home")
        result = ingest_game_state(scenario)
        rs = result["roster_state"]
        assert len(rs.bench) > 0
        for bp in rs.bench:
            assert isinstance(bp, BenchPlayer)


# ================================================================
# Step 8: Track mound visits remaining and replay challenge
# ================================================================

class TestStep8MoundVisitsAndChallenges:
    """Track mound visits remaining and replay challenge availability."""

    def test_mound_visits_from_intermediate(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.mound_visits_remaining == 4

    def test_challenge_available_from_intermediate(self):
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.challenge_available is True

    def test_challenge_unavailable(self):
        payload = _build_minimal_intermediate()
        payload["roster_state"]["challenge_available"] = False
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.challenge_available is False

    def test_mound_visits_zero(self):
        payload = _build_minimal_intermediate()
        payload["roster_state"]["mound_visits_remaining"] = 0
        result = ingest_game_state(payload)
        rs = result["roster_state"]
        assert rs.mound_visits_remaining == 0

    def test_mlb_api_default_mound_visits(self):
        """MLB API feed doesn't have mound visit data; default is 5."""
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        rs = result["roster_state"]
        assert rs.mound_visits_remaining == 5

    def test_mlb_api_default_challenge(self):
        """MLB API feed doesn't have challenge data; default is True."""
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed)
        rs = result["roster_state"]
        assert rs.challenge_available is True

    def test_mlb_api_mound_visits_override(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(
            feed, roster_overrides={"mound_visits_remaining": 2},
        )
        rs = result["roster_state"]
        assert rs.mound_visits_remaining == 2

    def test_mlb_api_challenge_override(self):
        feed = _build_mlb_api_feed()
        result = ingest_game_state(
            feed, roster_overrides={"challenge_available": False},
        )
        rs = result["roster_state"]
        assert rs.challenge_available is False

    def test_simulation_mound_visits_preserved(self):
        game_state, engine = _build_simulation_game_state()
        game_state.home.mound_visits_remaining = 3
        game_state.home.challenge_available = False
        scenario = game_state_to_scenario(game_state, "home")
        result = ingest_game_state(scenario)
        rs = result["roster_state"]
        assert rs.mound_visits_remaining == 3
        assert rs.challenge_available is False


# ================================================================
# Step 9: Validate and return clear errors for invalid data
# ================================================================

class TestStep9ValidationErrors:
    """Validate parsed models and return clear errors for invalid data."""

    def test_missing_matchup_state_key(self):
        payload = _build_minimal_intermediate()
        del payload["matchup_state"]
        try:
            ingest_game_state(payload)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "matchup_state" in str(e)

    def test_missing_roster_state_key(self):
        payload = _build_minimal_intermediate()
        del payload["roster_state"]
        try:
            ingest_game_state(payload)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "roster_state" in str(e)

    def test_missing_opponent_roster_state_key(self):
        payload = _build_minimal_intermediate()
        del payload["opponent_roster_state"]
        try:
            ingest_game_state(payload)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "opponent_roster_state" in str(e)

    def test_invalid_inning_in_intermediate(self):
        payload = _build_minimal_intermediate()
        payload["matchup_state"]["inning"] = 0  # Invalid: must be >= 1
        try:
            ingest_game_state(payload)
            assert False, "Should have raised IngestionValidationError"
        except IngestionValidationError as e:
            assert len(e.validation_errors) > 0
            assert any("MatchupState" in err["model"] for err in e.validation_errors)

    def test_invalid_outs_in_intermediate(self):
        payload = _build_minimal_intermediate()
        payload["matchup_state"]["outs"] = 5  # Invalid: must be 0-2
        try:
            ingest_game_state(payload)
            assert False, "Should have raised IngestionValidationError"
        except IngestionValidationError as e:
            assert len(e.validation_errors) > 0

    def test_invalid_balls_in_count(self):
        payload = _build_minimal_intermediate()
        payload["matchup_state"]["count"]["balls"] = 5
        try:
            ingest_game_state(payload)
            assert False, "Should have raised IngestionValidationError"
        except IngestionValidationError as e:
            assert len(e.validation_errors) > 0

    def test_invalid_lineup_position(self):
        payload = _build_minimal_intermediate()
        payload["roster_state"]["our_lineup_position"] = 99
        try:
            ingest_game_state(payload)
            assert False, "Should have raised IngestionValidationError"
        except IngestionValidationError as e:
            assert len(e.validation_errors) > 0
            assert any("RosterState" in err["model"] for err in e.validation_errors)

    def test_mlb_api_missing_game_data(self):
        feed = {"liveData": {}}
        try:
            ingest_game_state(feed)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "gameData" in str(e) or "missing" in str(e).lower()

    def test_mlb_api_missing_live_data(self):
        feed = {"gameData": {}}
        try:
            ingest_game_state(feed)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "liveData" in str(e) or "missing" in str(e).lower()

    def test_mlb_api_no_current_inning(self):
        feed = _build_mlb_api_feed()
        feed["liveData"]["linescore"]["currentInning"] = None
        try:
            ingest_game_state(feed)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "inning" in str(e).lower()

    def test_mlb_api_no_batter(self):
        feed = _build_mlb_api_feed()
        del feed["liveData"]["plays"]["currentPlay"]["matchup"]["batter"]
        try:
            ingest_game_state(feed)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "batter" in str(e).lower()

    def test_mlb_api_no_pitcher(self):
        feed = _build_mlb_api_feed()
        del feed["liveData"]["plays"]["currentPlay"]["matchup"]["pitcher"]
        try:
            ingest_game_state(feed)
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "pitcher" in str(e).lower()

    def test_mlb_api_invalid_managed_team_side(self):
        feed = _build_mlb_api_feed()
        try:
            ingest_game_state(feed, managed_team_side="neutral")
            assert False, "Should have raised IngestionError"
        except IngestionError as e:
            assert "managed_team_side" in str(e)

    def test_validation_error_has_details(self):
        payload = _build_minimal_intermediate()
        payload["matchup_state"]["inning"] = -5
        try:
            ingest_game_state(payload)
            assert False, "Should have raised"
        except IngestionValidationError as e:
            assert e.validation_errors is not None
            assert len(e.validation_errors) > 0
            assert "model" in e.validation_errors[0]

    def test_multiple_validation_errors(self):
        payload = _build_minimal_intermediate()
        payload["matchup_state"]["inning"] = -1
        payload["matchup_state"]["outs"] = 99
        try:
            ingest_game_state(payload)
            assert False, "Should have raised"
        except IngestionValidationError as e:
            assert len(e.validation_errors) >= 2

    def test_error_has_details_list(self):
        """IngestionError should have a details list for debugging."""
        # Use ingest_intermediate directly so format detection doesn't
        # reject it first.
        payload = _build_minimal_intermediate()
        del payload["matchup_state"]
        del payload["roster_state"]
        try:
            ingest_intermediate(payload)
            assert False
        except IngestionError as e:
            assert isinstance(e.details, list)
            assert len(e.details) >= 2


# ================================================================
# Helper function tests
# ================================================================

class TestHelpers:
    """Tests for internal helper functions."""

    def test_extract_half_top(self):
        assert _extract_half(True) == "TOP"

    def test_extract_half_bottom(self):
        assert _extract_half(False) == "BOTTOM"

    def test_extract_batting_team_top(self):
        assert _extract_batting_team("TOP") == "AWAY"

    def test_extract_batting_team_bottom(self):
        assert _extract_batting_team("BOTTOM") == "HOME"

    def test_safe_float_valid(self):
        assert _safe_float("3.14") == 3.14
        assert _safe_float(42) == 42.0
        assert _safe_float(0.0) == 0.0

    def test_safe_float_invalid(self):
        assert _safe_float(None) == 0.0
        assert _safe_float("not_a_number") == 0.0
        assert _safe_float(None, 5.0) == 5.0

    def test_safe_int_valid(self):
        assert _safe_int("7") == 7
        assert _safe_int(42) == 42
        assert _safe_int(0) == 0

    def test_safe_int_invalid(self):
        assert _safe_int(None) == 0
        assert _safe_int("bad") == 0
        assert _safe_int(None, 10) == 10

    def test_get_person_found(self):
        game_data = {
            "players": {
                "ID12345": {"fullName": "Test Player", "batSide": {"code": "L"}},
            }
        }
        p = _get_person(game_data, 12345)
        assert p["fullName"] == "Test Player"

    def test_get_person_not_found(self):
        p = _get_person({"players": {}}, 99999)
        assert p == {}

    def test_get_person_none_id(self):
        p = _get_person({"players": {}}, None)
        assert p == {}


# ================================================================
# Integration tests
# ================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_intermediate_round_trip(self):
        """Build intermediate -> ingest -> validate all models."""
        payload = _build_intermediate_with_runners()
        result = ingest_game_state(payload)

        ms = result["matchup_state"]
        rs = result["roster_state"]
        ors = result["opponent_roster_state"]

        # MatchupState
        assert ms.inning == 5
        assert ms.half.value == "TOP"
        assert ms.outs == 1
        assert ms.count.balls == 2
        assert ms.count.strikes == 1
        assert ms.runners.first is not None
        assert ms.runners.third is not None
        assert ms.runners.second is None
        assert ms.score.home == 3
        assert ms.score.away == 2

        # RosterState
        assert len(rs.our_lineup) == 9
        assert len(rs.bench) == 2
        assert len(rs.bullpen) == 2
        assert rs.mound_visits_remaining == 4
        assert rs.challenge_available is True

        # OpponentRosterState
        assert len(ors.their_lineup) == 9
        assert len(ors.their_bench) == 1
        assert len(ors.their_bullpen) == 1

    def test_full_mlb_api_round_trip(self):
        """Build MLB API feed -> ingest -> validate all models."""
        feed = _build_mlb_api_feed()
        result = ingest_game_state(feed, managed_team_side="home")

        ms = result["matchup_state"]
        rs = result["roster_state"]
        ors = result["opponent_roster_state"]

        # MatchupState
        assert ms.inning == 6
        assert ms.half.value == "TOP"
        assert ms.outs == 1
        assert ms.score.home == 3
        assert ms.score.away == 4
        assert ms.batter.name == "Mike Trout"
        assert ms.pitcher.name == "Gerrit Cole"
        assert ms.on_deck_batter.name == "Shohei Ohtani"
        assert ms.runners.first is not None
        assert ms.runners.first.name == "Aaron Judge"

        # Pitcher stats
        assert ms.pitcher.pitch_count_today == 92
        assert ms.pitcher.innings_pitched_today == 5.1

        # RosterState
        assert len(rs.our_lineup) > 0
        assert rs.mound_visits_remaining == 5
        assert rs.challenge_available is True

        # OpponentRosterState
        assert len(ors.their_lineup) > 0

    def test_simulation_scenario_round_trip(self):
        """Simulation -> game_state_to_scenario -> ingest -> valid models."""
        game_state, engine = _build_simulation_game_state()

        # Advance a few at-bats
        for _ in range(5):
            _advance_one_pa(engine, game_state)

        scenario = game_state_to_scenario(game_state, "home")
        result = ingest_game_state(scenario)

        ms = result["matchup_state"]
        rs = result["roster_state"]
        ors = result["opponent_roster_state"]

        assert isinstance(ms, MatchupState)
        assert isinstance(rs, RosterState)
        assert isinstance(ors, OpponentRosterState)

        # Basic sanity checks
        assert ms.inning >= 1
        assert ms.half.value in ("TOP", "BOTTOM")
        assert 0 <= ms.outs <= 2
        assert ms.batter.player_id != ""
        assert ms.pitcher.player_id != ""
        assert len(rs.our_lineup) == 9
        assert len(ors.their_lineup) == 9

    def test_away_team_perspective(self):
        """Ingesting from away team perspective swaps our/their rosters."""
        feed = _build_mlb_api_feed()

        # Make away team have a distinct batting order with unique first player
        feed["liveData"]["boxscore"]["teams"]["away"]["battingOrder"] = [
            592450, 660271, 545361, 608369,
            666182, 622663, 571578, 605141, 543037,
        ]
        # Make home team have a different first player
        feed["liveData"]["boxscore"]["teams"]["home"]["battingOrder"] = [
            605141, 608369, 545361, 660271,
            666182, 622663, 571578, 592450, 543037,
        ]

        result_home = ingest_game_state(feed, managed_team_side="home")
        result_away = ingest_game_state(feed, managed_team_side="away")

        # The first player in our lineup should differ based on perspective
        home_first = result_home["roster_state"].our_lineup[0].player_id
        away_first = result_away["roster_state"].our_lineup[0].player_id
        assert home_first != away_first

        # Home managing -> home lineup is ours -> first player is 605141
        assert home_first == "605141"
        # Away managing -> away lineup is ours -> first player is 592450
        assert away_first == "592450"

    def test_json_serialization_round_trip(self):
        """Ingest -> serialize to JSON -> ingest again."""
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)

        # Serialize models to dicts
        re_serialized = {
            "matchup_state": result["matchup_state"].model_dump(),
            "roster_state": result["roster_state"].model_dump(),
            "opponent_roster_state": result["opponent_roster_state"].model_dump(),
        }

        # Should be able to ingest again
        result2 = ingest_game_state(re_serialized)
        assert isinstance(result2["matchup_state"], MatchupState)
        assert result2["matchup_state"].inning == result["matchup_state"].inning

    def test_ingest_from_file(self, tmp_path):
        """Test the file ingestion convenience function."""
        payload = _build_minimal_intermediate()
        file_path = tmp_path / "game_state.json"
        with open(file_path, "w") as f:
            json.dump(payload, f)

        result = ingest_from_file(file_path)
        assert isinstance(result["matchup_state"], MatchupState)

    def test_ingest_from_file_not_found(self):
        try:
            ingest_from_file("/nonexistent/path/game.json")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_all_lineup_positions_valid(self):
        """Every lineup player should have valid fields."""
        payload = _build_minimal_intermediate()
        result = ingest_game_state(payload)
        for p in result["roster_state"].our_lineup:
            assert isinstance(p, LineupPlayer)
            assert p.player_id != ""
            assert p.name != ""
            assert p.position != ""

    def test_full_simulation_game_all_states_ingestible(self):
        """Run a simulation and verify every decision point produces ingestible state."""
        game_state, engine = _build_simulation_game_state()
        count = 0
        while not game_state.game_over and count < 100:
            scenario = game_state_to_scenario(game_state, "home")
            result = ingest_game_state(scenario)
            assert isinstance(result["matchup_state"], MatchupState)
            assert isinstance(result["roster_state"], RosterState)
            assert isinstance(result["opponent_roster_state"], OpponentRosterState)
            _advance_one_pa(engine, game_state)
            count += 1
        assert count > 0, "Should have processed at least one at-bat"
